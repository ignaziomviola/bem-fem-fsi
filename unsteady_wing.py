"""Time-dependent driver for the thick-wing source-doublet panel method.

The physics lives in thick_panel_wing.py and is not duplicated here: this
module owns the time step, the prescribed rigid motions, the history bundle and
the unsteady figures. Every fluid-dynamic call it makes - init_state,
update_points, solve, wake_node_velocity, shed, truncate, get_loads - is the
same block the steady driver calls, with the same signature.

Physics and verification: docs/UNSTEADY.md. Steady formulation: docs/DESIGN.md.
Block contracts: docs/ARCHITECTURE.md. Usage: README.md.

Frame and conventions (they follow the steady code and pin every sign):

- the body moves through an inertial frame in which the onset U(z) x_hat is
  fixed, so a wing advancing into still fluid and a wing held in a stream are
  both expressible and both are exact;
- u_rel = U_onset(c) - u_body(c) at every collocation point, so sigma =
  -n.u_rel enforces tangency on the MOVING surface;
- the wake lives in the inertial frame and its nodes are material points of the
  fluid, advected by onset + source + ring velocities, which are absolute; body
  motion reaches the wake only through the strengths;
- the newest wake strip carries the current trailing-edge doublet jump and
  every older strip keeps the strength it was shed with (n_bound = 1). That is
  Kelvin's theorem discretely, and it is why the wake must not be relaxed;
- the pressure is the unsteady Bernoulli relation in the body frame,
  p = 0.5 rho (|u_rel|^2 - |V|^2) - rho dmu/dt,
  with dmu/dt following the material panel. The added mass is carried entirely
  by that term; there is no separate added-mass model.

Library use (no prompts, prints or figures outside main()):
    import unsteady_wing as usw
    motion = usw.heave(points, amplitude=0.05, omega=2.0)
    history, state = usw.time_march(points, onset, motion, dt=0.05, nsteps=200)
    history["CL"], history["t"]
"""

import warnings

import numpy as np

import thick_panel_wing as tpw
from freewake_kernels import (load_velocity_profile, make_onset, pitch_mesh,
                              prompt)

# ------------------------------------------------------------- 1 constants

DPHI_DT_ORDER = 2             # backward difference order for dmu/dt
NMAX_DEFAULT = None           # wake rows kept; None keeps the whole history
STEP_WARN_CHORDS = 0.5        # warn above this convection distance per step


# --------------------------------------------------------------- 2 motions

# A motion is a callable t -> (points, point_velocities), both of the mesh
# shape (nwrap+1, nspan+1, 3), carrying a boolean attribute `rigid`. Velocities
# are analytic, never differenced: update_points samples them at the
# collocation points with the operator that placed those points, so a rigid
# velocity field is transferred exactly and u_rel carries no motion error.


def _tag(motion, rigid, label):
    motion.rigid = bool(rigid)
    motion.label = label
    return motion


def still(points0):
    """A stationary surface. The onset alone drives the flow."""
    zero = np.zeros_like(points0)

    def motion(t):
        return points0, zero
    return _tag(motion, True, "still")


def translate(points0, velocity):
    """Uniform translation at a constant velocity (3,).

    With a zero onset this is a wing accelerated impulsively into still fluid,
    the frame in which added mass is cleanest to read.
    """
    velocity = np.asarray(velocity, dtype=float)
    vel = np.broadcast_to(velocity, points0.shape).copy()

    def motion(t):
        return points0 + t * velocity, vel
    return _tag(motion, True, f"translate {velocity}")


def accelerate(points0, acceleration, velocity0=(0.0, 0.0, 0.0)):
    """Constant acceleration from a given initial velocity; both (3,)."""
    acceleration = np.asarray(acceleration, dtype=float)
    velocity0 = np.asarray(velocity0, dtype=float)

    def motion(t):
        v = velocity0 + t * acceleration
        return (points0 + t * velocity0 + 0.5 * t ** 2 * acceleration,
                np.broadcast_to(v, points0.shape).copy())
    return _tag(motion, True, f"accelerate {acceleration}")


def heave(points0, amplitude, omega, phase=0.0):
    """Sinusoidal plunge along z: h(t) = amplitude * sin(omega t + phase)."""
    def motion(t):
        h = amplitude * np.sin(omega * t + phase)
        hdot = amplitude * omega * np.cos(omega * t + phase)
        pts = points0.copy()
        pts[..., 2] += h
        vel = np.zeros_like(points0)
        vel[..., 2] = hdot
        return pts, vel
    return _tag(motion, True, f"heave a={amplitude} omega={omega}")


def pitch(points0, amplitude, omega, pivot, phase=0.0, mean=0.0):
    """Sinusoidal pitch about a spanwise axis through pivot (3,), radians.

    Positive is nose-up, the sense of freewake_kernels.pitch_mesh, so the
    angle of attack is mean + amplitude*sin(omega t + phase). The velocity of a
    rotation at rate theta_dot about y through the pivot is
    theta_dot * (z - z_p, 0, -(x - x_p)) evaluated at the ROTATED point.
    """
    pivot = np.asarray(pivot, dtype=float)

    def motion(t):
        th = mean + amplitude * np.sin(omega * t + phase)
        thdot = amplitude * omega * np.cos(omega * t + phase)
        c, s = np.cos(th), np.sin(th)
        dx = points0[..., 0] - pivot[0]
        dz = points0[..., 2] - pivot[2]
        pts = points0.copy()
        pts[..., 0] = pivot[0] + c * dx + s * dz
        pts[..., 2] = pivot[2] - s * dx + c * dz
        vel = np.zeros_like(points0)
        vel[..., 0] = thdot * (pts[..., 2] - pivot[2])
        vel[..., 2] = -thdot * (pts[..., 0] - pivot[0])
        return pts, vel
    return _tag(motion, True, f"pitch a={amplitude} omega={omega}")


def pitch_and_heave(points0, pitch_amplitude, heave_amplitude, omega, pivot,
                    pitch_phase=0.0, mean=0.0):
    """Pitch about the pivot, then plunge the pitched mesh along z.

    The two are composed in this order because a rotation followed by a
    translation is the rigid motion the flapping-foil literature means; the
    velocities add for the same reason.
    """
    rotate = pitch(points0, pitch_amplitude, omega, pivot, pitch_phase, mean)

    def motion(t):
        pts, vel = rotate(t)
        pts = pts.copy()
        vel = vel.copy()
        pts[..., 2] += heave_amplitude * np.sin(omega * t)
        vel[..., 2] += heave_amplitude * omega * np.cos(omega * t)
        return pts, vel
    return _tag(motion, True, f"pitch+heave omega={omega}")


# ----------------------------------------------------------- 3 the time loop

def init_unsteady_state(points, onset, u_ref=1.0, lifting=True,
                        point_velocities=None, core=None, config=None):
    """A state with an empty wake: nrows = 0, n_bound = 1, closure = False.

    The zero-row wake is the correct initial condition for an impulsive start:
    with no wake the solve returns the acyclic flow, which is what the fluid
    does at the instant the motion begins.

    core overrides the wake vortex core radius. The steady default,
    RC_WAKE_FRACTION times the spanwise panel spacing, is sized for the tip
    roll-up of a steady free wake and reaches a third of a chord on a wing of
    aspect ratio 20, which looks alarming for unsteady work and is not: the
    wake reaches the body through the solid-angle POTENTIAL, which carries no
    core at all, so the core changes only the wake's own advection. Sweeping it
    from 0.33 c to 0.01 c moved the plunge amplitude and phase by less than
    10^-4 of themselves. Set it when the wake shape itself matters - a rolling
    up sheet, a close encounter with the body - and otherwise leave it.
    """
    cfg = {"lifting": bool(lifting), "nwake": 0, "n_bound": 1, "closure": False}
    cfg.update(config or {})
    if point_velocities is not None:
        cfg["point_velocities"] = point_velocities
    state = tpw.init_state(points, onset, u_ref, config=cfg)
    if core is not None and state["wake"] is not None:
        state["wake"]["core"] = float(core)
    return state


def backward_difference(mu_history, dt, order=DPHI_DT_ORDER):
    """dmu/dt from the newest entries of mu_history (newest last).

    Two-level (3 mu^n - 4 mu^n-1 + mu^n-2)/(2 dt) when order is two and three
    levels exist, else (mu^n - mu^n-1)/dt, else zeros. The driver owns this,
    never a physics block, because only the driver knows the time levels.
    """
    if len(mu_history) < 2:
        return np.zeros_like(mu_history[-1])
    if order >= 2 and len(mu_history) >= 3:
        return (3.0 * mu_history[-1] - 4.0 * mu_history[-2]
                + mu_history[-3]) / (2.0 * dt)
    return (mu_history[-1] - mu_history[-2]) / dt


def time_march(points0, onset, motion=None, dt=0.05, nsteps=100, u_ref=1.0,
               rho=tpw.RHO, nmax=NMAX_DEFAULT, order=DPHI_DT_ORDER,
               lifting=True, p_ref=None, verbose=False, callback=None,
               state=None):
    """March the flow from rest-of-wake through nsteps steps of dt.

    Returns (history, state). The step is

        v = wake_node_velocity(state)      # previous time level, explicit
        shed(wake, v, dt, v_shed=te_convection_velocity(state))
        truncate(wake, nmax)
        points, velocities = motion(t)
        update_points(state, points, velocities, rigid=motion.rigid)
        solve(state, wake='frozen')        # newest strip folded, history on rhs
        loads = get_loads(state, rho, dphi_dt=(backward difference of mu))

    solve(wake='frozen') is the entry point, never a bare solve_once: after
    update_points the operator blocks may be stale and solve_once asserts.

    history holds one entry per recorded level, level zero being the initial
    acyclic solution before any vorticity has been shed:
      t, s (reduced time 2 U_ref t / mean chord), CL, CD, CM, CL_quasi_steady,
      force (n,3), moment (n,3), mu_w_mid (midspan strip strength), nrows,
      dphi_dt_rms, and the scalars u_ref, chord, s_ref, dt, rho.

    CL_quasi_steady omits dmu/dt, so the difference between it and CL is the
    added-mass contribution, reported separately because that is where a sign
    error would hide.
    """
    if motion is None:
        motion = still(points0)
    rigid = bool(getattr(motion, "rigid", False))
    pts0, vel0 = motion(0.0)
    if state is None:
        state = init_unsteady_state(pts0, onset, u_ref, lifting=lifting,
                                    point_velocities=vel0)
    else:
        # a caller-supplied state carries the reference mesh, which the motion
        # may already have displaced at t = 0; synchronise before the first
        # solve rather than trusting the two to agree
        state = tpw.update_points(state, pts0, vel0, rigid=rigid)
    state = tpw.solve(state, wake="frozen" if lifting else "none")

    chord = state["chord_mean"]
    per_step = convection_per_step(state, dt)
    if per_step > STEP_WARN_CHORDS:
        warnings.warn(
            f"the flow convects {per_step:.2f} chords per step; the wake is "
            f"under-resolved near the trailing edge and the indicial response "
            f"will be inaccurate even though the steady limit is not")
    hist = {k: [] for k in ("t", "s", "CL", "CD", "CM", "CL_quasi_steady",
                            "force", "moment", "mu_w_mid", "nrows",
                            "dphi_dt_rms")}
    mu_history = [state["mu"].copy()]

    def _record(t, dphi):
        pan = state["panels"]
        loads = tpw.get_loads(state, rho=rho, p_ref=p_ref, dphi_dt=dphi)
        res = loads["resultants"]
        # the quasi-steady load is the same integral without the dmu/dt term,
        # and p_qs = p + rho dmu/dt exactly, so it costs a subtraction rather
        # than a second pressure evaluation
        if dphi is None:
            cl_qs = res["CL"]
        else:
            d_force = (rho * dphi)[:, None] * pan["areas"][:, None] \
                * pan["normals"]
            cl_qs = (res["force"][2] - d_force[:, 2].sum()) \
                / (0.5 * rho * u_ref ** 2 * state["s_ref"])
        hist["t"].append(t)
        hist["s"].append(2.0 * u_ref * t / chord)
        hist["CL"].append(res["CL"])
        hist["CD"].append(res["CD_pressure"])
        hist["CM"].append(res["CM"])
        hist["CL_quasi_steady"].append(cl_qs)
        hist["force"].append(res["force"].copy())
        hist["moment"].append(res["moment"].copy())
        wake = state["wake"]
        hist["mu_w_mid"].append(
            float(wake["mu"][0, wake["mu"].shape[1] // 2])
            if wake is not None and wake["mu"].size else 0.0)
        hist["nrows"].append(0 if wake is None else int(wake["mu"].shape[0]))
        hist["dphi_dt_rms"].append(
            0.0 if dphi is None else float(np.sqrt(np.mean(dphi ** 2))))
        return loads

    _record(0.0, None)
    if verbose:
        print(f"  step   0  t = 0.000  CL = {hist['CL'][0]:+.5f}  "
              f"(acyclic start, no wake)")

    for n in range(1, nsteps + 1):
        t = n * dt
        if lifting:
            v_nodes = tpw.wake_node_velocity(state)
            tpw.shed(state["wake"], v_nodes, dt,
                     v_shed=tpw.te_convection_velocity(state))
            tpw.truncate(state["wake"], nmax)
            state["wake_version"] += 1
        pts, vel = motion(t)
        state = tpw.update_points(state, pts, vel, rigid=rigid)
        state = tpw.solve(state, wake="frozen" if lifting else "none")
        mu_history.append(state["mu"].copy())
        if len(mu_history) > 3:
            mu_history.pop(0)
        dphi = backward_difference(mu_history, dt, order)
        _record(t, dphi)
        if verbose:
            print(f"  step {n:3d}  t = {t:.3f}  s = {hist['s'][-1]:6.2f}  "
                  f"CL = {hist['CL'][-1]:+.5f}  "
                  f"(quasi-steady {hist['CL_quasi_steady'][-1]:+.5f}, "
                  f"{hist['nrows'][-1]} wake rows)")
        if callback:
            callback(n, t, hist, state)

    out = {k: np.array(v) for k, v in hist.items()}
    out.update(u_ref=float(u_ref), chord=float(chord), dt=float(dt),
               rho=float(rho), s_ref=float(state["s_ref"]),
               motion=getattr(motion, "label", "unnamed"))
    return out, state


def convection_per_step(state, dt):
    """Convection distance of one step in mean chords, the resolution measure.

    Below about 0.5 the wake is resolved in the sense that the strips near the
    trailing edge are shorter than the panels that shed them; above it the
    indicial response is under-resolved even though the steady limit is not.
    """
    speed = float(np.linalg.norm(state["u_rel"], axis=1).mean())
    return speed * dt / state["chord_mean"]


# --------------------------------------------------------- 4 harmonic loads

def first_harmonic(t, f, omega, cycles=1):
    """First harmonic of f(t), fitted over the LAST whole cycles.

    Returns (mean, amplitude, phase) of f ~ mean + amplitude cos(omega t +
    phase), by least squares on the three-term basis. Fitting only whole cycles
    at the end of the record keeps the starting transient out of the harmonic;
    on the plunge cases the answer was unchanged to four figures between three
    and six cycles, which is how that was checked rather than assumed.
    """
    period = 2.0 * np.pi / omega
    t_end = t[-1]
    sel = t >= t_end - cycles * period - 1e-12
    if sel.sum() < 8:
        raise ValueError("fewer than eight samples in the fitted cycles; "
                         "march longer or reduce dt")
    tt, ff = t[sel], f[sel]
    basis = np.stack([np.ones_like(tt), np.cos(omega * tt), np.sin(omega * tt)],
                     axis=1)
    coeff, *_ = np.linalg.lstsq(basis, ff, rcond=None)
    mean, a, b = coeff
    return float(mean), float(np.hypot(a, b)), float(np.arctan2(-b, a))


# -------------------------------------------------------------- 5 plotting

def plot_history(history, state=None, save=None):
    """Lift history, added-mass split, wake strength and wake shape. Figure."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["s"], history["CL"], "-", label=r"$C_L$")
    ax.plot(history["s"], history["CL_quasi_steady"], "--",
            label="quasi-steady")
    ax.set_xlabel(r"reduced time $s = 2 U t / c$")
    ax.set_ylabel(r"$C_L$")
    ax.set_title("Lift history")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["s"], history["CL"] - history["CL_quasi_steady"], "-")
    ax.set_xlabel(r"reduced time $s$")
    ax.set_ylabel(r"$C_L - C_{L,\rm qs}$")
    ax.set_title(r"Added mass, $-\rho\,{\rm d}\mu/{\rm d}t$")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(history["s"], history["mu_w_mid"], "-")
    ax.set_xlabel(r"reduced time $s$")
    ax.set_ylabel(r"$\mu_w$ at midspan")
    ax.set_title("Newest shed strip")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(history["s"], history["CM"], "-")
    ax.set_xlabel(r"reduced time $s$")
    ax.set_ylabel(r"$C_M$")
    ax.set_title("Pitching moment, root quarter chord")
    ax.grid(True, alpha=0.3)

    if state is not None and state["wake"] is not None:
        nodes = state["wake"]["nodes"]
        points = state["panels"]["points"]
        jm = nodes.shape[1] // 2
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(nodes[:, jm, 0], nodes[:, jm, 2], ".-", ms=2, lw=0.8)
        ax.plot(points[:, points.shape[1] // 2, 0],
                points[:, points.shape[1] // 2, 2], "k-", lw=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Midspan wake, side view")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 2], projection="3d")
        for j in range(nodes.shape[1]):
            tip = j in (0, nodes.shape[1] - 1)
            ax.plot(nodes[:, j, 0], nodes[:, j, 1], nodes[:, j, 2],
                    color="tab:red" if tip else "tab:blue",
                    lw=1.2 if tip else 0.5)
        nwrap = points.shape[0] - 1
        for i in range(0, nwrap + 1, max(nwrap // 12, 1)):
            ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2],
                    color="k", lw=0.5)
        ax.set_title("Wake")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


# ------------------------------------------------------------------ 6 CLI

def main():
    mesh_path = prompt("Wing mesh (.npz, thick-wrap-1)", "naca0012_rect_mesh.npz")
    profile_path = prompt("Velocity profile CSV (z, U)", "uniform_profile.csv")
    u_ref = float(prompt("Reference speed U_ref", "1.0"))
    alpha_deg = float(prompt("Mean pitch angle [deg]", "5.0"))
    kind = prompt("Motion: start | heave | pitch", "start")
    nsteps = int(prompt("Number of time steps", "60"))

    points = tpw.load_mesh(mesh_path)
    if abs(alpha_deg) > 0.0:
        points = tpw.validate_mesh(pitch_mesh(points, np.radians(alpha_deg)))
    z_table, u_table = load_velocity_profile(profile_path)
    onset = make_onset(z_table, u_table)

    chord = float(tpw.section_chords(points).mean())
    if kind == "start":
        motion = still(points)
        dt = 0.05 * chord / max(u_ref, 1e-12)
    else:
        k = float(prompt("Reduced frequency k = omega c / (2 U)", "0.2"))
        omega = 2.0 * k * u_ref / chord
        steps_per_cycle = float(prompt("Time steps per cycle", "40"))
        dt = 2.0 * np.pi / omega / steps_per_cycle
        if kind == "heave":
            motion = heave(points, float(prompt("Heave amplitude / c", "0.05"))
                           * chord, omega)
        else:
            te = tpw.te_nodes(points)
            jm = te.shape[0] // 2
            pivot = te[jm] + np.array([-0.75 * chord, 0.0, 0.0])
            motion = pitch(points, np.radians(
                float(prompt("Pitch amplitude [deg]", "2.0"))), omega, pivot)

    state = init_unsteady_state(points, onset, u_ref)
    print(f"\nMesh: {state['panels']['nwrap']} wrap x "
          f"{state['panels']['nspan']} spanwise panels, mean chord "
          f"{chord:.3f}, S_ref {state['s_ref']:.4f}")
    print(f"Motion: {getattr(motion, 'label', kind)}; dt = {dt:.4f}, "
          f"{nsteps} steps, convection per step "
          f"{u_ref * dt / chord:.3f} chords")
    print("Marching...")
    history, state = time_march(points, onset, motion, dt=dt, nsteps=nsteps,
                                u_ref=u_ref, verbose=True, state=state)

    print(f"\nFinal C_L = {history['CL'][-1]:.4f} "
          f"(quasi-steady {history['CL_quasi_steady'][-1]:.4f})")
    print(f"Final C_M = {history['CM'][-1]:.4f}")
    print(f"Wake rows: {history['nrows'][-1]}")

    import matplotlib.pyplot as plt
    plot_history(history, state)
    plt.show()


if __name__ == "__main__":
    main()
