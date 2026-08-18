"""A rigid surface-piercing wing in a pitch ramp, with free ventilation.

    python3 vent_ramp.py [--nspan 20 --nc 20 --dt 0.05 --steps 400]

Everything is nondimensional on the water density, the chord and the free-stream
speed: rho = 1, c = 1, U = 1, so a convective time is a chord of travel. The
strut is immersed to h = 4 chords, pierces the free surface, and is yawed from
alpha = 0 at t = 0 to alpha = 20 deg at t = 20 linearly in time. The depth
Froude number is Fn_h = U / sqrt(g h) = 10 and the stall angle - which is the
free-surface seal gate, not a fluid-dynamic stall in the panel solve - is 15 deg.

The wing is RIGID: no structural state is built and no coupling subiteration is
run. The geometry of each level is the reference strut pitched to the current
incidence, and the surface velocity is the exact rigid-body velocity of that
rotation, so the added-mass term reaches the pressure through dmu/dt as it does
in a coupled march.

The spanwise stations are cosine-clustered WITHIN EACH HALF, which puts the fine
spacing at the immersed tip and at the waterline - the two edges of the immersed
half - rather than at the tips of the doubled mesh alone. The chordwise wrap is
the vendored cosine one.

This file writes; it writes only under runs/ramp/.
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import freewake_kernels as fk
import make_thick_sample_inputs as mts
import thick_panel_wing as tpw
import unsteady_wing as usw
import vent_cavity as vcv
import vent_loads as vnl
import vent_mesh as vm
import vent_solve as vsl

OUT = os.path.join("runs", "ramp")
H = 4.0                    # immersed span, in chords
CHORD = 1.0
THICKNESS = 0.12           # NACA 0012
FN_H = 10.0                # U / sqrt(g h)
ALPHA_END = 20.0           # degrees, reached at T_END
T_END = 20.0               # convective times
ALPHA_STALL = 15.0         # degrees; the free-surface seal gate
GAMMA_ST = 1.0e-6          # negligible surface tension at this (nondimensional)
                           # scale, so the Weber gate never inhibits inception


def half_cosine_stations(nspan_half, h):
    """Depth stations of the doubled strut, clustered at tip AND waterline.

    -> (2*nspan_half + 1,) monotone, antisymmetric about y = 0, spanning [-h, h].
    """
    j = np.arange(nspan_half + 1)
    lower = -0.5 * h * (1.0 + np.cos(np.pi * j / nspan_half))   # -h .. 0
    return np.concatenate([lower, -lower[-2::-1]])


def ramp_mesh(alpha_deg, n_c, nspan_half, h=H, chord=CHORD):
    """The doubled strut at one incidence, on cosine stations. -> points"""
    nspan = 2 * nspan_half
    points = mts.thick_wing_mesh(n_c=n_c, nspan=nspan, span=2.0 * h,
                                 root_chord=chord, taper=1.0, sweep_deg=0.0,
                                 twist_deg=0.0, camber=0.0,
                                 thickness=THICKNESS)
    points[..., vm.DEPTH_AXIS] = half_cosine_stations(nspan_half, h)[None, :]
    points = vm.symmetrise_points(points, y_fs=0.0)
    points = tpw.pitch_mesh(points, np.radians(alpha_deg))
    return tpw.validate_mesh(points)


def pitch_velocity(points, alpha_dot):
    """Rigid-body velocity of a nose-up rotation about the y axis. -> like points

    p(alpha) = R(alpha) p0 with R about y through the quarter chord, so
    dp/dt = alpha_dot * (z, 0, -x) in the CURRENT coordinates.
    """
    v = np.zeros_like(points)
    v[..., 0] = alpha_dot * points[..., 2]
    v[..., 2] = -alpha_dot * points[..., 0]
    return v


def alpha_of(t):
    """The ramp, held at ALPHA_END beyond T_END. -> (alpha_deg, alpha_dot_rad)"""
    if t >= T_END:
        return ALPHA_END, 0.0
    return ALPHA_END * t / T_END, np.radians(ALPHA_END) / T_END


def wetted_area(pan, vent, cav):
    """Immersed-half areas: total, dry (ventilated) and wetted. -> dict"""
    real = np.asarray(vent["real_mask"], dtype=bool)
    areas = np.asarray(pan["areas"])[real]
    weight = np.clip(np.asarray(cav["weight"])[real], 0.0, 1.0)
    dry = float((weight * areas).sum())
    total = float(areas.sum())
    return {"total": total, "dry": dry, "wet": total - dry}


# ------------------------------------------------------------------ the march

def march(n_c=20, nspan_half=20, dt=0.05, nsteps=400, nwake=None, verbose=True):
    """The ramp, marched. -> (history, fluid, vent)"""
    points0 = ramp_mesh(0.0, n_c, nspan_half)
    onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))
    fluid = usw.init_unsteady_state(points0, onset, u_ref=1.0,
                                    config={"lifting": True, "nwake": 0})
    maps = vm.mirror_maps(points0.shape, image=True)
    vent = vcv.build_vent(maps, vm.half_points(points0, maps), h=H, chord=CHORD,
                          u_ref=1.0, alpha_rad=0.0, fn_h=FN_H, rho=1.0,
                          gamma_st=GAMMA_ST,
                          alpha_stall=np.radians(ALPHA_STALL), regime="FW")

    keys = ["t", "alpha", "CL", "CD", "CM", "regime", "l_c_max", "l_c_mean",
            "d_cav", "phi_bar", "area_dry", "area_wet", "area_total",
            "ready_fraction", "entrainment", "drift_y", "cav_iter", "nrows",
            "margin"]
    hist = {k: [] for k in keys}
    mu_committed = []

    def record(t, alpha, cav, loads, drift):
        res = loads["resultants"]
        area = wetted_area(fluid["panels"], vent, cav)
        hist["t"].append(t)
        hist["alpha"].append(alpha)
        hist["CL"].append(res["CL"])
        hist["CD"].append(res["CD_pressure"])
        hist["CM"].append(res["CM"])
        hist["regime"].append(vcv.REGIMES.index(vent["regime"]))
        hist["l_c_max"].append(float(np.max(cav["l_c"])))
        live = cav["l_c"] > 0.0
        hist["l_c_mean"].append(float(cav["l_c"][live].mean()) if live.any()
                                else 0.0)
        hist["d_cav"].append(float(cav["d_cav"]))
        hist["phi_bar"].append(float(np.degrees(cav["phi_bar"])))
        hist["area_dry"].append(area["dry"])
        hist["area_wet"].append(area["wet"])
        hist["area_total"].append(area["total"])
        hist["ready_fraction"].append(float(cav["ready_fraction"]))
        hist["entrainment"].append(float(np.sum(cav["entrainment"])))
        hist["drift_y"].append(float(drift))
        hist["cav_iter"].append(int(cav["iterations"]))
        wake = fluid["wake"]
        hist["nrows"].append(0 if wake is None else int(wake["mu"].shape[0]))
        hist["margin"].append(float(vsl.washout_margin(vent, res["CL"])))

    # level zero: the acyclic flow at alpha = 0, and the seed of the doublet
    # history - without it the first step's dmu/dt is zero and the added mass
    # lags by one level for the whole march
    fluid = tpw.update_points(fluid, points0,
                              pitch_velocity(points0, alpha_of(0.0)[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=1.0)
    mu_committed.append(fluid["mu"].copy())
    vcv.commit_vent(vent, cav, 0.0, dt)
    record(0.0, 0.0, cav, loads, 0.0)

    for n in range(1, nsteps + 1):
        t = n * dt
        alpha, alpha_dot = alpha_of(t)
        # shed ONCE per step, before the solve, and project the image wake
        fluid, drift = vsl.shed_and_project(fluid, vent, dt, nwake)
        points = ramp_mesh(alpha, n_c, nspan_half)
        fluid = tpw.update_points(fluid, points,
                                  pitch_velocity(points, alpha_dot))
        # the incidence is an INPUT of the cavity model - it sets the suction
        # side and the free-surface seal gate - so it moves with the geometry
        vent["alpha_rad"] = np.radians(alpha)
        dphi_of_mu = lambda m: usw.backward_difference(mu_committed + [m], dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=1.0,
                                                 dphi_of_mu=dphi_of_mu, dt=dt)
        mu_committed.append(fluid["mu"].copy())
        if len(mu_committed) > 2:
            mu_committed.pop(0)
        vcv.commit_vent(vent, cav, t, dt)
        vent["drift_y"] = drift
        record(t, alpha, cav, loads, drift)
        if verbose and (n % 10 == 0 or n <= 3):
            print(f"  step {n:4d}  t = {t:6.3f}  alpha = {alpha:5.2f} deg  "
                  f"CL = {hist['CL'][-1]:+.4f}  {vent['regime']:>2s}  "
                  f"dry = {100 * hist['area_dry'][-1] / hist['area_total'][-1]:5.1f}%"
                  f"  l_c,max = {hist['l_c_max'][-1]:.3f}", flush=True)

    out = {k: np.array(v) for k, v in hist.items()}
    out.update(dt=dt, n_c=n_c, nspan_half=nspan_half, fn_h=FN_H, h=H)
    return out, fluid, vent


# --------------------------------------------------------------- the figures

def plot_mesh(points, vent, save):
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1, linewidth=0.3,
                      color="0.35")
    ax.plot_surface(np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                    np.zeros((2, 2)),
                    np.array([[-1.0, -1.0], [1.0, 1.0]]),
                    color="tab:blue", alpha=0.15)
    ax.set_xlabel("x/c"), ax.set_ylabel("y/c (depth)"), ax.set_zlabel("z/c")
    ax.set_title("doubled mesh: immersed strut and its negative image")
    try:
        ax.set_box_aspect((2.0, 4.0, 2.0))
    except AttributeError:
        pass
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(points[:, :, 0], points[:, :, 1], color="0.35", linewidth=0.3)
    ax2.plot(points[:, :, 0].T, points[:, :, 1].T, color="0.35", linewidth=0.3)
    ax2.axhline(0.0, color="tab:blue", linewidth=1.5)
    ax2.set_xlabel("x/c"), ax2.set_ylabel("y/c (depth)")
    ax2.set_title("planform: cosine stations at tip and waterline")
    ax2.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save, dpi=160)
    plt.close(fig)


def plot_wake(fluid, vent, save):
    wake = fluid["wake"]
    nodes = wake["nodes"]
    pan = fluid["panels"]
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    body = pan["points"]
    ax.plot_wireframe(body[..., 0], body[..., 1], body[..., 2], rstride=2,
                      cstride=2, linewidth=0.2, color="0.5")
    ax.plot_wireframe(nodes[..., 0], nodes[..., 1], nodes[..., 2], rstride=1,
                      cstride=1, linewidth=0.25, color="tab:red")
    ax.set_xlabel("x/c"), ax.set_ylabel("y/c"), ax.set_zlabel("z/c")
    ax.set_title(f"free wake at t = 20 ({nodes.shape[0] - 1} rows)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(nodes[..., 0], nodes[..., 1], color="tab:red", linewidth=0.3)
    ax2.plot(body[:, :, 0], body[:, :, 1], color="0.5", linewidth=0.2)
    ax2.axhline(0.0, color="tab:blue", linewidth=1.5)
    ax2.set_xlabel("x/c"), ax2.set_ylabel("y/c (depth)")
    ax2.set_title("wake, top view: real sheet and its image")
    fig.tight_layout()
    fig.savefig(save, dpi=160)
    plt.close(fig)


def plot_history(hist, save):
    t, a = hist["t"], hist["alpha"]
    fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    ax[0, 0].plot(t, hist["CL"], color="tab:blue")
    ax[0, 0].set_ylabel("$C_L$ (side force)")
    ax[0, 0].set_title("lift history")
    axa = ax[0, 0].twinx()
    axa.plot(t, a, color="0.6", linewidth=0.8)
    axa.set_ylabel(r"$\alpha$ (deg)", color="0.5")

    ax[0, 1].plot(t, 100 * hist["area_dry"] / hist["area_total"],
                  color="tab:red", label="dry (ventilated)")
    ax[0, 1].plot(t, 100 * hist["area_wet"] / hist["area_total"],
                  color="tab:blue", label="wetted")
    ax[0, 1].set_ylabel("area (% of immersed surface)")
    ax[0, 1].set_title("dry and wetted area")
    ax[0, 1].legend(loc="center left", fontsize=8)

    ax[1, 0].plot(t, hist["CD"], color="tab:green")
    ax[1, 0].set_ylabel("$C_D$ (pressure)")
    ax[1, 0].set_title("drag history")

    ax[1, 1].plot(t, hist["l_c_max"], label=r"$\max L_c/c$")
    ax[1, 1].plot(t, hist["l_c_mean"], label=r"mean $L_c/c$ (ventilated)")
    ax[1, 1].plot(t, hist["d_cav"] / hist_h(hist), label=r"$d_{cav}/h$")
    ax[1, 1].set_ylabel("cavity extent")
    ax[1, 1].set_title("cavity length and depth")
    ax[1, 1].legend(fontsize=8)

    ax[2, 0].step(t, hist["regime"], where="post", color="tab:purple")
    ax[2, 0].set_yticks(range(len(vcv.REGIMES)))
    ax[2, 0].set_yticklabels(vcv.REGIMES)
    ax[2, 0].set_ylabel("regime")
    ax[2, 0].set_xlabel("t (convective times)")
    ax[2, 0].set_title("regime, and the ventilation-ready area fraction")
    axr = ax[2, 0].twinx()
    axr.plot(t, hist["ready_fraction"], color="0.6", linewidth=0.8)
    axr.set_ylabel("ready fraction", color="0.5")

    ax[2, 1].plot(t, hist["CM"], color="tab:orange", label="$C_M$ (yaw)")
    ax[2, 1].plot(t, hist["margin"], color="tab:cyan", label="washout margin")
    ax[2, 1].axhline(0.0, color="0.7", linewidth=0.6)
    ax[2, 1].set_xlabel("t (convective times)")
    ax[2, 1].set_title("yawing moment and washout margin")
    ax[2, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=160)
    plt.close(fig)


def hist_h(hist):
    return H


def plot_loading(fluid, vent, cav, save):
    """Depth loading and the cavity footprint at the final level."""
    pan = fluid["panels"]
    real = np.asarray(vent["real_mask"], dtype=bool)
    shape = (pan["areas"].shape[0] // (vent["shape_full"][1] - 1),
             vent["shape_full"][1] - 1)
    weight = np.asarray(cav["weight"]).reshape(shape)
    depth = vcv.panel_depth(pan, vent["y_fs"]).reshape(shape)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    keep = real.reshape(shape).any(axis=0)
    im = ax[0].pcolormesh(np.arange(shape[0] + 1),
                          np.r_[depth[0, keep], depth[0, keep][-1]],
                          weight[:, keep].T, cmap="Reds", vmin=0.0, vmax=1.0,
                          shading="auto")
    ax[0].set_xlabel("wrap panel index (TE $\\to$ LE $\\to$ TE)")
    ax[0].set_ylabel("depth y/c")
    ax[0].set_title("cavity coverage on the immersed half, t = 20")
    fig.colorbar(im, ax=ax[0], label="cavity fraction")
    cp = np.asarray(cav["cp"]).reshape(shape) if "cp" in cav else None
    if cp is not None:
        ax[1].plot(cp[:, keep].mean(axis=1), label="mean $C_p$ over depth")
    ax[1].plot(np.asarray(cav["cp_baseline"]).reshape(shape)[:, keep].mean(axis=1),
               label="wetted baseline $C_p$")
    ax[1].set_xlabel("wrap panel index")
    ax[1].set_ylabel("$C_p$")
    ax[1].legend(fontsize=8)
    ax[1].set_title("depth-averaged pressure, ventilated against wetted")
    fig.tight_layout()
    fig.savefig(save, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", type=int, default=20)
    p.add_argument("--nspan", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--nwake", type=int, default=None)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print(f"rigid surface-piercing strut, h/c = {H:.0f}, Fn_h = {FN_H:.0f}, "
          f"alpha 0 -> {ALPHA_END:.0f} deg in {T_END:.0f} convective times")
    print(f"  {2 * args.nc} x {2 * args.nspan} panels on the doubled mesh "
          f"({args.nc} chordwise per surface, {args.nspan} spanwise immersed), "
          f"dt = {args.dt}, {args.steps} steps")
    plot_mesh(ramp_mesh(ALPHA_END, args.nc, args.nspan), None,
              os.path.join(OUT, "mesh.png"))

    hist, fluid, vent = march(n_c=args.nc, nspan_half=args.nspan, dt=args.dt,
                              nsteps=args.steps, nwake=args.nwake)
    np.savez(os.path.join(OUT, "history.npz"), **hist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cav_end, _ = vsl.solve_cavity(fluid, vent, rho=1.0)
    plot_wake(fluid, vent, os.path.join(OUT, "wake.png"))
    plot_history(hist, os.path.join(OUT, "history.png"))
    plot_loading(fluid, vent, cav_end, os.path.join(OUT, "cavity.png"))
    frame = vcv.chordwise_frame(fluid["panels"])
    rep = vcv.cavity_report(fluid["panels"], frame, vent, cav_end)
    print("\nfinal state")
    for k, v in rep.items():
        print(f"  {k:>18s}  {v}")
    print(f"\nfigures under {OUT}/")


if __name__ == "__main__":
    main()
