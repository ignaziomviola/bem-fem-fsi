"""Partitioned coupling of the panel method to the finite element solid.

Two drivers: `static_aeroelastic`, the steady fixed point that the fluid
repository's ARCHITECTURE.md wrote out on paper, and `time_march_fsi`, the
strongly coupled time march. Both are Gauss-Seidel fixed points on the
structural displacement, accelerated by Aitken relaxation or by IQN-ILS.

Prompts, prints and figures live only under main(), on both sides of the
coupling, so importing this module creates none of them.

## The one thing about the time step that had to be checked, not assumed

`shed` prepends a wake row AND convects every existing node, mutating the wake
irreversibly. It must therefore be called ONCE per time step, before the
coupling subiterations, using the converged previous level; everything inside
the subiteration is `update_points` followed by `solve(wake="frozen")`, which
is deterministic and idempotent at fixed geometry. `update_points` re-pins wake
row zero onto the moved trailing edge on every call, so the newly shed strip
follows the deforming surface without the wake being shed again.

That is what makes the coupling residual a fixed function of the displacement,
which is what quasi-Newton acceleration requires. Equally, the current iterate
must not enter the doublet history until the step converges, or dmu/dt - and
with it the entire added mass - would depend on the path the iteration took to
get there rather than on the time level.

## Why strong coupling is not optional in water

A staggered scheme that solves each field once per step is unconditionally
unstable below a critical ratio of structural to added mass, however small the
time step: the fluid's response to an acceleration of the surface is
instantaneous and proportional to that acceleration, so the explicit lag acts
as a negative mass. In air, at mass ratios of order 100, nobody notices. In
water, at mass ratios of order one, it diverges in a handful of steps.
`coupling="loose"` is kept so that this can be demonstrated rather than
asserted - it is a verification case, not a recommendation.
"""

import warnings

import numpy as np

import thick_panel_wing as tpw
import unsteady_wing as usw
import fem_solid as fes
import fsi_transfer as fsx

TOL_COUPLING = 1e-6         # relative displacement residual of the fixed point
MAX_SUB = 25                # subiterations per time step
OMEGA_START = 0.3           # first-iteration relaxation, before any history
OMEGA_MAX = 1.0e3           # blow-up guard on the Aitken factor, not a limiter
IQN_REUSE = 8               # columns kept in the IQN-ILS least-squares system
STEP_WARN_CHORDS = 0.5      # fluid resolution: chords convected per step
STEPS_PER_PERIOD_WARN = 10  # structural resolution: steps per softest period


# --------------------------------------------------------- 1 acceleration

class FixedPointAccelerator:
    """Aitken relaxation or IQN-ILS over a sequence of coupling residuals.

    Both take the residual r = u_tilde - u of the fixed point u -> u_tilde and
    return the next iterate. Aitken carries one scalar, IQN-ILS a small
    least-squares system built from the differences of successive residuals and
    outputs; on a stiff interface the second is worth several times the first,
    and on an easy one they cost the same.
    """

    def __init__(self, method="aitken", omega=OMEGA_START, reuse=IQN_REUSE):
        if method not in ("aitken", "iqn", "constant"):
            raise ValueError(f"unknown accelerator '{method}'; use 'aitken', "
                             f"'iqn' or 'constant'")
        self.method, self.omega0, self.reuse = method, float(omega), int(reuse)
        self.reset()

    def reset(self):
        self.omega = self.omega0
        self.r_prev = None
        self.u_prev = None
        self.cols_v, self.cols_w = [], []

    def step(self, u, u_tilde):
        """One accelerated iterate. u and u_tilde are (Ns, 3); returns (Ns, 3)."""
        shape = np.shape(u)
        u_f = np.asarray(u, dtype=float).reshape(-1)
        t_f = np.asarray(u_tilde, dtype=float).reshape(-1)
        r = t_f - u_f
        if self.method == "constant":
            return (u_f + self.omega0 * r).reshape(shape)
        if self.method == "aitken":
            if self.r_prev is not None:
                d_r = r - self.r_prev
                den = float(d_r @ d_r)
                if den > 0.0:
                    self.omega = -self.omega * float(self.r_prev @ d_r) / den
                    # a magnitude guard against a degenerate denominator, NOT a
                    # limiter: the optimal factor for a fixed point whose
                    # dominant eigenvalue is lambda is 1/(1 - lambda), which is
                    # 50 at lambda = 0.98, so clipping to a small range removes
                    # exactly the acceleration Aitken exists to supply
                    self.omega = float(np.clip(self.omega, -OMEGA_MAX, OMEGA_MAX))
                    if abs(self.omega) < 1e-6 or not np.isfinite(self.omega):
                        self.omega = self.omega0
            self.r_prev = r
            return (u_f + self.omega * r).reshape(shape)
        # IQN-ILS: least-squares over the differences, Degroote's form
        if self.r_prev is not None:
            self.cols_v.append(r - self.r_prev)
            self.cols_w.append(t_f - self.u_prev)
            self.cols_v = self.cols_v[-self.reuse:]
            self.cols_w = self.cols_w[-self.reuse:]
        self.r_prev, self.u_prev = r, t_f
        if not self.cols_v:
            return (u_f + self.omega0 * r).reshape(shape)
        v_mat = np.stack(self.cols_v, axis=1)
        w_mat = np.stack(self.cols_w, axis=1)
        coeff, *_ = np.linalg.lstsq(v_mat, -r, rcond=None)
        return (u_f + w_mat @ coeff + r).reshape(shape)


# ------------------------------------------------------------- 2 assembly

def init_fsi(points, onset, model, u_ref=1.0, rho=tpw.RHO, unsteady=True,
             core=None, config=None, rayleigh=(0.0, 0.0), lumped_mass=False):
    """Build the fluid state, the structural state and the transfer operator.

    The transfer is built on the two REFERENCE configurations and never rebuilt,
    which is what keeps the coupling residual a fixed function of the state.
    """
    if unsteady:
        fluid = usw.init_unsteady_state(points, onset, u_ref, core=core,
                                        config=config)
    else:
        fluid = tpw.init_state(points, onset, u_ref, config=config)
    sstate = fes.init_state(model, rayleigh=rayleigh, lumped_mass=lumped_mass)
    transfer = fsx.build_transfer(fluid, model)
    return fluid, sstate, transfer


def _loads_on_structure(fluid, transfer, rho, p_ref=None, dphi_dt=None):
    """One fluid load evaluation, delivered as structural nodal forces."""
    loads = tpw.get_loads(fluid, rho=rho, p_ref=p_ref, dphi_dt=dphi_dt)
    return fsx.structural_forces(transfer, fluid["panels"], loads["force"]), loads


def _level_from_u(sstate, u, dt, par):
    """Velocity and acceleration implied by a displacement, from Newmark.

    The coupling unknown is the displacement alone; the integrator's own
    relations then fix the other two exactly, so an accelerated iterate stays a
    consistent kinematic state instead of an interpolation of three of them.
    """
    beta, gamma = par["beta"], par["gamma"]
    u_n, v_n, a_n = (sstate[k].reshape(-1) for k in ("u", "v", "a"))
    u_p = u_n + dt * v_n + dt ** 2 * (0.5 - beta) * a_n
    v_p = v_n + dt * (1.0 - gamma) * a_n
    acc = (np.asarray(u, dtype=float).reshape(-1) - u_p) / (beta * dt ** 2)
    return {"u": np.asarray(u, dtype=float).reshape(-1, 3),
            "v": (v_p + gamma * dt * acc).reshape(-1, 3),
            "a": acc.reshape(-1, 3)}


# ------------------------------------------------------- 3 static coupling

def static_aeroelastic(fluid, sstate, transfer, rho=tpw.RHO, p_ref=None,
                       tol=TOL_COUPLING, max_iter=MAX_SUB, accel="aitken",
                       omega=OMEGA_START, relax_wake_every=0, verbose=False,
                       callback=None):
    """Steady aeroelastic fixed point. Returns (fluid, sstate, history).

    The loop of the fluid repository's ARCHITECTURE.md, with the acceleration
    it left to the structural driver: relax the wake on the first pass, freeze
    it afterwards, and iterate displacement against load until the interface
    stops moving. relax_wake_every > 0 re-relaxes periodically, which matters
    only when the deflection is large enough to move the sheet.

    Divergence shows up here as a fixed point that stops contracting; the
    history carries the residual sequence so the approach to it can be measured
    rather than inferred from a failure.
    """
    points0 = np.array(fluid["panels"]["points"])
    acc = FixedPointAccelerator(accel, omega)
    if sstate["K"] is None:
        fes.assemble_operators(sstate)
    u = np.array(sstate["u"])
    hist = {"residual": [], "tip": [], "iterations": 0, "converged": False}
    scale = max(float(np.linalg.norm(np.ptp(points0.reshape(-1, 3), axis=0))),
                1e-300)
    for it in range(max_iter):
        pts = fsx.displaced_points(transfer, points0, u)
        fluid = tpw.update_points(fluid, pts)
        mode = "relax" if it == 0 or (relax_wake_every
                                      and it % relax_wake_every == 0) else "frozen"
        fluid = tpw.solve(fluid, wake=mode if fluid["wake"] is not None else "none")
        f_struct, _ = _loads_on_structure(fluid, transfer, rho, p_ref)
        level = fes.solve_static(sstate, f_struct, u0=u)
        resid = float(np.linalg.norm(level["u"] - u) / scale)
        hist["residual"].append(resid)
        hist["tip"].append(float(np.abs(level["u"]).max()))
        if verbose:
            print(f"  coupling {it + 1:2d}: residual {resid:.3e}, "
                  f"max displacement {hist['tip'][-1]:.6e}")
        if callback:
            callback(it, resid, fluid, sstate)
        if resid < tol:
            fes.commit(sstate, level, f_struct)
            hist["converged"] = True
            hist["iterations"] = it + 1
            return fluid, sstate, hist
        u = acc.step(u, level["u"])
        fes.commit(sstate, _level_from_u_static(level, u), f_struct)
    hist["iterations"] = max_iter
    warnings.warn(f"the static coupling did not converge in {max_iter} "
                  f"iterations (last residual {hist['residual'][-1]:.3e}); at a "
                  f"dynamic pressure near divergence it will not, which is the "
                  f"answer rather than a failure")
    return fluid, sstate, hist


def _level_from_u_static(level, u):
    return {"u": np.asarray(u, dtype=float).reshape(-1, 3),
            "v": level["v"], "a": level["a"], "history": level.get("history")}


# ------------------------------------------------------ 4 the coupled step

def time_march_fsi(fluid, sstate, transfer, dt=0.05, nsteps=100, rho=tpw.RHO,
                   coupling="strong", accel="aitken", omega=OMEGA_START,
                   tol=TOL_COUPLING, max_sub=MAX_SUB, rho_inf=fes.RHO_INF,
                   nmax=None, order=usw.DPHI_DT_ORDER, p_ref=None,
                   verbose=False, callback=None):
    """March the coupled system. Returns (history, fluid, sstate).

    One step is

        v = wake_node_velocity(fluid)                 # converged level n
        shed(wake, v, dt, v_shed=te_convection_velocity(fluid))   # ONCE
        truncate(wake, nmax)
        repeat:
            update_points(fluid, displaced_points(u), point_velocities(v))
            solve(fluid, wake='frozen')
            dphi_dt = backward_difference(committed levels + this iterate, dt)
            f = structural_forces(get_loads(fluid, rho, dphi_dt))
            u_tilde = step_dynamic(level n, f)         # pure: level n intact
            u = accelerate(u, u_tilde)
        until ||u_tilde - u|| < tol
        commit the structural level and the doublet strengths

    coupling='loose' takes exactly one subiteration. It is provided because its
    instability at low mass ratio is a verification result, not because it is
    a recommended mode.

    history holds one entry per level, level zero being the initial state
    before any vorticity has been shed: t, s, CL, CD, CM, CL_quasi_steady,
    force, moment, tip displacement and its three components, strain and
    kinetic energy, the cumulative work done by the fluid, the number of
    subiterations, the final coupling residual and the wake row count.
    """
    if coupling not in ("strong", "loose"):
        raise ValueError(f"coupling must be 'strong' or 'loose'; got {coupling}")
    par = fes.integrator_parameters(rho_inf)
    points0 = np.array(fluid["panels"]["points"])
    lifting = fluid["wake"] is not None
    if sstate["K"] is None:
        fes.assemble_operators(sstate)

    # level zero: the structure where it stands, the fluid acyclic
    pts = fsx.displaced_points(transfer, points0, sstate["u"])
    vel = fsx.point_velocities(transfer, sstate["v"])
    fluid = tpw.update_points(fluid, pts, vel)
    fluid = tpw.solve(fluid, wake="frozen" if lifting else "none")
    f_struct, loads = _loads_on_structure(fluid, transfer, rho, p_ref)
    sstate["f"] = f_struct
    sstate["a"] = fes.initial_acceleration(sstate, f_struct)

    chord = fluid["chord_mean"]
    per_step = usw.convection_per_step(fluid, dt)
    if per_step > STEP_WARN_CHORDS:
        warnings.warn(
            f"the flow convects {per_step:.2f} chords per step; the wake is "
            f"under-resolved near the trailing edge and the indicial response "
            f"will be inaccurate even though the steady limit is not")
    n_per = steps_per_period(sstate, dt)
    if n_per < STEPS_PER_PERIOD_WARN:
        warnings.warn(
            f"only {n_per:.1f} steps per period of the softest structural mode; "
            f"the unresolved modes are undamped at rho_inf = 1 and their "
            f"acceleration reaches the pressure through dmu/dt, which shows as a "
            f"two-step ripple in the lift over a smooth quasi-steady part. "
            f"Resolve the mode, start from the static equilibrium, or lower "
            f"rho_inf")

    keys = ("t", "s", "CL", "CD", "CM", "CL_quasi_steady", "force", "moment",
            "tip", "tip_x", "tip_y", "tip_z", "strain_energy", "kinetic_energy",
            "fluid_work", "sub", "resid", "nrows", "mu_w_mid")
    hist = {k: [] for k in keys}
    # seeded with the acyclic level, exactly as unsteady_wing.time_march seeds
    # its own: leaving it empty makes the first step's dmu/dt zero and lags the
    # added mass by one level for the whole march
    mu_committed = [fluid["mu"].copy()]
    work = 0.0

    def _record(t, dphi, sub, resid, loads):
        res = loads["resultants"]
        pan = fluid["panels"]
        if dphi is None:
            cl_qs = res["CL"]
        else:
            d_force = (rho * dphi)[:, None] * pan["areas"][:, None] \
                * pan["normals"]
            cl_qs = (res["force"][2] - d_force[:, 2].sum()) \
                / (0.5 * rho * fluid["u_ref"] ** 2 * fluid["s_ref"])
        u_s = sstate["u"]
        big = int(np.argmax(np.linalg.norm(u_s, axis=1)))
        hist["t"].append(t)
        hist["s"].append(2.0 * fluid["u_ref"] * t / chord)
        hist["CL"].append(res["CL"])
        hist["CD"].append(res["CD_pressure"])
        hist["CM"].append(res["CM"])
        hist["CL_quasi_steady"].append(cl_qs)
        hist["force"].append(res["force"].copy())
        hist["moment"].append(res["moment"].copy())
        hist["tip"].append(float(np.linalg.norm(u_s[big])))
        for c, name in enumerate(("tip_x", "tip_y", "tip_z")):
            hist[name].append(float(u_s[big, c]))
        hist["strain_energy"].append(fes.strain_energy(sstate["model"], u_s))
        hist["kinetic_energy"].append(fes.kinetic_energy(sstate))
        hist["fluid_work"].append(work)
        hist["sub"].append(sub)
        hist["resid"].append(resid)
        wake = fluid["wake"]
        hist["nrows"].append(0 if wake is None else int(wake["mu"].shape[0]))
        hist["mu_w_mid"].append(
            float(wake["mu"][0, wake["mu"].shape[1] // 2])
            if wake is not None and wake["mu"].size else 0.0)

    _record(0.0, None, 0, 0.0, loads)
    if verbose:
        print(f"  step   0  t = 0.000  CL = {hist['CL'][0]:+.5f}  "
              f"(acyclic start, no wake)")
    scale = max(float(np.linalg.norm(np.ptp(points0.reshape(-1, 3), axis=0))),
                1e-300)
    n_sub_max = 1 if coupling == "loose" else max_sub

    for n in range(1, nsteps + 1):
        t = n * dt
        # --- shed ONCE, from the converged previous level
        if lifting:
            v_nodes = tpw.wake_node_velocity(fluid)
            tpw.shed(fluid["wake"], v_nodes, dt,
                     v_shed=tpw.te_convection_velocity(fluid))
            tpw.truncate(fluid["wake"], nmax)
            fluid["wake_version"] += 1

        acc = FixedPointAccelerator(accel, omega)
        u_k = sstate["u"] + dt * sstate["v"] + 0.5 * dt ** 2 * sstate["a"]
        level, resid, sub = None, np.inf, 0
        for sub in range(1, n_sub_max + 1):
            kin = _level_from_u(sstate, u_k, dt, par)
            fluid = tpw.update_points(
                fluid, fsx.displaced_points(transfer, points0, kin["u"]),
                fsx.point_velocities(transfer, kin["v"]))
            fluid = tpw.solve(fluid, wake="frozen" if lifting else "none")
            dphi = usw.backward_difference(mu_committed + [fluid["mu"]], dt,
                                           order)
            f_struct, loads = _loads_on_structure(fluid, transfer, rho, p_ref,
                                                  dphi_dt=dphi)
            level = fes.step_dynamic(sstate, f_struct, dt, par)
            resid = float(np.linalg.norm(level["u"] - u_k) / scale)
            if resid < tol:
                break
            u_k = acc.step(u_k, level["u"])
        if coupling == "strong" and resid >= tol:
            warnings.warn(f"step {n}: the coupling residual stalled at "
                          f"{resid:.3e} after {n_sub_max} subiterations")
        # --- commit: only now do the doublets become a time level
        work += float(np.einsum("mc,mc->", 0.5 * (sstate["f"] + f_struct),
                                level["u"] - sstate["u"]))
        fes.commit(sstate, level, f_struct)
        mu_committed.append(fluid["mu"].copy())
        if len(mu_committed) > 2:
            mu_committed.pop(0)
        _record(t, dphi, sub, resid, loads)
        if verbose:
            print(f"  step {n:3d}  t = {t:.3f}  s = {hist['s'][-1]:6.2f}  "
                  f"CL = {hist['CL'][-1]:+.5f}  tip = {hist['tip'][-1]:+.4e}  "
                  f"({sub} subiterations, residual {resid:.2e})")
        if callback:
            callback(n, t, hist, fluid, sstate)

    out = {k: np.array(v) for k, v in hist.items()}
    out.update(u_ref=float(fluid["u_ref"]), chord=float(chord), dt=float(dt),
               rho=float(rho), s_ref=float(fluid["s_ref"]),
               coupling=coupling, accel=accel)
    return out, fluid, sstate


# --------------------------------------------------------- 5 diagnostics

def steps_per_period(sstate, dt, nmodes=1):
    """Steps per period of the softest constrained mode. The second resolution
    measure, beside the fluid's own chords-convected-per-step.

    A coupled march has to resolve BOTH. Below about ten steps per structural
    period the response is not merely inaccurate: generalised-alpha does not
    damp unresolved modes at rho_inf = 1, so they ring at the Nyquist frequency
    with a tiny displacement and a large acceleration, and the acceleration is
    what the added-mass pressure sees. The symptom is a two-step ripple in the
    lift with a perfectly smooth quasi-steady part, which is exactly how it was
    found. The cures are to resolve the mode, to start from the static
    aeroelastic equilibrium instead of from rest, or to set rho_inf below one -
    in that order of preference.
    """
    freq, _ = fes.modes(sstate, nmodes)
    return float(1.0 / (max(freq[0], 1e-300) * dt))


def added_mass_ratio(fluid, sstate, transfer, rho, dt, amplitude=None):
    """Ratio of fluid added mass to structural mass along the softest mode.

    Measured, not modelled: the surface is given the softest mode shape as an
    ACCELERATION field, with no velocity and no displacement, and the fluid is
    asked for the dmu/dt pressure that results. mu is linear in the surface
    velocity, so the difference quotient over one step returns the response to
    a unit generalised acceleration and is independent of dt to leading order.
    The generalised force divided by the generalised mass of the same shape is
    then the added-mass ratio, in the normalisation the mode itself sets.

    It is the number that decides whether a staggered scheme can be used at
    all, and it costs two fluid solves. The fluid state is restored to the
    geometry it arrived with.
    """
    if sstate["K"] is None:
        fes.assemble_operators(sstate)
    freq, phi = fes.modes(sstate, 1)
    shape = phi[:, :, 0]
    points0 = np.array(fluid["panels"]["points"])
    scale = amplitude or (1e-4 * float(np.linalg.norm(
        np.ptp(points0.reshape(-1, 3), axis=0))))
    shape = shape * (scale / max(np.abs(shape).max(), 1e-300))
    mu_levels = []
    for step in (0, 1):
        disp = 0.5 * shape * (step * dt) ** 2
        vel = shape * (step * dt)
        fluid = tpw.update_points(fluid,
                                  fsx.displaced_points(transfer, points0, disp),
                                  fsx.point_velocities(transfer, vel))
        fluid = tpw.solve(fluid, wake="frozen" if fluid["wake"] is not None
                          else "none")
        mu_levels.append(fluid["mu"].copy())
    dphi = (mu_levels[1] - mu_levels[0]) / dt
    pan = fluid["panels"]
    # p = -rho dmu/dt and f = -p A n, so the added-mass force is +rho dmu/dt A n
    force = (rho * dphi)[:, None] * pan["areas"][:, None] * pan["normals"]
    f_struct = fsx.structural_forces(transfer, pan, force)
    modal_force = float(np.einsum("mc,mc->", f_struct, shape))
    modal_mass = float(shape.reshape(-1) @ (sstate["M"] @ shape.reshape(-1)))
    fluid = tpw.update_points(fluid, points0)
    fluid = tpw.solve(fluid, wake="frozen" if fluid["wake"] is not None
                      else "none")
    return {"added_mass": -modal_force, "modal_mass": modal_mass,
            "ratio": -modal_force / max(modal_mass, 1e-300),
            "frequency_dry": float(freq[0]),
            "frequency_wet_estimate": float(freq[0]) / np.sqrt(
                max(1.0 + -modal_force / max(modal_mass, 1e-300), 1e-30))}


def aerodynamic_stiffness(fluid, sstate, transfer, rho, nmodes=4, amplitude=None):
    """Reduced aerodynamic stiffness on the softest modes. (m, m)

    The static aeroelastic problem is linear in the displacement while the
    deflections are small, so it is (K - q A) u = q f_0 with A the derivative
    of the load with respect to the displacement divided by the dynamic
    pressure. A is measured here by one steady fluid solve per mode - m + 1
    solves in all, not one per degree of freedom - and reduced onto the modal
    basis, which is enough to locate a divergence because divergence is a
    property of the softest few modes by definition.

    Returns (A_r (m, m), K_r (m, m), modes (m,), shapes) with K_r = Phi^T K Phi.
    """
    if sstate["K"] is None:
        fes.assemble_operators(sstate)
    freq, phi = fes.modes(sstate, nmodes)
    points0 = np.array(fluid["panels"]["points"])
    q_dyn = 0.5 * rho * fluid["u_ref"] ** 2
    scale = amplitude or (1e-3 * float(np.linalg.norm(
        np.ptp(points0.reshape(-1, 3), axis=0))))
    fluid = tpw.update_points(fluid, points0)
    fluid = tpw.solve(fluid, wake="relax" if fluid["wake"] is not None else "none")
    f_0, _ = _loads_on_structure(fluid, transfer, rho)
    a_cols = []
    for j in range(nmodes):
        step = scale / max(np.abs(phi[:, :, j]).max(), 1e-300)
        fluid = tpw.update_points(
            fluid, fsx.displaced_points(transfer, points0, step * phi[:, :, j]))
        fluid = tpw.solve(fluid, wake="frozen" if fluid["wake"] is not None
                          else "none")
        f_j, _ = _loads_on_structure(fluid, transfer, rho)
        # f = q A u with u = step * phi_j, so A phi_j = (f_j - f_0)/(q * step)
        a_cols.append((f_j - f_0).reshape(-1) / (q_dyn * step))
    fluid = tpw.update_points(fluid, points0)
    fluid = tpw.solve(fluid, wake="frozen" if fluid["wake"] is not None
                      else "none")
    basis = phi.reshape(-1, nmodes)
    a_red = basis.T @ np.stack(a_cols, axis=1)
    k_red = basis.T @ (sstate["K"] @ basis)
    return {"A": a_red, "K": k_red, "frequency": freq, "shapes": phi,
            "f0": basis.T @ f_0.reshape(-1)}


def divergence_pressure(fluid, sstate, transfer, rho, nmodes=4, amplitude=None):
    """Smallest dynamic pressure at which the static coupled operator is singular.

    Divergence is the loss of positive definiteness of K - q A, so it is the
    smallest positive eigenvalue of K_r x = q A_r x on the modal basis. This is
    a prediction from two independently measured operators - the structure's
    own stiffness and one aerodynamic derivative per mode - and the fixed-point
    iteration of `static_aeroelastic` approaching the same number from below is
    what makes the pair a verification rather than a definition.
    """
    red = aerodynamic_stiffness(fluid, sstate, transfer, rho, nmodes, amplitude)
    # posed as K_r^-1 A_r x = (1/q) x rather than A_r^-1 K_r x = q x: K_r is
    # positive definite by construction and A_r need not even be invertible
    vals = np.linalg.eigvals(np.linalg.solve(red["K"], red["A"]))
    real = vals[np.abs(vals.imag) < 1e-6 * np.maximum(np.abs(vals.real), 1e-30)]
    positive = np.sort(real.real[real.real > 0.0])[::-1]
    red["q_divergence"] = float(1.0 / positive[0]) if len(positive) else np.inf
    red["u_divergence"] = (np.sqrt(2.0 * red["q_divergence"] / rho)
                           if np.isfinite(red["q_divergence"]) else np.inf)
    return red


def growth_rate(t, signal, skip=0.25):
    """Exponential growth rate and frequency of a decaying or growing response.

    Fitted to the peaks of the signal over its last (1 - skip) portion, which
    keeps the starting transient out. A positive rate is an instability. This
    is the measurement a flutter boundary is read from; it says nothing about
    which mode is responsible, which is what the modal content of the record is
    for.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(signal, dtype=float)
    start = int(skip * len(t))
    t, y = t[start:], y[start:] - np.mean(y[start:])
    peaks = [i for i in range(1, len(y) - 1)
             if abs(y[i]) > abs(y[i - 1]) and abs(y[i]) >= abs(y[i + 1])]
    if len(peaks) < 3:
        return {"rate": np.nan, "frequency": np.nan, "peaks": len(peaks)}
    t_p, y_p = t[peaks], np.abs(y[peaks])
    good = y_p > 1e-14 * max(y_p.max(), 1e-300)
    slope, _ = np.polyfit(t_p[good], np.log(y_p[good]), 1)
    # the frequency comes from the zero crossings, linearly interpolated, and
    # not from the peak spacing: a peak sits at a stationary point, so its
    # position is only as accurate as the step, while a crossing is where the
    # signal moves fastest and interpolating it is accurate to second order
    sign_change = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(sign_change) >= 3:
        zeros = t[sign_change] - y[sign_change] \
            * (t[sign_change + 1] - t[sign_change]) \
            / (y[sign_change + 1] - y[sign_change])
        period = 2.0 * float(np.mean(np.diff(zeros)))
    else:
        period = 2.0 * float(np.mean(np.diff(t_p)))
    return {"rate": float(slope), "frequency": float(1.0 / period),
            "peaks": len(peaks), "crossings": int(len(sign_change))}


# ----------------------------------------------------------- 6 plotting

def plot_fsi_history(history, save=None):
    """Lift, tip displacement, energies and coupling effort. Returns Figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    s = history["s"]
    ax = axes[0, 0]
    ax.plot(s, history["CL"], label="$C_L$")
    ax.plot(s, history["CL_quasi_steady"], "--", label="quasi-steady")
    ax.set_xlabel("reduced time $s = 2Ut/c$"), ax.set_ylabel("$C_L$")
    ax.legend(), ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(s, history["tip_z"], label="$u_z$")
    ax.plot(s, history["tip_x"], ":", label="$u_x$")
    ax.set_xlabel("reduced time"), ax.set_ylabel("largest nodal displacement")
    ax.legend(), ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(s, history["strain_energy"], label="strain")
    ax.plot(s, history["kinetic_energy"], label="kinetic")
    ax.plot(s, history["fluid_work"], "--", label="work by the fluid")
    ax.set_xlabel("reduced time"), ax.set_ylabel("energy")
    ax.legend(), ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.step(s, history["sub"], where="mid", label="subiterations")
    ax.set_xlabel("reduced time"), ax.set_ylabel("subiterations")
    ax2 = ax.twinx()
    ax2.semilogy(s[1:], np.maximum(history["resid"][1:], 1e-16), "r.",
                 markersize=3, label="residual")
    ax2.set_ylabel("final coupling residual", color="r")
    ax.grid(alpha=0.3)
    fig.suptitle(f"coupled history: {history['coupling']} coupling, "
                 f"{history['accel']} acceleration")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


# ---------------------------------------------------------------- 7 CLI

def main():
    """A flexible rectangular wing released into a uniform stream."""
    import matplotlib.pyplot as plt
    import fem_materials as fmat
    import fem_mesh
    import make_thick_sample_inputs as mts

    print("fsi_driver.py - panel method coupled to a finite element solid\n")
    span = float(input("span [6.0]: ") or 6.0)
    thickness = float(input("thickness/chord [0.12]: ") or 0.12)
    alpha = float(input("incidence, degrees [5.0]: ") or 5.0)
    e_mod = float(input("Young's modulus [2.0e8]: ") or 2.0e8)
    rho_s = float(input("structural density [1200.0]: ") or 1200.0)
    rho_f = float(input("fluid density [1000.0]: ") or 1000.0)
    dt = float(input("time step [0.05]: ") or 0.05)
    nsteps = int(input("steps [40]: ") or 40)

    points = mts.thick_wing_mesh(n_c=8, nspan=8, span=span, root_chord=1.0,
                                 taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                                 camber=0.0, thickness=thickness)
    points = tpw.pitch_mesh(points, np.radians(alpha))
    z, u = np.array([-10.0, 10.0]), np.array([1.0, 1.0])
    onset = tpw.make_onset(z, u)

    mesh = fem_mesh.solid_foil_mesh(points, n_thick=2)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(e_mod, 0.3, rho_s))
    fluid, sstate, transfer = init_fsi(points, onset, model, u_ref=1.0,
                                       rho=rho_f)
    fes.clamp(sstate, mesh["node_sets"]["root"])
    fes.assemble_operators(sstate)
    print(f"\n{len(fluid['panels']['areas'])} panels, "
          f"{len(mesh['elements'])} elements, "
          f"{3 * len(mesh['nodes'])} structural degrees of freedom")
    print(f"transfer offset {transfer['offset_max']:.3e} "
          f"({transfer['offset_rel']:.1e} of the bounding box)")
    freq, _ = fes.modes(sstate, 3)
    print("dry frequencies: " + ", ".join(f"{f:.4f} Hz" for f in freq))
    ratio = added_mass_ratio(fluid, sstate, transfer, rho_f, dt)
    print(f"added mass / structural mass {ratio['ratio']:.3f}, so the wet "
          f"frequency is near {ratio['frequency_wet_estimate']:.4f} Hz")
    print(f"{steps_per_period(sstate, dt):.1f} steps per dry structural period "
          f"and {usw.convection_per_step(fluid, dt):.3f} chords convected per step")

    # start from the static aeroelastic equilibrium rather than from rest: a
    # structure released from rest rings in every mode it has, including the
    # ones the step cannot resolve, and their acceleration reaches the pressure
    # through dmu/dt. This is the first of the two cures docs/COUPLING.md gives.
    print("\nstatic aeroelastic equilibrium:")
    fluid, sstate, static = static_aeroelastic(fluid, sstate, transfer,
                                               rho=rho_f, verbose=True)
    print(f"  {'converged' if static['converged'] else 'DID NOT CONVERGE'} in "
          f"{static['iterations']} iterations")

    print("\ncoupled time march:")
    history, fluid, sstate = time_march_fsi(fluid, sstate, transfer, dt=dt,
                                            nsteps=nsteps, rho=rho_f,
                                            rho_inf=0.9, verbose=True)
    print(f"\nfinal C_L {history['CL'][-1]:+.5f}, "
          f"largest displacement {history['tip'][-1]:.5e}, "
          f"mean subiterations {history['sub'][1:].mean():.2f}")
    plot_fsi_history(history)
    fes.plot_deformed(model, sstate["u"],
                      scale=0.2 * span / max(np.abs(sstate["u"]).max(), 1e-30))
    plt.show()


if __name__ == "__main__":
    main()
