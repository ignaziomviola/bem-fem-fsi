"""The ventilated solve: the free-surface image, the cavity, and their fixed point.

The system half of the seam to the vendored solver. Everything here is one
composition of `thick_panel_wing.assemble_system` under two masks - the image,
which flips the sign of the source strengths on the mirror half, and the cavity,
which swaps the unknown from the doublet strength to the source strength on the
cavity panels - and the fixed point that closes the cavity.

Seven dependency-ordered sections, and the same three discipline rules: nothing
imports downward, every block is a pure function over explicit arguments with all
mutable data in an explicit state dictionary, and prompts, prints and figures
appear only under main().

    1 constants     the wake relaxation limits
    2 imports       numpy; thick_panel_wing, vent_mesh, vent_cavity, vent_loads
    3 composition   blocks, wake influence, and the masked solve
    4 the wetted    the image-only solve, which replaces tpw.solve
    5 the cavity    the dynamic condition, and the fixed point on the length
    6 relaxation    the wake, which this module must own
    7 diagnostics   antisymmetry, and the washout margin

The image and the cavity live in one module because they are two masks over one
linear system and they INTERACT: the cavity mask has to be mirror-symmetric, or
the antisymmetry that makes the free surface exact is broken by the cavity.

Five decisions are worth stating before the code.

**The vendored `solve_once` cannot be used, and its own docstring says so** - "the
cavity driver reuses it for its wetted baseline and composes `assemble_system`
directly thereafter". It recomputes the source strengths from `-n.u_rel`, which
is exactly what the image has to override. Section 3 is the twenty-five lines
that replaces it.

**On a doubled mesh, the vendored solve IS the rigid-wall answer.** For an onset
with no spanwise component, mirroring leaves `-n.u_rel` unchanged, so the
unflipped source strengths are the positive image: `tpw.solve` on a ventilating
state returns the zero-Froude, rigid-wall solution, silently and plausibly, with
about twice the lift and its peak loading at the waterline instead of zero there.
That is measured, not feared - `test_vent` asserts the bit-for-bit agreement
between the positive image and `tpw.solve`, so the hazard is on the record.

**This module owns the wake relaxation, and must.** The vendored `solve`'s
docstring warns that a driver composing its own system must; here there is a
second reason, which is that the image wake has to be projected after every
convection rather than advected. See `vent_mesh.symmetrise_wake`.

**The cavity iteration is cold-started from the committed length every call.** A
warm start from the previous coupling subiteration would make the converged cavity
depend on the iteration path at the level of the cavity tolerance, and the outer
quasi-Newton accelerator differentiates that noise. The cost is a few extra
iterations and it buys the property the whole partitioned scheme rests on: the
load is a deterministic function of the displacement and the committed regime.

**The cavity condition enters the system on panels more than half covered, while
the pressure blends continuously.** The system mask has to be binary, so
something must switch discretely somewhere. It is placed at the panel midpoint
because at the converged length the cavity pressure equals the wetted pressure at
the closure point - that is what determines the length - so the prescribed and
the solved doublet strength agree there to first order and the switch is smooth.
The load's continuity in the cavity length is then measured, not assumed.
"""

# ------------------------------------------------------------- 1 constants

import numpy as np

import thick_panel_wing as tpw
import vent_mesh as vm
import vent_cavity as vcv
import vent_loads as vnl
import vent_section as vs

MAX_WAKE_ITER = 12             # wake relaxation sweeps; the vendored default
WAKE_TOL = 1e-4                # wake displacement, in chords
MASK_SWITCH = 0.5              # coverage at which a panel joins the cavity system
GAIN_THICKNESS = 4.0           # first step of the closure secant, before it has a
                               # pair to work from; the residual is a thickness
                               # over a chord and the length is a chord fraction,
                               # so the gain is O(chord/thickness) ~ O(10)


# ---------------------------------------------------------- 3 composition

def _system(fluid, vent, wake_mode):
    """Geometry blocks and wake influence, assembled once per geometry. -> dict

    Held outside the cavity iteration deliberately: the cavity changes only
    WHICH columns of D and S are selected, so the O(N^2) assembly and the wake
    influence are paid once per coupling subiteration and the cavity loop costs
    one dense solve per iteration.
    """
    pan = fluid["panels"]
    if fluid["blocks"] is None or fluid["blocks_version"] != fluid["geom_version"]:
        fluid["blocks"] = tpw.assemble_body(pan)
        fluid["blocks_version"] = fluid["geom_version"]
    p_bound, phi_known = (None, None)
    if fluid["wake"] is not None and wake_mode != "none":
        p_bound, phi_known = tpw.wake_influence(pan["centroids"], fluid["wake"],
                                                fluid["onset"])
    return {"D": fluid["blocks"]["D"], "S": fluid["blocks"]["S"],
            "p_bound": p_bound, "phi_known": phi_known}


def image_sigma(pan, vent, u_rel, sign=-1.0):
    """Source strengths with the image half overridden. -> (N,)

    sigma = -n.u_rel on the immersed half, and MINUS the mirror of that on the
    image half. The flip is the whole free surface: it makes the potential
    antisymmetric, hence zero on the plane y = y_fs, which is the linearised
    high-Froude free-surface condition. sign = +1 gives the zero-Froude rigid
    wall instead, and for a y-independent onset that is bit-for-bit what the
    vendored solve produces on its own.
    """
    sigma = tpw.source_strengths(pan["normals"], u_rel)
    if not vent["image"]:
        return sigma
    out = sigma.copy()
    mirrored = sigma[np.asarray(vent["panel_mirror"])]
    image = np.asarray(vent["image_mask"], dtype=bool)
    out[image] = sign * mirrored[image]
    return out


def mixed_solve(fluid, vent, sys, sigma, unknown_type=None, mu_fixed=None):
    """The masked Dirichlet system, solved. -> (mu (N,), sigma (N,))

    unknown_type[k] = 1 puts the source strength of panel k among the unknowns
    and its doublet strength among the knowns, which is the cavity condition;
    all-zero is the wetted system and reproduces the vendored solve bit for bit.
    The vendored trailing-edge fold already handles a cavity panel that is also a
    Kutta panel, sending its prescribed strength to the right-hand side.
    """
    pan = fluid["panels"]
    n = len(pan["areas"])
    unknown_type = np.zeros(n, dtype=np.int8) if unknown_type is None \
        else np.asarray(unknown_type, dtype=np.int8)
    mu_fixed = np.zeros(n) if mu_fixed is None else np.asarray(mu_fixed, float)
    a_mat, rhs = tpw.assemble_system(sys["D"], sys["S"], sys["p_bound"],
                                     sys["phi_known"], pan["k_up"], pan["k_low"],
                                     unknown_type, mu_fixed, sigma)
    x = np.linalg.solve(a_mat, rhs)
    mask = unknown_type == 1
    mu = np.where(mask, mu_fixed, x)
    sig = np.where(mask, x, sigma)
    return mu, sig


def _commit_strengths(fluid, vent, mu, sigma):
    """Write the strengths into the fluid state and sync the wake. -> fluid"""
    fluid["mu"], fluid["sigma"] = mu, sigma
    if fluid["wake"] is not None:
        tpw.set_bound_strengths(fluid["wake"], mu, fluid["panels"]["k_up"],
                                fluid["panels"]["k_low"])
    fluid["vent_sign"] = -1.0 if vent["image"] else 0.0
    return fluid


# ------------------------------------------------------------ 4 the wetted

def solve_wetted(fluid, vent, wake="frozen", sign=-1.0):
    """One wetted solve with the free-surface image. Replaces tpw.solve. -> fluid

    wake='frozen' is the per-step entry the coupled march uses; 'relax' iterates
    the wake geometry through section 6, which this module owns; 'none' is the
    non-lifting body.
    """
    if wake not in ("frozen", "relax", "none"):
        raise ValueError(f"unknown wake mode '{wake}'; choose from 'frozen', "
                         f"'relax', 'none'")
    if wake == "relax":
        return relax_wake(fluid, vent, sign=sign)
    sys = _system(fluid, vent, wake)
    sigma = image_sigma(fluid["panels"], vent, fluid["u_rel"], sign)
    mu, sig = mixed_solve(fluid, vent, sys, sigma)
    return _commit_strengths(fluid, vent, mu, sig)


# ------------------------------------------------------------ 5 the cavity

def dynamic_mu(pan, frame, vent, weight, mu_wet, u_rel, v_span=None):
    """The doublet strength prescribed on the cavity by the dynamic condition.

    The pressure inside the cavity is known, so Bernoulli fixes the total surface
    speed at q_c = u sqrt(1 + sigma_c); the chordwise component follows from it
    with the spanwise component lagged, and

        dmu/ds = V_s - u_rel . s_hat

    integrated along the chordwise chain from the detachment panel, where mu is
    continuous with the wetted solution. Detachment is FIXED at the leading edge:
    the paper's photographs show the cavity wall detaching there in fully
    ventilated flow, so the Villat-Brillouin smooth-detachment search is not
    needed and is not implemented. -> (N,) with the wetted value off the cavity
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    q_c = vnl.cavity_speed(pan, vent)
    v_span = np.zeros(nwrap * nspan) if v_span is None \
        else np.asarray(v_span, dtype=float)
    v_s = np.sqrt(np.maximum(q_c ** 2 - v_span ** 2, 0.0))
    # along the panel's OWN unit tangent, which is the direction the vendored
    # wrap stencil differentiates in, and which on the suction side runs from the
    # leading edge aft - the direction the cavity flow takes
    t_1 = np.asarray(pan["frames"])[:, 0, :]
    grad = v_s - np.einsum("kc,kc->k", np.asarray(u_rel, dtype=float), t_1)
    index = np.arange(nwrap * nspan).reshape(nwrap, nspan)
    xi = frame["xi"].reshape(nwrap, nspan)
    w = np.asarray(weight, dtype=float).reshape(nwrap, nspan)
    out = np.asarray(mu_wet, dtype=float).reshape(nwrap, nspan).copy()
    for j in range(nspan):
        live = np.where(w[:, j] > 0.0)[0]
        if live.size == 0:
            continue
        order = live[np.argsort(xi[live, j])]
        chain = index[order, j]
        step = vcv.wrap_spacing(pan, chain[:-1], chain[1:])
        inc = 0.5 * (grad[chain[:-1]] + grad[chain[1:]]) * step
        out[order, j] = out[order[0], j] + np.concatenate([[0.0], np.cumsum(inc)])
    return out.ravel(), v_s


def _extent_target(fluid, vent, frame, cav, rho, length, detach, memory):
    """Detachment and cavity length the current state asks for. -> (detach, length)

    `length` is the CURRENT iterate, not the committed one: the thickness rule is
    a secant on the closure residual and has to move with the iteration.
    """
    pan = fluid["panels"]
    rule = vent["extent_rule"]
    if rule == "frozen":
        return detach, np.asarray(length, dtype=float)
    if rule == "pressure":
        # on the BASELINE wetted pressure, never on the pressure the cavity has
        # itself imposed: the dynamic condition makes the latter equal the cavity
        # pressure on the cavity, so the margin would be identically zero there
        return vcv.pressure_target(pan, frame, cav["cp_baseline"], cav["cp_cav"],
                                   vent["alpha_rad"], vent["real_mask"])
    if rule == "section":
        depth, cl = vnl.depth_loading(pan, vent, cav["p_gauge"], rho)
        sigma_c = vs.sigma_cavity(depth, vent["h"], vent["fn_h"], vent["dsigma"])
        target = np.zeros(pan["nspan"])
        wet = np.asarray(vent["real_mask"], bool).reshape(pan["nwrap"],
                                                          pan["nspan"])[0]
        target[wet] = vcv.section_target(cl, sigma_c, np.asarray(length)[wet],
                                         vent["length_model"])[::-1]
        new_detach, _ = vcv.pressure_target(pan, frame, cav["cp_baseline"],
                                            cav["cp_cav"], vent["alpha_rad"],
                                            vent["real_mask"])
        return new_detach, np.minimum(target, 1.0)
    # 'thickness': lengthen while the cavity has not closed, shorten when it has
    # overshot. The residual is the thickness at the closure point and changes
    # sign at the right length, so a secant with one step of memory converges;
    # where the cavity has not started at all, the pressure rule opens it.
    length = np.asarray(length, dtype=float)
    new_detach, base = vcv.pressure_target(pan, frame, cav["cp_baseline"],
                                           cav["cp_cav"], vent["alpha_rad"],
                                           vent["real_mask"])
    resid = vcv.closure_thickness(pan, frame, cav["thickness"], cav["weight"])
    prev_l, prev_r = memory
    secant = np.full_like(length, np.nan)
    if prev_l is not None:
        d_r = resid - prev_r
        good = np.abs(d_r) > 1e-14
        secant = np.where(good, length - resid * (length - prev_l)
                          / np.where(good, d_r, 1.0), np.nan)
    fallback = length + GAIN_THICKNESS * resid
    target = np.where(np.isfinite(secant), secant, fallback)
    return new_detach, np.where(length > 0.0, np.clip(target, 0.0, 1.0), base)


def solve_cavity(fluid, vent, rho=1.0, dphi_of_mu=None, wake="frozen",
                 sign=-1.0, dt=None):
    """The ventilated solve: image, cavity, and the fixed point on the length.

    Returns (fluid, cav, loads). `cav` is the ITERATE - the pressure, the mask,
    the thickness, the closure line, the residual - and is deliberately not part
    of the `vent` state: nothing that changes within a coupling subiteration is
    stored, which is what keeps the load a fixed function of the displacement.
    `commit_vent` takes what survives.
    """
    pan = fluid["panels"]
    frame = vcv.chordwise_frame(pan)
    sys = _system(fluid, vent, wake)
    sigma_wet = image_sigma(pan, vent, fluid["u_rel"], sign)
    real = np.asarray(vent["real_mask"], dtype=bool)

    # the wetted baseline: needed for the separation indicator, for the pressure
    # margin, and for mu at detachment
    mu_wet, _ = mixed_solve(fluid, vent, sys, sigma_wet)
    dphi = None if dphi_of_mu is None else dphi_of_mu(mu_wet)
    wet = vnl.vent_pressure(pan, mu_wet, fluid["u_rel"], vent["u_ref"], vent,
                            np.zeros(len(sigma_wet)), dphi_dt=dphi, rho=rho)
    sep = vcv.separated(pan, wet["cp_solved"], vent["alpha_rad"],
                        vent["recovery"], real)
    # ventilation-ready: the wetted pressure lies BELOW the cavity pressure, so a
    # cavity there would raise the pressure. The margin's sign is the whole
    # direction of the effect and is asserted with no tolerance in test_vent.
    ready = (wet["cp_cav"] - wet["cp_solved"]) > 0.0
    band = vcv.waterline_band(pan, vent)
    # Separation gates INCEPTION, not each panel of an existing cavity. The paper
    # is explicit that entrained air modifies the local pressure gradients and so
    # propagates separation ahead of itself, the cavity growing into flow that the
    # cavity itself separated; gating every panel on the wetted separation
    # indicator would forbid the cavity from ever reaching the leading edge, where
    # its own photographs show it detaching.
    #
    # The free-surface SEAL is a separate question, and it is the stall angle that
    # answers it. At sub-stall incidence the paper's oil-film visualisations show
    # the separation bubble stopping short of the free surface, leaving a thin
    # layer of attached flow that seals the ventilation-prone flow from the air; a
    # panel method cannot resolve that layer, so the seal is taken to break at the
    # stall angle, which is exactly the vertical stall boundary of the paper's
    # regime map, or when air is injected. Once ventilated, the cavity holds its
    # own path open and the question is not asked again - that asymmetry is the
    # hysteresis.
    seal_broken = bool(abs(vent["alpha_rad"]) >= vent["alpha_stall"]
                       or vent.get("inject_active", False)
                       or vent["regime"] != "FW")
    candidate = real & ready
    connected = vcv.air_path(pan, candidate, band & candidate) if seal_broken \
        else np.zeros_like(candidate)
    n_ready = int(np.count_nonzero(connected))
    suction = vcv.suction_side(pan, vent["alpha_rad"]) & real
    ready_fraction = float(np.asarray(pan["areas"])[connected].sum()
                           / max(np.asarray(pan["areas"])[suction].sum(), 1e-30))

    length = np.asarray(vent["l_c"], dtype=float).copy()      # COLD start
    detach = np.asarray(vent["detach"], dtype=float).copy()
    # the cavity front advances at a finite speed, so within one time step it can
    # only move so far from the committed extent. This bounds the LOAD, not merely
    # the next step's starting point: the equilibrium extent is where the cavity is
    # heading, and a step that jumped straight to it would apply an instantaneous
    # load step whose structural response is a property of dt rather than of the
    # flow. With dt=None - the steady path - the bound is off.
    lo, hi = 0.0, 1.0
    if dt is not None and vent["growth_chords"] > 0.0:
        cap = (vent["growth_chords"] * vent["u_ref"] * float(dt)
               / max(vent["chord"], 1e-30))
        lo, hi = np.maximum(length - cap, 0.0), np.minimum(length + cap, 1.0)
    if vent["regime"] == "FW":
        length = np.zeros_like(length)
    reachable = np.zeros(pan["nspan"], dtype=bool)
    conn_by_strip = connected.reshape(pan["nwrap"], pan["nspan"])
    reachable[:] = conn_by_strip.any(axis=0)
    ext = np.median(vcv.wrap_extent(pan, frame))
    mu, sig, cav = mu_wet, sigma_wet, None
    first, flip, resid, flips, sign_prev = None, False, 0.0, 0, None
    iterations = 0
    # the spanwise surface velocity, lagged one iteration: the dynamic condition
    # fixes the total speed and the chordwise component follows from it, so the
    # spanwise one has to come from somewhere. It converges in two iterations.
    span_dir = np.asarray(pan["dual"])[:, 1, :]
    span_dir = span_dir / np.maximum(np.linalg.norm(span_dir, axis=1,
                                                    keepdims=True), 1e-30)
    v_span = np.zeros(len(sigma_wet))
    memory = (None, None)          # one step of secant memory on the closure
    for iterations in range(1, vent["sub_max"] + 1):
        weight, _ = vcv.mask_from_lengths(pan, frame, length, vent["alpha_rad"],
                                          real, detach)
        weight = np.where(connected, weight, 0.0)
        if vent["image"]:                    # the cavity mask must be symmetric
            weight = np.maximum(weight, weight[np.asarray(vent["panel_mirror"])])
        if vent["closure"] == "dirichlet" and np.any(weight > MASK_SWITCH):
            mu_pre, v_s = dynamic_mu(pan, frame, vent, weight, mu_wet,
                                     fluid["u_rel"], v_span)
            if vent["image"]:
                mirror = np.asarray(vent["panel_mirror"])
                image = np.asarray(vent["image_mask"], dtype=bool)
                mu_pre = np.where(image, -mu_pre[mirror], mu_pre)
            utype = (weight > MASK_SWITCH).astype(np.int8)
            mu, sig = mixed_solve(fluid, vent, sys, sigma_wet, utype, mu_pre)
        else:
            v_s = vnl.cavity_speed(pan, vent)
            mu, sig = mu_wet, sigma_wet
        dphi = None if dphi_of_mu is None else dphi_of_mu(mu)
        fields = vnl.vent_pressure(pan, mu, fluid["u_rel"], vent["u_ref"], vent,
                                   weight, dphi_dt=dphi, rho=rho)
        v_span = np.einsum("kc,kc->k", fields["v_surf"], span_dir)
        thick = vcv.thickness(pan, frame, sig, sigma_wet, weight, v_s)
        cav = dict(fields)
        cav.update({"weight": weight, "l_c": length.copy(), "thickness": thick,
                    "closure_residual": vcv.closure_residual(pan, sig, sigma_wet,
                                                             weight),
                    "n_ready": n_ready, "connected": connected,
                    "separated": sep, "v_s": v_s, "dphi_dt": dphi,
                    "cp_baseline": wet["cp_solved"],
                    "interior": vcv.cavity_interior(pan, weight),
                    "ready": ready, "seal_broken": seal_broken,
                    "ready_fraction": ready_fraction,
                    "entrainment": vcv.entrainment(pan, sig, sigma_wet, weight)})
        new_detach, target = _extent_target(fluid, vent, frame, cav, rho, length,
                                            detach, memory)
        memory = (length.copy(),
                  vcv.closure_thickness(pan, frame, cav["thickness"], weight))
        detach = np.where(reachable, new_detach, detach)
        target = np.where(reachable, target, 0.0)
        if vent["regime"] == "FW":
            target = np.zeros_like(target)
        # one panel width per iteration at most: the extent map is non-monotonic
        # near L = 0.5, where Acosta's relation turns over, and an unlimited step
        # would jump the branch instead of stopping at it
        step = np.clip(target - length, -ext, ext)
        new = np.clip(length + vent["sub_omega"] * step, lo, hi)
        resid = float(np.abs(new - length).max())
        if first is None:
            first = max(resid, 1e-300)
        sign_now = np.sign(step[np.argmax(np.abs(step))]) if np.any(step) else 0.0
        if sign_prev is not None and sign_now != 0.0 and sign_now != sign_prev:
            flips += 1
            flip = flips >= 2      # a persistent reversal is the branch boundary
        sign_prev = sign_now
        length = new
        if vent["extent_rule"] == "frozen" or flip:
            break
        # measured against the FIRST change of the call, not against an absolute
        # scale: a fully wetted or fully ventilated call starts at zero change
        if resid < vent["sub_tol"] or resid < 1e-6 * first:
            break
    cav["l_c"] = length
    cav["detach"] = detach
    geom = vcv.closure_geometry(pan, frame, length, vent)
    cav.update({"d_cav": geom["d_cav"], "phi_bar": geom["phi_bar"],
                "closure": geom, "iterations": iterations, "resid": resid,
                "branch_flip": flip, "frame": frame})
    fluid = _commit_strengths(fluid, vent, mu, sig)
    loads = vnl.immersed_loads(fluid, vent, cav["p_gauge"], rho)
    loads["cav"] = cav
    return fluid, cav, loads


# ----------------------------------------------------------- 6 relaxation

def relax_wake(fluid, vent, max_iter=MAX_WAKE_ITER, tol=WAKE_TOL, sign=-1.0):
    """Steady wetted solve with a relaxed wake, owned here rather than vendored.

    The vendored relaxation would recompute the source strengths and lose the
    image; and `wake_node_velocity` advects the wake in the field of the doubled
    body, which is right for the immersed half and wrong for the image half,
    whose position is not a degree of freedom. So each sweep solves, convects and
    then PROJECTS the image wake back onto the mirror of the real one.
    """
    for _ in range(max(int(max_iter), 1)):
        fluid = solve_wetted(fluid, vent, "frozen", sign)
        if fluid["wake"] is None:
            break
        v_nodes = tpw.wake_node_velocity(fluid)
        moved = tpw.relax_wake_step(fluid["wake"], v_nodes, 0.5, fluid["x_near"])
        vm.symmetrise_wake(fluid["wake"], vent)
        fluid["wake_version"] += 1
        if moved is not None and float(np.max(np.abs(moved))) \
                < tol * fluid["chord_mean"]:
            break
    return solve_wetted(fluid, vent, "frozen", sign)


def shed_and_project(fluid, vent, dt, nmax=None):
    """One wake step: shed, truncate, and project the image wake.

    Called ONCE per time step, outside the coupling subiteration, exactly as the
    vendored `shed` requires. The projection belongs here, immediately after the
    convection that made it necessary.

    Returns the discarded waterline drift rather than storing it, so that nothing
    outside `commit_vent` writes to the ventilation state. -> (fluid, drift_y)
    """
    if fluid["wake"] is None:
        return fluid, 0.0
    v_nodes = tpw.wake_node_velocity(fluid)
    tpw.shed(fluid["wake"], v_nodes, dt,
             v_shed=tpw.te_convection_velocity(fluid))
    if nmax is not None:
        tpw.truncate(fluid["wake"], nmax)
    drift = vm.symmetrise_wake(fluid["wake"], vent)["drift_y"]
    fluid["wake_version"] += 1
    return fluid, float(drift)


# ---------------------------------------------------------- 7 diagnostics

def antisymmetry_residual(fluid, vent, probe=None):
    """How exactly the image is an image. -> dict of relative residuals

    mu and sigma must be MINUS the mirror of themselves, to round-off: it is an
    index permutation of an imposed constraint, so anything above round-off means
    the antisymmetry is being solved for rather than held. phi on the plane
    follows from it by pairwise cancellation and is the free-surface condition
    itself.
    """
    if not vent["image"]:
        return {"mu": 0.0, "sigma": 0.0, "wake_mu": 0.0, "phi_plane": 0.0}
    pan = fluid["panels"]
    mirror = np.asarray(vent["panel_mirror"])
    out = {}
    for key in ("mu", "sigma"):
        f = np.asarray(fluid[key], dtype=float)
        out[key] = float(np.abs(f + f[mirror]).max()
                         / max(np.abs(f).max(), 1e-300))
    out["wake_mu"] = vm.symmetrise_wake(fluid["wake"], vent)["mu_residual"] \
        if fluid["wake"] is not None else 0.0
    if probe is None:
        span = np.ptp(pan["points"][..., 0])
        gx, gz = np.meshgrid(np.linspace(-span, 2.0 * span, 9),
                             np.linspace(-span, span, 7))
        probe = np.stack([gx.ravel(), np.full(gx.size, vent["y_fs"]),
                          gz.ravel()], axis=1)
        keep = np.linalg.norm(probe - pan["centroids"].mean(axis=0), axis=1) \
            > 0.5 * span
        probe = probe[keep]
    d_blk = tpw.doublet_potential_matrix(probe, pan["corners"])
    s_blk = tpw.source_potential_matrix(probe, pan["corners"])
    phi = d_blk @ fluid["mu"] + s_blk @ fluid["sigma"]
    scale = max(float(np.abs(fluid["mu"]).max()), 1e-300)
    out["phi_plane"] = float(np.abs(phi).max() / scale)
    return out


def washout_margin(vent, cl):
    """Signed distance from the washout boundary (4.5). -> float

    Negative means a fully ventilated cavity cannot be sustained at this lift and
    Froude number. Small in magnitude means the regime is metastable and a
    marched run will flip, which is what the driver warns on.
    """
    if abs(cl) < 1e-12:
        return np.inf
    return vs.washout_margin(vent["fn_h"], abs(cl), vent["ar"])


def report(fluid, vent, cav, rho=1.0):
    """Everything worth asking a ventilated solve. -> dict of measured scalars"""
    out = vcv.cavity_report(fluid["panels"], cav["frame"], vent, cav)
    out.update(antisymmetry_residual(fluid, vent))
    loads = vnl.immersed_loads(fluid, vent, cav["p_gauge"], rho)
    res = loads["resultants"]
    out.update({"CL": res["CL"], "CD": res["CD_pressure"], "CM": res["CM"],
                "drift_y": vent["drift_y"]})
    out["washout_margin"] = washout_margin(vent, out["CL"])
    return out
