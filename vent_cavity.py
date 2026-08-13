"""The ventilated cavity on a panel mesh: extent, inception, and the regime.

The panel-level half of the ventilation model. It owns the `vent` state - the
fifth state of this repository, beside `fluid`, `model`, `sstate` and `transfer` -
and it is the only module that mutates it, and only in `commit_vent`, exactly as
`fem_solid.commit` is the only thing that advances the structural level.

Eight dependency-ordered sections, and the same three discipline rules: nothing
imports downward, every block is a pure function over explicit arguments with all
mutable data in an explicit state dictionary so two states coexist freely, and
prompts, prints and figures appear only under main().

    1 constants     tolerances of the cavity iteration and the flood fill
    2 imports       numpy; vent_section. NOT thick_panel_wing - see below
    3 geometry      the chordwise frame, the depth field, which side is suction
    4 extent        lengths <-> the ventilated panel set, with fractional weights
    5 inception     separation, the air path, and the injection override
    6 closure       the closure line, its mean angle, and the flow regime
    7 the state     build_vent, commit_vent, and the branch controls
    8 report        what a misbehaving ventilated run should be asked

**This module does not import thick_panel_wing.** It consumes a panels dictionary
as data - centroids, normals, areas, ds_wrap, the four adjacency arrays, k_up and
k_low - so the flood fill, the closure-line fit and the regime machine are all
testable on a hand-built four-by-four panels dictionary with no solver in the
loop, which is where the sign and connectivity errors will actually be found.

Five decisions are worth stating before the code, because each looks like an
over-complication and none is.

**The cavity is parameterised by a continuous per-station length, not by a
boolean panel mask.** A boolean mask makes the load a piecewise-constant function
of the displacement: the cavity chatters between two adjacent panels, the
coupling residual is discontinuous at the panel scale, IQN-ILS fits its
least-squares system to a staircase, and the coupling cannot converge below one
panel width at any tolerance. With the length continuous and the closure panel
carrying a fractional weight, the map is continuous and piecewise smooth, the
outer accelerator sees what it expects, and the length resolves to a fraction of
a panel. Everything else here is secondary to that.

**The regime advances only on commit.** The same rule as the doublet history and
for the same reason, plus a stronger one: the flow regimes are BI-STABLE, so a
regime that flipped inside a coupling subiteration would not merely make the load
path-dependent, it would stop the load being a function of the displacement at
all. `transition` is called only from `commit_vent`.

**The hysteresis is structural, not a tolerance band.** Inception requires an air
path, a sub-cavity pressure AND the stall gate or an injection; persistence
requires only that ventilation-ready panels remain. So a cavity survives to
angles well below the one that started it, which is the paper's own mechanism and
not a numerical dodge. A single threshold with a dead band would reproduce a
sweep and fail `test_vent.TestRegime`.

**The air path is seeded on separation at the waterline, not on pressure.** With
the negative free-surface image the loading, and hence Cp, goes to zero at the
free surface, and for natural ventilation sigma_c(0) = 0 as well. The criterion
Cp < -sigma_c therefore reads 0 < 0 exactly where the air has to get in, and a
flood fill seeded on it may never start. The physics says the same thing: the air
source is the free surface wherever the surface-piercing station is separated.
Pressure then propagates the cavity downward and aft.

**Separation is modelled, not computed, and says so.** A panel method has no
boundary layer. The stall angle and the pressure-recovery fraction are INPUTS,
they live in the state dictionary where they can be seen, and every verification
case prints them as inputs. No constant of this model may be invisible.
"""

# ------------------------------------------------------------- 1 constants

import numpy as np

import vent_section as vs

TOL_LENGTH = 1e-5              # convergence of the cavity length, in chords. A
                               # hundred-thousandth of a chord is far below the
                               # panel scale and below anything physical; the
                               # iteration limit-cycles at about 1e-6 because the
                               # crossing interpolation is not differentiable at a
                               # panel edge, and chasing that is pointless.
MAX_CAVITY = 40                # cavity iterations; the map contracts, so this is
                               # a backstop and not a working limit
OMEGA_CAVITY = 0.5             # relaxation of the length update. The extent map
                               # is non-monotonic near L = 0.5, where Acosta's
                               # relation turns over, and THAT TURNOVER IS THE
                               # BI-STABILITY: it must not be resolved inside a
                               # subiteration, so a persistent sign change stops
                               # the iteration and is deferred to commit.
STEP_CAP = 1.0                 # largest length change per iteration, in panels
RECOVERY = 0.35                # fraction of the peak-to-trailing-edge pressure
                               # rise that counts as separated. An INPUT.
TOL_REGIME = 1e-6              # depth tolerance of the D = 0 and D = h tests
GROWTH_CHORDS = 1.0            # how fast the cavity front may advance, in chords
                               # of cavity per chord of travel. The paper's time
                               # histories show formation taking a finite time -
                               # of order a tenth of a second on a 0.1 m chord at
                               # a few metres per second, so of order one chord of
                               # travel - because the cavity grows by propagating
                               # separation ahead of itself. Without a limit the
                               # committed cavity jumps from nothing to full
                               # coverage in ONE step, which is an instantaneous
                               # load step no time integrator can be asked to
                               # follow: it rings the structure at an amplitude
                               # set by the step size rather than by the physics.
                               # An INPUT, and the one that sets how a transition
                               # is resolved in time.
INCEPT_FRACTION = 0.05         # fraction of the immersed suction area that has to
                               # be connected and ventilation-ready before
                               # inception is declared, so that one marginal panel
                               # cannot trigger it. An INPUT.
REGIMES = ("FW", "PV", "FV")   # stored as an index so the history array is numeric


# ---------------------------------------------------------------- 3 geometry

def panel_depth(pan, y_fs=0.0, axis=1):
    """Depth of each panel below the free surface, z' = y_fs - y. -> (N,)

    Positive below the surface and negative above it, so the image half of a
    doubled mesh carries negative depths and is excluded by sign alone.
    """
    return y_fs - np.asarray(pan["centroids"])[:, axis]


def suction_side(pan, alpha_rad):
    """Which panels are on the suction surface. -> (N,) bool

    The wrap index runs lower-surface trailing edge (i = 0) to leading edge
    (i = n_c) to upper-surface trailing edge, so the upper surface is i >= n_c.
    At positive incidence that is the suction side. Taken from the index rather
    than from the normal because near the leading edge the normal points forward
    and its lift component changes sign on both surfaces.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    i_of = np.repeat(np.arange(nwrap), nspan)
    upper = i_of >= nwrap // 2
    return upper if alpha_rad >= 0.0 else ~upper


def chordwise_frame(pan):
    """Chordwise coordinate of every panel, and the section chord. -> dict

    xi is the panel centroid projected onto its own section's chord line and
    normalised, so xi = 0 at the leading edge and 1 at the trailing edge, and a
    cavity length in xi IS Lc/c. Computed per spanwise station from the leading-
    and trailing-edge nodes of that station, so it follows a yawed, deflected
    section without assuming the chord lies along x.

    Keys: xi (N,), chord (nspan,), le (nspan,3), dir (nspan,3), i_of, j_of.
    """
    points = np.asarray(pan["points"])
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    n_c = nwrap // 2
    # station nodes: the wrap mesh's leading edge is i = n_c, trailing edge i = 0
    node_le = 0.5 * (points[n_c, :-1, :] + points[n_c, 1:, :])
    node_te = 0.5 * (points[0, :-1, :] + points[0, 1:, :])
    vec = node_te - node_le
    chord = np.linalg.norm(vec, axis=1)
    direction = vec / np.maximum(chord, 1e-30)[:, None]
    i_of = np.repeat(np.arange(nwrap), nspan)
    j_of = np.tile(np.arange(nspan), nwrap)
    rel = np.asarray(pan["centroids"]) - node_le[j_of]
    xi = np.einsum("kc,kc->k", rel, direction[j_of]) / np.maximum(chord[j_of], 1e-30)
    return {"xi": np.clip(xi, 0.0, 1.0), "chord": chord, "le": node_le,
            "dir": direction, "i_of": i_of, "j_of": j_of}


def wrap_spacing(pan, a, b):
    """The spacing the vendored wrap stencil uses between two panels. -> array

    max(centroid distance, mean of the two wrap extents), which is exactly what
    `_directional_derivative` divides by when it is handed `ds_wrap`. The
    prescribed doublet strength of the cavity has to be integrated with THIS
    spacing, not with the panel extent alone and not with the chordwise distance:
    the dynamic condition is a statement about the surface speed along the
    surface, and using the chord instead puts the leading-edge panels badly wrong
    because there the surface tangent is nearly normal to the chord.
    """
    cc = np.linalg.norm(np.asarray(pan["centroids"])[b]
                        - np.asarray(pan["centroids"])[a], axis=-1)
    ds = np.asarray(pan["ds_wrap"])
    return np.maximum(cc, 0.5 * (ds[a] + ds[b]))


def cavity_interior(pan, weight, threshold=0.5):
    """Cavity panels all four of whose neighbours are also cavity panels.

    The dynamic condition can only be satisfied here. At the detachment panel the
    wrap stencil reaches across the leading edge into wetted flow on the pressure
    side, and at the closure panel it reaches into the wetted flow aft, so the
    reconstructed surface speed at those two panels mixes a prescribed doublet
    strength with a solved one and is not the cavity speed. Measured: the interior
    satisfies the condition to about 3e-4 in the pressure coefficient while the
    two edge panels are out by 0.2 to 0.8. That is a property of a
    surface-gradient pressure evaluation, not a defect of the cavity, and it does
    not reach the load, which imposes the cavity pressure directly. -> (N,) bool
    """
    w = np.asarray(weight, dtype=float) > threshold
    out = w.copy()
    for key in ("wrap_prev", "wrap_next", "span_prev", "span_next"):
        nb = np.asarray(pan[key])
        out &= (nb >= 0) & w[np.maximum(nb, 0)]
    return out


def wrap_extent(pan, frame):
    """Chordwise extent of each panel in xi, for the fractional weight. -> (N,)

    Taken from ds_wrap projected onto the chord direction, which is the vendored
    per-panel wrap extent that `docs/fluid/ARCHITECTURE.md` lists as the cavity
    integration hook.
    """
    ds = np.asarray(pan["ds_wrap"])
    return ds / np.maximum(frame["chord"][frame["j_of"]], 1e-30)


# ------------------------------------------------------------------ 4 extent

def mask_from_lengths(pan, frame, lengths, alpha_rad, real_mask=None,
                      detach=None):
    """Ventilated weight of every panel from the per-station cavity length.

    A suction panel spanning [xi_a, xi_b] chordwise is covered by a cavity running
    from xi_detach to xi_detach + L to the fraction of its own extent that the
    overlap occupies, so the weight is continuous and piecewise linear in L, the
    load is continuous in the displacement, and the length resolves to a fraction
    of a panel instead of chattering between two of them.

    detach is the per-station detachment point, which is at or just aft of the
    leading edge: the first panel of a yawed section straddles the stagnation
    point and carries a POSITIVE pressure coefficient, so a cavity anchored at
    xi = 0 exactly would be anchored in flow that cannot hold it. None puts it at
    the leading edge.

    lengths is per spanwise station (nspan,), zero where the cavity has not
    reached that depth. -> (weight (N,), mask (N,) bool)
    """
    half = 0.5 * wrap_extent(pan, frame)
    lo, hi = frame["xi"] - half, frame["xi"] + half
    start = np.zeros(pan["nspan"]) if detach is None \
        else np.asarray(detach, dtype=float)
    a = start[frame["j_of"]]
    b = a + np.asarray(lengths, dtype=float)[frame["j_of"]]
    overlap = np.minimum(hi, b) - np.maximum(lo, a)
    weight = np.clip(overlap / np.maximum(hi - lo, 1e-30), 0.0, 1.0)
    live = suction_side(pan, alpha_rad)
    if real_mask is not None:
        live = live & np.asarray(real_mask, dtype=bool)
    weight = np.where(live, weight, 0.0)
    return weight, weight > 0.0


def station_lengths(pan, frame, weight):
    """Recover the per-station cavity length from the panel weights. -> (nspan,)

    The inverse of `mask_from_lengths` to within round-off on a monotone weight
    profile, which is what ties the two directions together.
    """
    ext = wrap_extent(pan, frame)
    covered = (weight * ext).reshape(pan["nwrap"], pan["nspan"])
    return covered.sum(axis=0)


def pressure_target(pan, frame, cp, cp_cavity, alpha_rad, real_mask=None):
    """Detachment point and cavity length from the pressure condition.

    The extent rule of an OPEN, naturally ventilated cavity: it occupies the run
    over which the flow would otherwise be below the cavity pressure, detaching
    where the pressure first falls below it and closing where the pressure has
    recovered back to it. Both crossings are interpolated inside the panel where
    they happen, so the target is continuous in the pressure and there is no
    staircase.

    Detachment is found rather than fixed at xi = 0 because the first suction
    panel of a yawed section straddles the stagnation point and carries a positive
    pressure coefficient; a scan that started there would abort at once and report
    no cavity at all, at every incidence. -> (detach (nspan,), length (nspan,))
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    live = suction_side(pan, alpha_rad)
    if real_mask is not None:
        live = live & np.asarray(real_mask, dtype=bool)
    live_g = live.reshape(nwrap, nspan)
    margin = (np.asarray(cp_cavity) - np.asarray(cp)).reshape(nwrap, nspan)
    xi = frame["xi"].reshape(nwrap, nspan)
    detach, length = np.zeros(nspan), np.zeros(nspan)
    for j in range(nspan):
        rows = np.where(live_g[:, j])[0]
        if rows.size == 0:
            continue
        rows = rows[np.argsort(xi[rows, j])]
        m, x = margin[rows, j], xi[rows, j]
        ready = m > 0.0
        if not np.any(ready):
            continue
        first = int(np.argmax(ready))
        # detachment: interpolate back to the crossing ahead of the first ready
        # panel, or take the leading edge when the section is ready from the nose
        if first == 0:
            detach[j] = 0.0
        else:
            m0, m1 = m[first - 1], m[first]
            detach[j] = x[first - 1] + (x[first] - x[first - 1]) \
                * (-m0) / max(m1 - m0, 1e-30)
        tail = ready[first:]
        if np.all(tail):
            length[j] = 1.0 - detach[j]
            continue
        end = first + int(np.argmax(~tail))
        m0, m1 = m[end - 1], m[end]
        close = x[end - 1] + (x[end] - x[end - 1]) * m0 / max(m0 - m1, 1e-30)
        length[j] = max(close - detach[j], 0.0)
    return detach, length


def section_target(cl_strip, sigma_c_strip, length_prev=None,
                   length_model="exact"):
    """Cavity length from the sectional model, (1.7) via the lift slope. -> (nspan,)

    The sectional effective incidence is inferred from the strip lift coefficient
    the panel solve produced, alpha_2D = Cl/a0, with a0 of (1.9) evaluated on the
    PREVIOUS length; one lagged evaluation, as the paper's own iteration does.
    Used as the initial guess for the other rules, and as the extent rule in its
    own right for the strips whose cavity has left the body, where the closure
    line the washout criterion needs lies in the wake and the panel mesh has
    nothing to say about it.
    """
    cl_strip = np.asarray(cl_strip, dtype=float)
    prev = np.zeros_like(cl_strip) if length_prev is None \
        else np.asarray(length_prev, dtype=float)
    alpha_2d = np.abs(cl_strip) / np.maximum(vs.lift_slope(prev), 1e-30)
    return np.asarray(vs.cavity_length(vs.psi(np.maximum(sigma_c_strip, 0.0),
                                              alpha_2d), length_model),
                      dtype=float)


def thickness(pan, frame, sigma, sigma_wetted, weight, v_s):
    """Cavity thickness by chordwise integration of the transpiration. -> (N,)

    On a cavity panel the unknown source strength is the cavity's transpiration,
    and linearised the thickness follows from
        t(s) = (1/V_s) integral_detach^s [sigma - sigma_wetted] ds',  t = 0 at
    detachment. Integrated along the wrap chain from the leading edge, weighted by
    the fractional cavity coverage so a partially covered closure panel
    contributes its fraction.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    ds = np.asarray(pan["ds_wrap"]).reshape(nwrap, nspan)
    excess = ((np.asarray(sigma) - np.asarray(sigma_wetted))
              * np.asarray(weight)).reshape(nwrap, nspan)
    xi = frame["xi"].reshape(nwrap, nspan)
    speed = np.maximum(np.abs(np.asarray(v_s)).reshape(nwrap, nspan), 1e-30)
    out = np.zeros((nwrap, nspan))
    for j in range(nspan):
        rows = np.argsort(xi[:, j])
        run = np.cumsum(excess[rows, j] * ds[rows, j])
        out[rows, j] = run / speed[rows, j]
    return out.ravel()


def closure_thickness(pan, frame, thick, weight):
    """Cavity thickness at the closure point, per depth station. -> (nspan,)

    Normalised by the local chord. THE closure condition: the cavity surface has
    to come back onto the body at its downstream end, so a positive value means
    the cavity has not closed yet and must lengthen, and a negative one means it
    has overshot. This is a KINEMATIC condition and it is the right one for a
    ventilated cavity as well as a vaporous one - what distinguishes the two is
    that air is supplied continuously, so a small net flux is admissible, not that
    the cavity need not close.

    It cannot be replaced by a pressure condition when the cavity pressure is
    imposed on the solve: the dynamic condition then makes the computed pressure
    equal the cavity pressure everywhere on the cavity, so the pressure margin is
    identically zero there and a length read off it collapses to nothing and
    limit-cycles at one panel width. The pressure rule is self-consistent only for
    the 'clip' closure, whose solve stays wetted.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    w = np.asarray(weight, dtype=float).reshape(nwrap, nspan)
    t = np.asarray(thick, dtype=float).reshape(nwrap, nspan)
    xi = frame["xi"].reshape(nwrap, nspan)
    out = np.zeros(nspan)
    for j in range(nspan):
        live = np.where(w[:, j] > 0.0)[0]
        if live.size == 0:
            continue
        last = live[np.argmax(xi[live, j])]
        out[j] = t[last, j] / max(frame["chord"][j], 1e-30)
    return out


def entrainment(pan, sigma, sigma_wetted, weight):
    """Volume flux of gas into the cavity per depth station. -> (nspan,)

    The net transpiration of the cavity panels, which for a cavity open to the
    atmosphere is the rate at which air is entrained from the free surface, in
    volume per unit time. A physically meaningful OUTPUT, and the reason the
    closure condition of a ventilated cavity is not that this vanishes.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    areas = np.asarray(pan["areas"]).reshape(nwrap, nspan)
    excess = ((np.asarray(sigma) - np.asarray(sigma_wetted))
              * np.asarray(weight)).reshape(nwrap, nspan)
    return (excess * areas).sum(axis=0)


def closure_residual(pan, sigma, sigma_wetted, weight):
    """Net cavity transpiration per station, normalised. -> (nspan,)

    Zero for a CLOSED cavity: the closure condition of a partial vaporous cavity
    is that its thickness returns to zero at the closure point, which after
    integration is exactly that the net source strength vanishes.

    A naturally ventilated cavity is NOT closed. It is an open system, fed
    continuously with air from the free surface, so its net transpiration is the
    entrainment rate and not a residual to be driven to zero; its extent is set
    instead by the pressure condition, that the cavity ends where the wetted
    pressure has recovered to the cavity pressure. That is why `extent_rule`
    defaults to 'pressure' for ventilation and 'thickness' belongs to a closed
    cavity, which is what a non-zero dsigma makes. Reported either way, because
    for a ventilated cavity it is a measurement rather than an error.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    ds = np.asarray(pan["ds_wrap"]).reshape(nwrap, nspan)
    excess = ((np.asarray(sigma) - np.asarray(sigma_wetted))
              * np.asarray(weight)).reshape(nwrap, nspan)
    scale = np.maximum((np.abs(np.asarray(sigma_wetted)).reshape(nwrap, nspan)
                        * ds).sum(axis=0), 1e-30)
    return (excess * ds).sum(axis=0) / scale


# --------------------------------------------------------------- 5 inception

def separated(pan, cp, alpha_rad, recovery=RECOVERY, real_mask=None):
    """Modelled leading-edge separation on the suction side. -> (N,) bool

    A panel method has no boundary layer, so this is an assumption and not a
    computation. Thin-aerofoil separation is taken to occupy the run aft of the
    suction peak over which the pressure has recovered by more than `recovery` of
    the peak-to-trailing-edge rise; the paper's oil-film visualisations show
    exactly such a bubble, growing with incidence and not strongly dependent on
    Reynolds number for a sharp-nosed section.

    This is the separation BUBBLE, which exists at sub-stall incidence too. It is
    a necessary condition for inception, not a sufficient one: the free-surface
    seal also has to be broken, which is the stall gate or an injection.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    live = suction_side(pan, alpha_rad)
    if real_mask is not None:
        live = live & np.asarray(real_mask, dtype=bool)
    cp_g = np.where(live, np.asarray(cp), np.inf).reshape(nwrap, nspan)
    out = np.zeros((nwrap, nspan), dtype=bool)
    for j in range(nspan):
        col = cp_g[:, j]
        if not np.any(np.isfinite(col)):
            continue
        peak = int(np.argmin(col))
        finite = np.where(np.isfinite(col))[0]
        aft = finite[finite > peak]
        if aft.size == 0:
            continue
        rise = col[aft[-1]] - col[peak]
        if rise <= 0.0:
            continue
        out[aft, j] = (col[aft] - col[peak]) > recovery * rise
    return out.ravel() & live


def air_path(pan, candidate, seed):
    """Panels reachable from an air source through the candidate set. -> (N,) bool

    A breadth-first fill over the vendored face adjacency (wrap_prev/next,
    span_prev/next). This is the whole difference between ventilation and
    cavitation: a sub-cavity-pressure pocket with no connected path to the
    atmosphere is a cavitation site, not a ventilated one, and is refused.
    """
    candidate = np.asarray(candidate, dtype=bool)
    reached = np.asarray(seed, dtype=bool) & candidate
    if not np.any(reached):
        return np.zeros_like(candidate)
    neighbours = np.stack([np.asarray(pan[k]) for k in
                           ("wrap_prev", "wrap_next", "span_prev", "span_next")],
                          axis=1)
    frontier = reached.copy()
    while np.any(frontier):
        nxt = neighbours[frontier].ravel()
        nxt = nxt[nxt >= 0]
        grow = np.zeros_like(reached)
        grow[nxt] = True
        grow &= candidate & ~reached
        reached |= grow
        frontier = grow
    return reached


def waterline_band(pan, vent):
    """The surface-piercing strip: where air can enter. -> (N,) bool

    The shallowest immersed strip on the suction side. Whether the air actually
    gets in is a separate question - the seal has to be broken, by stall or by an
    injection - and that is asked once, in `solve_cavity`, rather than per panel.
    An injection can be confined to one station, which is the paper's air jet at
    the junction of the leading edge and the free surface.
    """
    nspan = pan["nspan"]
    j_of = np.tile(np.arange(nspan), pan["nwrap"])
    band = j_of == max(vent["waterline_j"] - 1, 0)
    station = vent.get("inject_station")
    if vent.get("inject_active", False) and station is not None:
        band = band & (j_of == station)
    return band & suction_side(pan, vent["alpha_rad"])


def inject(vent, station=None, active=True):
    """Turn the perturbation-induced formation route on or off. -> vent

    The paper's air jet at the junction of the leading edge and the free surface.
    It breaks the surface seal, so inception no longer waits for stall; a cavity
    formed this way persists after the injection stops if the flow is in the
    bi-stable region, which is what `test_vent` checks.
    """
    vent = dict(vent)
    vent["inject_active"] = bool(active)
    vent["inject_station"] = station
    return vent


# ----------------------------------------------------------------- 6 closure

def closure_geometry(pan, frame, lengths, vent):
    """The cavity closure line, its depth extent and its mean angle. -> dict

    The closure line is the chordwise station of cavity closure against depth,
    x_closure(z'), taken over the ventilated stations only. Its mean angle from
    the horizontal is `vent_section.closure_angle`, and the sign convention is
    pinned there: a cavity of uniform length gives 90 degrees and is the
    classical unstable two-dimensional case.

    Keys: depth, x_closure, phi_bar, phi_local, d_cav, stations.
    """
    lengths = np.asarray(lengths, dtype=float)
    nspan = pan["nspan"]
    j_of = np.tile(np.arange(nspan), pan["nwrap"])
    depth_strip = np.array([panel_depth(pan, vent["y_fs"])[j_of == j].mean()
                            for j in range(nspan)])
    live = (lengths > 0.0) & (depth_strip > 0.0)
    if live.sum() < 2:
        return {"depth": depth_strip[live], "x_closure": lengths[live],
                "phi_bar": np.pi / 2.0, "phi_local": np.zeros(int(live.sum())),
                "d_cav": float(depth_strip[live].max()) if live.any() else 0.0,
                "stations": live}
    order = np.argsort(depth_strip[live])
    depth = depth_strip[live][order]
    x_closure = lengths[live][order] * frame["chord"][live][order]
    phi_bar, phi_local = vs.closure_angle(depth, x_closure)
    return {"depth": depth, "x_closure": x_closure, "phi_bar": phi_bar,
            "phi_local": phi_local, "d_cav": float(depth.max()),
            "stations": live}


def regime_of(cav, vent):
    """The flow regime of a converged cavity. -> 'FW' | 'PV' | 'FV'"""
    return vs.regime(cav["d_cav"], cav["phi_bar"], vent["h"], tol=TOL_REGIME)


def transition(vent, cav):
    """The hysteretic regime update. Called ONLY from commit_vent. -> (regime, why)

    Formation needs an air path to ventilation-ready flow AND the free-surface
    seal broken. Elimination needs only that one of the sustaining conditions has
    failed. The asymmetry between the two IS the hysteresis: a cavity survives to
    incidences well below the one that formed it, because persistence never asks
    the seal question again. That is the paper's own mechanism, and it is why the
    regions of its map overlap.

    Inception is decided on the inception CRITERION and not on a trial cavity,
    because a fully wetted state does not compute one: while the regime is FW the
    cavity length is held at zero, so a rule that waited for a cavity to appear
    would never fire - the cavity cannot grow until the regime changes and the
    regime cannot change until the cavity grows. The criterion is that a connected
    ventilation-ready region covers more than INCEPT_FRACTION of the immersed
    suction side, which keeps one marginal panel from triggering it.
    """
    was = vent["regime"]
    ready = bool(cav.get("ready_fraction", 0.0) > INCEPT_FRACTION)
    now = vs.regime(cav["d_cav"], cav["phi_bar"], vent["h"], tol=TOL_REGIME)
    if was == "FW":
        if ready and cav.get("seal_broken", False):
            return "PV", "inception"
        return "FW", ""
    if not ready:
        return "FW", "rewetting"
    if was == "PV":
        if now == "FV":
            return "FV", "stabilisation"
        return "PV", ""
    # was FV: an unstable closure line or a cavity off the tip takes it back to PV
    if now != "FV":
        return "PV", "washout"
    return "FV", ""


# --------------------------------------------------------------- 7 the state

def build_vent(maps, points_half, h, chord, u_ref, alpha_rad, fn_h=None,
               dsigma=0.0, g=vs.G, gamma_st=vs.GAMMA_ST, rho=vs.RHO_WATER,
               alpha_stall=vs.ALPHA_STALL, closure="dirichlet",
               extent_rule=None, length_model="exact", recovery=RECOVERY,
               sub_tol=TOL_LENGTH, sub_max=MAX_CAVITY, sub_omega=OMEGA_CAVITY,
               x_ref=None, regime="FW", growth_chords=GROWTH_CHORDS):
    """The fifth state. Built once; only `commit_vent` changes it. -> vent

    closure='dirichlet' prescribes mu on the cavity from the dynamic condition
    and solves for the source strength, so the cavity has thickness and a
    displacement effect on the flow. closure='clip' overrides the pressure at
    fixed mu; it is the auditable first stage and a fallback, and it is NOT a
    self-consistent flow model - it leaves the trailing-edge jump, and hence the
    wake circulation and the induced field, at their wetted values.

    x_ref is fixed once here, deliberately. `integrate_loads` locates its own
    moment reference by a max-distance search on the mid-span section, which on a
    doubled mesh is the waterline and which moves with every deflection, so the
    yawing moment would acquire a spurious drift.
    """
    if closure not in ("dirichlet", "clip"):
        raise ValueError(f"unknown closure model '{closure}'; choose from "
                         f"'dirichlet', 'clip'")
    if extent_rule is None:
        # 'pressure' for BOTH closures: a naturally ventilated cavity is open to
        # the atmosphere, so its extent is set by where the wetted pressure has
        # recovered to the cavity pressure, not by a zero-net-transpiration
        # closure condition, which belongs to a closed vaporous cavity.
        extent_rule = "pressure"
    if extent_rule not in ("thickness", "pressure", "section", "frozen"):
        raise ValueError(f"unknown extent rule '{extent_rule}'; choose from "
                         f"'thickness', 'pressure', 'section', 'frozen'")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime '{regime}'; choose from "
                         f"{', '.join(REGIMES)}")
    points_half = np.asarray(points_half, dtype=float)
    nspan_full = maps["shape_full"][1] - 1
    fn_h = vs.froude_h(u_ref, h, g) if fn_h is None else float(fn_h)
    if x_ref is None:
        # MID-CHORD of the immersed half, which is the reference the paper's (3.3c)
        # uses for the yawing moment, so CM is directly comparable with its figure
        # 14. Fixed once, and deliberately: `integrate_loads` locates its own
        # reference by a max-distance search on the mid-span section, which on a
        # doubled mesh is the waterline and which moves with every deflection, so
        # the yawing moment would acquire a spurious drift.
        n_c = (points_half.shape[0] - 1) // 2
        le = points_half[n_c].mean(axis=0)
        te = points_half[0].mean(axis=0)
        x_ref = 0.5 * (le + te)
    vent = dict(maps)
    vent.update({
        "h": float(h), "chord": float(chord), "ar": float(h / chord),
        "u_ref": float(u_ref), "fn_h": fn_h, "dsigma": float(dsigma),
        "g": float(g), "gamma_st": float(gamma_st), "rho": float(rho),
        "alpha_rad": float(alpha_rad), "alpha_stall": float(alpha_stall),
        "closure": closure, "extent_rule": extent_rule,
        "length_model": length_model, "recovery": float(recovery),
        "sub_tol": float(sub_tol), "sub_max": int(sub_max),
        "sub_omega": float(sub_omega),
        "growth_chords": float(growth_chords),
        "x_ref": np.asarray(x_ref, dtype=float),
        "points_half_ref": points_half.copy(),
        "s_ref_wet": float(h * chord),
        "weber": vs.weber(u_ref, chord, rho, gamma_st),
        "inject_active": False, "inject_station": None,
        # the time level of the ventilation; advances only on commit
        "regime": regime, "l_c": np.zeros(nspan_full),
        "detach": np.zeros(nspan_full),
        "weight": np.zeros((maps["shape_full"][0] - 1) * nspan_full),
        "d_cav": 0.0, "phi_bar": np.pi / 2.0, "step": 0, "time": 0.0,
        "transitions": [], "drift_y": 0.0,
    })
    return vent


def commit_vent(vent, cav, t=None, dt=None):
    """Advance the ventilation to the next time level. In place, like fes.commit.

    THE ONLY mutation of the committed part of the state, and the only place
    `transition` is called. Everything a coupling subiteration computes lives in
    `cav` and is discarded; what survives is the regime, the cavity extent that
    seeds the next step's iteration, and the diagnostics.

    With a time step, the committed cavity front is RATE-LIMITED to
    GROWTH_CHORDS chords of cavity per chord of travel. The cavity does not
    appear all at once: it grows by propagating separation ahead of itself, over
    a time the paper measures in tenths of a second, and the equilibrium extent
    the cavity iteration returns is where it is heading rather than where it is.
    Without the limit a transition is an instantaneous load step and the
    structural response it produces is a property of the time step rather than of
    the flow. Called with dt=None - as the steady driver does - the limit is off
    and the committed extent is the equilibrium one.
    """
    regime, why = transition(vent, cav)
    if why:
        vent["transitions"].append((float(t or 0.0), vent["regime"], regime, why))
    vent["regime"] = regime
    target = np.asarray(cav["l_c"], dtype=float)
    if dt is not None and vent["growth_chords"] > 0.0:
        cap = (vent["growth_chords"] * vent["u_ref"] * float(dt)
               / max(vent["chord"], 1e-30))
        target = np.clip(target, vent["l_c"] - cap, vent["l_c"] + cap)
    vent["l_c"] = target.copy()
    vent["detach"] = np.asarray(cav["detach"], dtype=float).copy()
    vent["weight"] = np.asarray(cav["weight"], dtype=float).copy()
    vent["d_cav"] = float(cav["d_cav"])
    vent["phi_bar"] = float(cav["phi_bar"])
    vent["step"] += 1
    vent["time"] = float(t) if t is not None else vent["time"] + float(dt or 0.0)
    del dt
    return vent


def set_regime(vent, regime):
    """Force a branch, for a bi-stability sweep. -> a new vent, not in place."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime '{regime}'; choose from "
                         f"{', '.join(REGIMES)}")
    out = dict(vent)
    out["regime"] = regime
    if regime == "FW":
        out["l_c"] = np.zeros_like(vent["l_c"])
        out["weight"] = np.zeros_like(vent["weight"])
        out["detach"] = np.zeros_like(vent["detach"])
    return out


def freeze(vent):
    """A copy whose cavity does not move, for the added-mass diagnostics. -> vent

    The added mass of a ventilated foil is the frozen-cavity added mass: a cavity
    boundary free to move is a different and much harder problem, so the
    distinction is made explicit rather than left to whoever reads the number.
    """
    out = dict(vent)
    out["extent_rule"] = "frozen"
    return out


# ---------------------------------------------------------------- 8 report

def cavity_report(pan, frame, vent, cav):
    """What to ask a misbehaving ventilated run. -> dict of measured scalars"""
    weight = np.asarray(cav["weight"])
    ext = wrap_extent(pan, frame)
    return {
        "regime": vent["regime"], "d_cav_over_h": cav["d_cav"] / vent["h"],
        "phi_bar_deg": np.degrees(cav["phi_bar"]),
        "unstable_closure": vs.unstable_closure(cav["phi_bar"]),
        "l_c_max": float(np.max(cav["l_c"])), "l_c_mean_wet":
            float(np.mean(cav["l_c"][cav["l_c"] > 0.0])) if np.any(cav["l_c"] > 0)
            else 0.0,
        "n_cavity": int(np.count_nonzero(weight)),
        "cavity_area": float((weight * np.asarray(pan["areas"])).sum()),
        "cavity_volume": float((np.asarray(cav.get("thickness", 0.0)) * weight
                                * np.asarray(pan["areas"])).sum()),
        "thickness_min": float(np.min(np.asarray(cav.get("thickness", [0.0])))),
        "closure_residual": float(np.abs(cav.get("closure_residual", 0.0)).max()),
        "iterations": int(cav.get("iterations", 0)),
        "resid": float(cav.get("resid", 0.0)),
        "branch_flip": bool(cav.get("branch_flip", False)),
        "fn_h": vent["fn_h"], "weber": vent["weber"], "ar_h": vent["ar"],
        "weber_ok": vent["weber"] > vs.WE_MIN,
        "coverage": float((weight * ext).sum() / max(ext.sum(), 1e-30)),
    }
