"""Surface-piercing bodies: a free surface by the method of images.

A body that pierces a free surface is wetted over part of its surface only, and
a partially wetted wrap mesh is not a legal fluid mesh: thick_panel_wing's
validate_mesh demands a closed trailing edge, pinched tips and positive signed
volume, and update_points re-checks the volume on every call. The way out taken
here is to cut the body at the waterline, REFLECT the wetted part about the
free-surface plane, and hand the panel method the doubled body. That mesh is
legal by construction, so init_state, update_points, shed, wake_quads and every
kernel work unchanged, and the vendored fluid files stay byte-for-byte
untouched - this module is the "mirror-image free-surface wrapper" of row 13 of
their own extension table, written entirely on the driver side.

The two classical linearised free-surface conditions are then a single sign on
the image half, applied to the PERTURBATION potential at z = z_fs:

    condition   Froude limit   condition at z = z_fs      image strengths
    'rigid'     Fr -> 0        dphi/dz = 0 (double body)  eps = +1
    'open'      Fr -> infinity phi = 0                    eps = -1

'rigid' is what the unmodified solver already produces on the doubled mesh,
because the reflected body is then a real body satisfying a real tangency
condition. 'open' is not - tangency on the image half would give it +sigma, not
-sigma - so solve_free_surface assembles its own system: fold each image column
into its real partner with weight eps, impose sigma_image = eps * sigma, solve
the N x N real system, and scatter mu_full = (mu, eps * mu) back through
tpw.set_bound_strengths. The fold uses tpw.assemble_system unchanged, in its
documented bit-exact all-mu-unknown branch, so nothing here reimplements the
wake fold or the Kutta condition. With eps = +1 the fold is redundant, and that
it reproduces the plain solve to round-off is the sharpest gate in the suite.

Because the folded state carries GENUINE singularities at genuine positions,
wake_node_velocity, shed and get_loads stay correct with no image bookkeeping,
which is how this sidesteps the trap documented at thick_panel_wing.py:1133.
The one exception is that for eps = -1 the image system is a mathematical
construct whose velocity field is not the mirror of the real one, so the image
wake is re-imposed as the mirror after every shed and every relaxation pass;
for eps = +1 that re-imposition is a no-op to round-off, which is a gate.

The elevation comes from whichever linearised condition the image did NOT
enforce: the dynamic one for 'rigid', the kinematic one for 'open'. It is a
genuine output of the hydrodynamic solution and it is fed back to the body
through the pressure and through the waterline the next re-cut uses.

Axes. The vendored code fixes x streamwise, y spanwise, z vertical: make_onset
interpolates on z, p_ref is called with the collocation z, and CL is force[2].
Gravity is along -z, so a PIERCING wing has its span along z - root above the
plane in air, tip below it in water - and upright_mesh puts an ordinary wing
mesh into that attitude. Three vendored scalars are then computed on the wrong
axis and are re-pinned by _reference_scalars after every update_points: s_ref
(reference_area integrates over y and would collapse to zero, a division by
zero in CL), span_extent and the wake near-field cut x_near it feeds.

The functions are pure over explicit arguments, all mutable data lives in the
free-surface state dict, and prompts, prints and figures live only under
main().
"""

import numpy as np

import freewake_kernels as fk
import make_thick_sample_inputs as mts
import thick_panel_wing as tpw
import unsteady_wing as usw

G = 9.81                       # m/s^2, only ever a default
CONDITIONS = {"rigid": +1.0, "open": -1.0}
STATION_TOL = 1.0e-9           # a station must be a plane of constant z
UPSTREAM_FETCH = 6.0           # chords of fetch for the kinematic integration
FETCH_POINTS = 24              # samples along it


# ------------------------------------------------------------- 1 geometry

def mirror_points(points, z_fs):
    """Reflect a point array about the horizontal plane z = z_fs."""
    out = np.array(points, dtype=float)
    out[..., 2] = 2.0 * z_fs - out[..., 2]
    return out


def upright_mesh(points):
    """A quarter turn about x: span y -> z, section z -> -y.

    A proper rotation, so the trailing edge stays closed, both tips stay
    pinched and the signed volume keeps its sign. Station j = 0 of an ordinary
    wing mesh is at y = -span/2 and therefore becomes the LOWEST station, which
    is the immersed tip of the strut.
    """
    pts = np.asarray(points, dtype=float)
    out = np.empty_like(pts)
    out[..., 0] = pts[..., 0]
    out[..., 1] = -pts[..., 2]
    out[..., 2] = pts[..., 1]
    return out


def yaw_mesh(points, beta_rad):
    """Rotate about z, the incidence of an upright strut.

    tpw.pitch_mesh rotates about y, which is the incidence of a wing lying in
    its natural attitude; an upright strut yaws instead.
    """
    pts = np.asarray(points, dtype=float)
    c, s = np.cos(beta_rad), np.sin(beta_rad)
    out = np.empty_like(pts)
    out[..., 0] = c * pts[..., 0] + s * pts[..., 1]
    out[..., 1] = -s * pts[..., 0] + c * pts[..., 1]
    out[..., 2] = pts[..., 2]
    return out


def strut_mesh(n_c=8, nspan=12, draught=1.0, freeboard=0.5, chord=1.0,
               thickness=0.12, camber=0.0, taper=1.0, sweep_deg=0.0,
               twist_deg=0.0, pinch_root=False):
    """Upright surface-piercing strut, tip at z = -draught, root at +freeboard.

    The construction of mts.thick_wing_mesh in the attitude of upright_mesh:
    the section is laid in (x, -y), the span runs along z, and taper, sweep and
    twist are measured from the ROOT downwards, which is how a strut is drawn.
    The root station is left UNPINCHED by default: the fluid never sees this
    mesh - it sees the doubled wetted part - and a pinched root would give the
    structural mesh zero-thickness elements exactly where it is clamped and the
    stress is largest.
    """
    nwrap = 2 * int(n_c)
    span = float(draught) + float(freeboard)
    z_sta = np.linspace(-float(draught), float(freeboard), int(nspan) + 1)
    tan_sweep = np.tan(np.radians(sweep_deg))
    points = np.zeros((nwrap + 1, int(nspan) + 1, 3))
    for j, z in enumerate(z_sta):
        eta = (float(freeboard) - z) / span              # 0 root, 1 tip
        c = chord * (1.0 + (taper - 1.0) * eta)
        theta = np.radians(twist_deg) * eta
        pinched = (j == 0) or (j == int(nspan) and pinch_root)
        xs, ys = mts.thick_section(int(n_c), camber, thickness, pinched=pinched)
        x_sec, y_sec = (xs - 0.25) * c, ys * c
        x_rot = x_sec * np.cos(theta) + y_sec * np.sin(theta)
        y_rot = -x_sec * np.sin(theta) + y_sec * np.cos(theta)
        points[:, j, 0] = tan_sweep * (float(freeboard) - z) + 0.25 * c + x_rot
        points[:, j, 1] = -y_rot
        points[:, j, 2] = z
    points[-1, :, :] = points[0, :, :]                   # bit-exact TE closure
    for i in range(int(n_c) + 1):                        # bit-exact tip pinch
        points[nwrap - i, 0] = points[i, 0]
        if pinch_root:
            points[nwrap - i, int(nspan)] = points[i, int(nspan)]
    return points


def station_levels(points):
    """The z of each spanwise station, checking each is a plane of constant z.

    The waterline cut is exact only if a station is a level plane, which is
    what upright_mesh delivers for any sweep, taper or twist, since all three
    act within the station.
    """
    pts = np.asarray(points, dtype=float)
    z = pts[..., 2]
    spread = float(np.ptp(z, axis=0).max())
    scale = max(float(np.linalg.norm(np.ptp(pts.reshape(-1, 3), axis=0))),
                1e-300)
    if spread > STATION_TOL * scale:
        raise ValueError(
            f"spanwise stations are not level planes (largest spread within a "
            f"station {spread:.3e}); the waterline cut needs span along z, "
            f"which upright_mesh delivers")
    return z.mean(axis=0)


def _interpolate_stations(points, t):
    """The wrap mesh sampled at fractional station coordinates t."""
    pts = np.asarray(points, dtype=float)
    nspan = pts.shape[1] - 1
    t = np.clip(np.asarray(t, dtype=float), 0.0, float(nspan))
    j0 = np.clip(np.floor(t).astype(int), 0, nspan - 1)
    w = (t - j0)[None, :, None]
    return (1.0 - w) * pts[:, j0] + w * pts[:, j0 + 1]


def cut_at_waterline(points, z_level, nspan_wet=None):
    """The wetted part of a strut, from the immersed tip up to z_level.

    Returns a wrap mesh whose station 0 is the original pinched tip and whose
    last station lies exactly on the waterline. The stations in between are the
    original spanwise distribution rescaled onto the shortened span, so
    nspan_wet - and with it the panel count, the wake width and the transfer
    size - is held FIXED as the waterline moves. That is what lets the
    waterline be re-cut every step without re-gridding the wake.
    """
    pts = np.asarray(points, dtype=float)
    z_sec = station_levels(pts)
    if z_sec[0] > z_sec[-1]:
        raise ValueError("stations must run upwards; station 0 is the "
                         "immersed tip")
    z_level = float(z_level)
    if not z_sec[0] < z_level <= z_sec[-1]:
        raise ValueError(
            f"waterline z = {z_level:.6g} is outside the strut, whose stations "
            f"run {z_sec[0]:.6g} to {z_sec[-1]:.6g}; a piercing body needs the "
            f"tip below the plane and the root above it")
    t_wl = float(np.interp(z_level, z_sec, np.arange(len(z_sec))))
    n_wet = int(nspan_wet if nspan_wet is not None else max(int(np.ceil(t_wl)), 2))
    if n_wet < 1:
        raise ValueError(f"need nspan_wet >= 1; got {n_wet}")
    wet = _interpolate_stations(pts, np.linspace(0.0, t_wl, n_wet + 1))
    wet[:, 0] = pts[:, 0]                              # the tip stays exact
    wet[-1] = wet[0]                                   # bit-exact TE closure
    return wet


def double_mesh(points_wet, z_fs):
    """The wetted body plus its reflection: a closed, tip-pinched wrap mesh.

    Station m > J is the mirror of station 2J - m, which keeps the spanwise
    tangent reversed exactly where the reflection reverses handedness, so every
    image normal comes out as the mirror of its real partner - outward - and
    the signed volume stays positive. The waterline station appears once, and
    is its own mirror because it lies on the plane.
    """
    pw = np.asarray(points_wet, dtype=float)
    if pw.shape[1] < 2:
        raise ValueError("the wetted mesh needs at least one spanwise strip")
    img = mirror_points(pw[:, -2::-1], z_fs)
    return np.concatenate([pw, img], axis=1)


def image_index(nspan_d):
    """Panel h -> its image panel, for a doubled mesh of nspan_d strips."""
    m = np.arange(nspan_d)
    return nspan_d - 1 - m


def wetted_partition(nwrap, nspan_d):
    """(real, image, strips_real, strips_image) flat panel indices.

    The real half is the first nspan_d/2 strips of every wrap row, kept in
    increasing flat-index order so that a real panel's position within `real`
    is i * (nspan_d // 2) + m. That identity is what lets the folded system
    reuse tpw.assemble_system's k_up / k_low without a lookup table.
    """
    if nspan_d % 2:
        raise ValueError(f"a doubled mesh has an even nspan_d; got {nspan_d}")
    half = nspan_d // 2
    i = np.arange(nwrap)[:, None]
    m = np.arange(half)[None, :]
    real = (i * nspan_d + m).ravel()
    image = (i * nspan_d + (nspan_d - 1 - m)).ravel()
    return real, image, np.arange(half), nspan_d - 1 - np.arange(half)


# ---------------------------------------------------------------- 2 onset

def mirror_onset(onset, z_fs):
    """An onset closure made symmetric about the plane.

    The image is exact only if the incident flow is itself mirror-symmetric.
    make_onset returns U(z) x_hat, which has no vertical component already, so
    all that is needed is to evaluate it at the reflected height. It also keeps
    the wake closure legs - which are laid along the local onset direction at
    the last wake node - mirror-symmetric, which the image wake needs.
    """
    def _onset(pts):
        p = np.array(pts, dtype=float)
        p[:, 2] = z_fs - np.abs(p[:, 2] - z_fs)
        return onset(p)
    return _onset


# ----------------------------------------------------------------- 3 state

def init_free_surface(points_full, onset, z_fs=0.0, condition="rigid",
                      nspan_wet=None, g=G, u_ref=1.0, unsteady=True,
                      core=None, config=None):
    """Build the free-surface state from a full-span strut and a waterline.

    points_full is the whole body, air and water, in its reference
    configuration; it is kept so that every re-cut starts from the same
    undeformed geometry rather than from the previous cut.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}; "
                         f"got {condition!r}")
    pts_full = np.asarray(points_full, dtype=float)
    wet = cut_at_waterline(pts_full, z_fs, nspan_wet)
    doubled = double_mesh(wet, z_fs)
    onset_m = mirror_onset(onset, z_fs)
    if unsteady:
        fluid = usw.init_unsteady_state(doubled, onset_m, u_ref, core=core,
                                        config=config)
    else:
        fluid = tpw.init_state(doubled, onset_m, u_ref, config=config)
    nwrap, nspan_d = fluid["panels"]["nwrap"], fluid["panels"]["nspan"]
    real, image, s_real, s_image = wetted_partition(nwrap, nspan_d)
    fs = {"fluid": fluid,
          "condition": condition, "eps": CONDITIONS[condition],
          "z_fs": float(z_fs), "g": float(g),
          "onset": onset, "onset_mirrored": onset_m,
          "points_full": pts_full, "points_wet": wet,
          "nspan_wet": nspan_d // 2, "nwrap": nwrap, "nspan_d": nspan_d,
          "real": real, "image": image,
          "strips_real": s_real, "strips_image": s_image,
          "eta_waterline": np.zeros(nspan_d // 2 + 1),
          "s_ref": 0.0, "span_extent_z": 0.0, "chord_mean": 0.0,
          "draught": float(z_fs - station_levels(pts_full)[0])}
    _reference_scalars(fs)
    return fs


def _reference_scalars(fs):
    """Re-pin the three vendored scalars that assume a wing, not a strut.

    reference_area integrates the chord against |dy| and collapses to zero for
    an upright strut, which would divide by zero in CL; span_extent is the y
    extent for the same reason; and x_near, the wake near-field convergence
    cut, is built from span_extent. The wetted reference area used here is the
    REAL half only, so a coefficient means what it says for the immersed part.
    """
    fluid = fs["fluid"]
    pts = fluid["panels"]["points"]
    wet = pts[:, :fs["nspan_wet"] + 1]
    chords = tpw.section_chords(wet)
    z_sec = wet[..., 2].mean(axis=0)
    c_strip = 0.5 * (chords[:-1] + chords[1:])
    fs["s_ref"] = float(np.sum(c_strip * np.abs(np.diff(z_sec))))
    fs["span_extent_z"] = float(np.ptp(wet[..., 2]))
    fs["chord_mean"] = float(chords.mean())
    fluid["s_ref"] = fs["s_ref"]
    fluid["span_extent"] = fs["span_extent_z"]
    fluid["chord_mean"] = fs["chord_mean"]
    fluid["x_near"] = float(pts[..., 0].max()) + fs["span_extent_z"]
    return fs


def update_free_surface(fs, points_wet, point_velocities=None, validate="fast"):
    """The single geometry entry point: double, update, re-pin, mirror.

    point_velocities are given on the WETTED mesh; the image half takes their
    reflection, which is what makes sigma on the image half the mirror of the
    real one before the fold ever sees it.
    """
    wet = np.asarray(points_wet, dtype=float)
    doubled = double_mesh(wet, fs["z_fs"])
    vel = None
    if point_velocities is not None:
        v = np.asarray(point_velocities, dtype=float)
        v_img = v[:, -2::-1].copy()
        v_img[..., 2] *= -1.0
        vel = np.concatenate([v, v_img], axis=1)
    fs["points_wet"] = wet
    fs["fluid"] = tpw.update_points(fs["fluid"], doubled, vel, validate=validate)
    _reference_scalars(fs)
    return fs


# ----------------------------------------------------------------- 4 solve

def mirror_wake(fs):
    """Re-impose the image wake as the exact mirror of the real one.

    For eps = +1 this is a no-op to round-off, because the image half is a real
    body in a mirror-symmetric flow and convects there by itself. For eps = -1
    the image system is a mathematical construct: its velocity field is not the
    mirror of the real one, so the constraint has to be restated after every
    shed and every relaxation pass rather than left to the advection.
    """
    wake = fs["fluid"]["wake"]
    if wake is None or wake["nodes"].shape[0] == 0:
        return fs
    n_node = wake["nodes"].shape[1] - 1
    real = wake["nodes"][:, :n_node // 2 + 1]
    img = mirror_points(real[:, -2::-1], fs["z_fs"])
    wake["nodes"][:, n_node // 2 + 1:] = img
    half = wake["mu"].shape[1] // 2
    wake["mu"][:, half:] = fs["eps"] * wake["mu"][:, half - 1::-1]
    return fs


def _fresh_blocks(fluid):
    """Reassemble the geometry-only blocks if update_points invalidated them."""
    if fluid["blocks"] is None or \
            fluid["blocks_version"] != fluid["geom_version"]:
        fluid["blocks"] = tpw.assemble_body(fluid["panels"])
        fluid["blocks_version"] = fluid["geom_version"]
    return fluid


def solve_free_surface(fs, wake="frozen", max_wake_iter=tpw.MAX_ITER,
                       verbose=False):
    """One folded solve. wake = 'frozen' | 'relax' | 'none'.

    The image half of every column is folded into its real partner with weight
    eps and the source strengths on the image half are set to eps * sigma, so
    the N x N system is solved on the real panels alone and the free-surface
    condition is satisfied identically rather than to a tolerance. The solution
    is scattered back over the whole doubled state, which is what keeps the
    load, wake and advection blocks correct with no image bookkeeping.
    """
    fluid = _fresh_blocks(fs["fluid"])
    if wake == "none" or fluid["wake"] is None:
        _folded_solve(fs)
        return fs
    if wake == "frozen":
        _folded_solve(fs)
        return fs
    if wake != "relax":
        raise ValueError(f"wake must be 'relax', 'frozen' or 'none'; "
                         f"got {wake!r}")
    for it in range(max_wake_iter):
        _folded_solve(fs)
        v_nodes = tpw.wake_node_velocity(fluid).reshape(fluid["wake"]["nodes"].shape)
        delta = tpw.relax_wake_step(fluid["wake"], v_nodes, tpw.OMEGA,
                                    fluid["x_near"])
        mirror_wake(fs)
        fluid["wake_version"] += 1
        if verbose:
            print(f"  wake {it + 1:2d}: {delta / fluid['chord_mean']:.3e} chords")
        if delta < tpw.TOL * fluid["chord_mean"]:
            break
    fluid["residuals"] = {"wake_disp_chords": delta / fluid["chord_mean"],
                          "iterations": it + 1}
    _folded_solve(fs)
    return fs


def _folded_solve(fs):
    """sigma -> wake influence -> fold the image half -> LU -> scatter back.

    The shape of tpw.solve_once, with the image fold added and the system
    restricted to the real panels. tpw.assemble_system is used in its
    documented all-mu-unknown branch, so the Kutta fold and the wake bookkeeping
    are the vendored ones and not a second implementation of them.
    """
    fluid, eps = fs["fluid"], fs["eps"]
    pan, real, image = fluid["panels"], fs["real"], fs["image"]
    n_all = len(pan["centroids"])
    n = len(real)
    d_blk, s_blk = fluid["blocks"]["D"], fluid["blocks"]["S"]
    d_f = d_blk[np.ix_(real, real)] + eps * d_blk[np.ix_(real, image)]
    s_f = s_blk[np.ix_(real, real)] + eps * s_blk[np.ix_(real, image)]

    sigma_r = tpw.source_strengths(pan["normals"][real], fluid["u_rel"][real])
    if fluid["wake"] is not None:
        p_bound, phi_known = tpw.wake_influence(pan["centroids"][real],
                                                fluid["wake"], fluid["onset"])
        p_bound = p_bound[:, fs["strips_real"]] \
            + eps * p_bound[:, fs["strips_image"]]
    else:
        p_bound, phi_known = None, None

    half = fs["nspan_wet"]
    k_low = np.arange(half)
    k_up = (fs["nwrap"] - 1) * half + np.arange(half)
    a_mat, rhs = tpw.assemble_system(d_f, s_f, p_bound, phi_known, k_up, k_low,
                                     np.zeros(n, dtype=np.int8), np.zeros(n),
                                     sigma_r)
    mu_r = np.linalg.solve(a_mat, rhs)

    mu = np.empty(n_all)
    sigma = np.empty(n_all)
    mu[real], mu[image] = mu_r, eps * mu_r
    sigma[real], sigma[image] = sigma_r, eps * sigma_r
    fluid["mu"], fluid["sigma"] = mu, sigma
    if fluid["wake"] is not None:
        tpw.set_bound_strengths(fluid["wake"], mu, pan["k_up"], pan["k_low"])
    return fs


def shed_free_surface(fs, dt, nmax=None):
    """One shed of the doubled wake, then the image mirror re-imposed.

    Called ONCE per time step, outside any coupling subiteration, exactly as
    tpw.shed is in the immersed driver: it prepends a row and convects, which
    mutates the wake irreversibly.
    """
    fluid = fs["fluid"]
    if fluid["wake"] is None:
        return fs
    v_nodes = tpw.wake_node_velocity(fluid).reshape(fluid["wake"]["nodes"].shape)
    tpw.shed(fluid["wake"], v_nodes, dt,
             v_shed=tpw.te_convection_velocity(fluid))
    if nmax:
        tpw.truncate(fluid["wake"], nmax)
    mirror_wake(fs)
    fluid["wake_version"] += 1
    return fs


# ------------------------------------------------------------- 5 elevation

def flow_velocity(fs, pts):
    """Total velocity at arbitrary points: onset + sources + rings.

    The composition of tpw.wake_node_velocity, evaluated off the wake instead
    of on it. It is correct for the image half without any extra term because
    the image singularities are genuine entries of the state.
    """
    fluid = fs["fluid"]
    pan = fluid["panels"]
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    v = fluid["onset"](pts)
    v = v + tpw.source_velocity(pts, pan["corners"], fluid["sigma"])
    p1, p2, g, core = tpw.body_segments(pan, fluid["mu"])
    if fluid["wake"] is not None and fluid["wake"]["nodes"].shape[0] > 1:
        wp1, wp2, wg, wc = tpw.wake_segments(fluid["wake"], fluid["onset"])
        p1 = np.concatenate([p1, wp1])
        p2 = np.concatenate([p2, wp2])
        g = np.concatenate([g, wg])
        core = np.concatenate([core, wc])
    return v + fk.induced_velocity(pts, p1, p2, g, core)


def potential_at(fs, pts):
    """Perturbation potential at arbitrary points, from the current state.

    phi = (D mu + S sigma) with the vendored kernels, over the whole doubled
    body and the wake. Identically zero on the plane when eps = -1, which is
    the gate on that condition.
    """
    fluid = fs["fluid"]
    pan = fluid["panels"]
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    phi = tpw.doublet_potential_matrix(pts, pan["corners"]) @ fluid["mu"]
    phi = phi + tpw.source_potential_matrix(pts, pan["corners"]) @ fluid["sigma"]
    if fluid["wake"] is not None and fluid["wake"]["nodes"].shape[0] > 0:
        quads = tpw.wake_quads(fluid["wake"], fluid["onset"])
        if len(quads):
            per = fluid["wake"]["nodes"].shape[0] - 1 \
                + (1 if fluid["wake"]["closure"] else 0)
            strengths = _wake_quad_strengths(fluid["wake"], per)
            phi = phi + tpw.doublet_potential_matrix(pts, quads) @ strengths
    return phi


def _wake_quad_strengths(wake, per_strip):
    """The doublet strength of every wake quad, in wake_quads order."""
    nrows = wake["nodes"].shape[0] - 1
    nspan = wake["mu"].shape[1]
    out = np.empty(nspan * per_strip)
    for j in range(nspan):
        base = j * per_strip
        out[base:base + nrows] = wake["mu"][:, j]
        if per_strip > nrows:
            out[base + nrows] = wake["mu"][-1, j]
    return out


def wave_elevation(fs, x, y, phi_dot=None):
    """Free-surface elevation on the tensor grid (x, y). Returns (nx, ny).

    The image enforces one of the two linearised free-surface conditions
    exactly; the elevation is what the OTHER one then says.

    'rigid' (dphi/dz = 0 imposed): the dynamic condition gives
        g eta = 0.5 (|u_onset|^2 - |V|^2) - dphi/dt,
    the same Bernoulli convention tpw.pressure_fields uses on the body, so the
    elevation is the dynamic pressure on the plane divided by rho g.

    'open' (phi = 0 imposed): the dynamic condition is satisfied identically
    and the elevation comes from the kinematic one, U d(eta)/dx = w, integrated
    downstream from undisturbed water. x must be ascending and start upstream
    of the body for that integration to mean anything.

    phi_dot, when given, is the material rate of the perturbation potential on
    the grid; the driver owns the time levels, exactly as it owns dmu/dt.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    gx, gy = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(),
                           np.full(gx.size, fs["z_fs"])])
    v = flow_velocity(fs, pts)
    if fs["condition"] == "rigid":
        u_inf = fs["fluid"]["onset"](pts)
        base = np.einsum("kc,kc->k", u_inf, u_inf) \
            - np.einsum("kc,kc->k", v, v)
        eta = 0.5 * base / fs["g"]
        if phi_dot is not None:
            eta = eta - np.asarray(phi_dot, dtype=float).ravel() / fs["g"]
        return eta.reshape(gx.shape)
    w = v[:, 2].reshape(gx.shape)
    u_x = np.maximum(np.abs(fs["fluid"]["onset"](pts)[:, 0].reshape(gx.shape)),
                     1e-12)
    eta = np.zeros_like(w)
    if len(x) > 1:
        dx = np.diff(x)[:, None]
        mid = 0.5 * (w[:-1] / u_x[:-1] + w[1:] / u_x[1:])
        eta[1:] = np.cumsum(mid * dx, axis=0)
    return eta


def top_row(fs):
    """Flat indices of the wetted panel row adjacent to the waterline."""
    half, nspan_d = fs["nspan_wet"], fs["nspan_d"]
    return np.arange(fs["nwrap"]) * nspan_d + (half - 1)


def waterline_elevation(fs, dphi_dt=None, fetch=UPSTREAM_FETCH,
                        n_fetch=FETCH_POINTS):
    """Elevation along the body's own waterline: one value per wrap station.

    The waterline lies ON the hull, where an off-body kernel evaluation is
    singular, so the velocity is taken from the topmost wetted panel row's own
    surface velocity - the same quantity tpw.pressure_fields uses - rather than
    from a field point sitting on the surface.

    'rigid': the dynamic condition directly,
        g eta = 0.5 (|u_rel|^2 - |V|^2) - dphi/dt.
    'open': the kinematic condition U d(eta)/dx = w, integrated from the
    leading edge along each side of the section, started from the elevation of
    the undisturbed upstream fetch ahead of the nose.
    """
    fluid = fs["fluid"]
    pan = fluid["panels"]
    row = top_row(fs)
    v_surf = tpw.surface_velocity(pan, fluid["mu"], fluid["u_rel"])[row]
    u_rel = fluid["u_rel"][row]
    if fs["condition"] == "rigid":
        base = np.einsum("kc,kc->k", u_rel, u_rel) \
            - np.einsum("kc,kc->k", v_surf, v_surf)
        if dphi_dt is not None:
            base = base - 2.0 * np.asarray(dphi_dt, dtype=float)[row]
        return 0.5 * base / fs["g"]

    x = pan["centroids"][row, 0]
    w = v_surf[:, 2]
    u_x = np.maximum(np.abs(u_rel[:, 0]), 1e-12)
    n_le = fs["nwrap"] // 2                       # i = nwrap/2 is the LE
    nose = pan["centroids"][row[n_le]]
    s = np.linspace(-fetch * fs["chord_mean"], 0.0, n_fetch)
    probe = np.column_stack([nose[0] + s, np.full(n_fetch, nose[1]),
                             np.full(n_fetch, fs["z_fs"])])
    v_probe = flow_velocity(fs, probe)
    u_probe = np.maximum(np.abs(fluid["onset"](probe)[:, 0]), 1e-12)
    eta = np.empty(len(row))
    eta[n_le] = float(np.trapezoid(v_probe[:, 2] / u_probe, s))
    for side in (slice(n_le, None, -1), slice(n_le, None, 1)):
        idx = np.arange(len(row))[side]
        for a, b in zip(idx[:-1], idx[1:]):
            eta[b] = eta[a] + 0.5 * (w[a] / u_x[a] + w[b] / u_x[b]) \
                * (x[b] - x[a])
    return eta


def panel_elevation(fs, eta_wl):
    """The waterline elevation carried down onto the wetted panels.

    A panel is lifted by the elevation of the waterline directly above it,
    which is the linearised statement that the wave is long compared with the
    draught. It is the Froude-Krylov correction of linear seakeeping, and it is
    the path by which the shape of the free surface acts back on the body.
    """
    eta = np.asarray(eta_wl, dtype=float)
    half, nspan_d = fs["nspan_wet"], fs["nspan_d"]
    out = np.zeros(len(fs["fluid"]["panels"]["centroids"]))
    for i in range(fs["nwrap"]):
        out[i * nspan_d:i * nspan_d + half] = eta[i]
    out[fs["image"]] = out[fs["real"]]
    return out


# ----------------------------------------------------------------- 6 loads

def hydrostatic_p_ref(rho, g, z_fs, eta=0.0):
    """The vendored p_ref hook: rho g (z_fs + eta - z), zero above the surface.

    pressure_fields adds this to the gauge pressure at the collocation z and
    deliberately keeps it out of cp. It is hook 14 of the fluid architecture,
    unused until now.
    """
    def _p_ref(z):
        return rho * g * np.maximum(z_fs + eta - np.asarray(z, dtype=float), 0.0)
    return _p_ref


def wetted_loads(fs, rho, dphi_dt=None, gravity=True, eta=None):
    """Loads on the WETTED half, hydrostatics included, image half zeroed.

    The dry part of the body carries no fluid load at all - the aerodynamic
    forces are neglected, which is the whole premise - and the image half is a
    construct, so both are zeroed before the forces ever reach the transfer.
    """
    fluid = fs["fluid"]
    p_ref = None
    if gravity:
        eta_p = 0.0 if eta is None else panel_elevation(fs, eta)
        p_ref = hydrostatic_p_ref(rho, fs["g"], fs["z_fs"], eta_p)
    loads = tpw.get_loads(fluid, rho=rho, p_ref=p_ref, dphi_dt=dphi_dt)
    force = np.array(loads["force"])
    force[fs["image"]] = 0.0
    loads["force"] = force
    loads["force_wetted"] = force[fs["real"]]
    loads["resultants"] = tpw.integrate_loads(
        fluid["panels"], loads["p_gauge"] * _real_mask(fs), rho,
        fluid["u_ref"], fs["s_ref"])
    return loads


def _real_mask(fs):
    mask = np.zeros(len(fs["fluid"]["panels"]["centroids"]))
    mask[fs["real"]] = 1.0
    return mask


def buoyancy(fs, rho, eta=None):
    """Displaced weight of the wetted half, rho g V, from the panels alone.

    The wetted surface is open at the waterline, but the waterplane lid it is
    missing carries exactly zero hydrostatic pressure, so the open integral
    still closes the identity. That makes this an exact gate rather than an
    approximate one.
    """
    pan = fs["fluid"]["panels"]
    eta_p = 0.0 if eta is None else panel_elevation(fs, eta)
    p = hydrostatic_p_ref(rho, fs["g"], fs["z_fs"], eta_p)(
        pan["centroids"][:, 2])
    f = -p[:, None] * pan["areas"][:, None] * pan["normals"]
    return f[fs["real"]].sum(axis=0)


def wetted_volume(fs):
    """Volume below the waterline, by the divergence theorem on the real half."""
    pan = fs["fluid"]["panels"]
    z = pan["centroids"][:, 2] - fs["z_fs"]
    integrand = z * pan["areas"] * pan["normals"][:, 2]
    return float(integrand[fs["real"]].sum())


def wetted_interface(fs):
    """tpw.interface restricted to the wetted half, for build_transfer.

    The transfer must never see the image half: it is not a piece of structure
    and its loads are a construct. Node indices are renumbered onto the wetted
    mesh, and the weld map is renumbered with them.
    """
    fluid = fs["fluid"]
    nwrap, nspan_d, half = fs["nwrap"], fs["nspan_d"], fs["nspan_wet"]
    keep = (np.arange(nwrap + 1)[:, None] * (nspan_d + 1)
            + np.arange(half + 1)[None, :]).ravel()
    lookup = -np.ones((nwrap + 1) * (nspan_d + 1), dtype=np.int64)
    lookup[keep] = np.arange(len(keep))
    iface = tpw.interface(fluid)
    weld = lookup[iface["weld_map"][keep]]
    if np.any(weld < 0):                    # a welded partner outside the half
        weld = np.where(weld < 0, np.arange(len(keep)), weld)
    pn = fs["points_wet"].shape
    panel_nodes = (np.arange(nwrap)[:, None, None] * (half + 1)
                   + np.arange(half)[None, :, None]
                   + np.array([0, half + 1, half + 2, 1])[None, None, :]
                   ).reshape(-1, 4)
    tip = np.zeros(nwrap * half, dtype=bool)
    tip[np.arange(nwrap) * half] = True
    return {"nodes": fs["points_wet"].reshape(-1, 3).copy(),
            "panel_nodes": panel_nodes, "weld_map": weld, "tip_flag": tip,
            "shape": pn}


# ----------------------------------------------------------- 7 diagnostics

def froude(u_ref, length, g=G):
    """Froude number U / sqrt(g L); the two conditions are its two limits."""
    return float(u_ref) / np.sqrt(float(g) * float(length))


def plane_residual(fs, x=None, y=None, n=9):
    """How well the imposed free-surface condition is actually satisfied.

    'rigid' reports max |w| on the plane against the onset speed; 'open'
    reports max |phi| against u_ref times the mean chord. Both are round-off
    when the fold is right, because the image satisfies the condition
    identically rather than to a tolerance.
    """
    pts_x = np.linspace(-1.0, 2.0, n) * fs["chord_mean"] if x is None else x
    span = fs["span_extent_z"]
    pts_y = np.linspace(-span, span, n) if y is None else y
    gx, gy = np.meshgrid(pts_x, pts_y, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(),
                           np.full(gx.size, fs["z_fs"])])
    if fs["condition"] == "rigid":
        w = flow_velocity(fs, pts)[:, 2]
        return float(np.abs(w).max() / max(fs["fluid"]["u_ref"], 1e-300))
    phi = potential_at(fs, pts)
    return float(np.abs(phi).max()
                 / max(fs["fluid"]["u_ref"] * fs["chord_mean"], 1e-300))


def report(fs):
    """A dict of the numbers worth printing after a piercing solve."""
    fluid = fs["fluid"]
    return {"condition": fs["condition"], "eps": fs["eps"],
            "panels_total": len(fluid["panels"]["centroids"]),
            "panels_wetted": len(fs["real"]),
            "draught": fs["draught"], "s_ref": fs["s_ref"],
            "chord_mean": fs["chord_mean"],
            "froude_chord": froude(fluid["u_ref"], fs["chord_mean"], fs["g"]),
            "froude_draught": froude(fluid["u_ref"], max(fs["draught"], 1e-12),
                                     fs["g"]),
            "wetted_volume": wetted_volume(fs),
            "plane_residual": plane_residual(fs)}


# ------------------------------------------------------------------ 8 main

def main():
    """Solve one surface-piercing strut in each limit and report."""
    import matplotlib.pyplot as plt

    n_c, nspan = 8, 10
    points = strut_mesh(n_c=n_c, nspan=nspan, draught=1.0, freeboard=0.4,
                        chord=1.0, thickness=0.12)
    points = yaw_mesh(points, np.radians(5.0))
    onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, cond in zip(axes, ("rigid", "open")):
        fs = init_free_surface(points, onset, z_fs=0.0, condition=cond,
                               nspan_wet=6, u_ref=1.0, unsteady=False)
        fs = solve_free_surface(fs, wake="relax")
        rep = report(fs)
        print(f"\n{cond}: eps = {rep['eps']:+.0f}, "
              f"{rep['panels_wetted']} wetted of {rep['panels_total']} panels")
        print(f"  Fr(chord) {rep['froude_chord']:.3f}, "
              f"Fr(draught) {rep['froude_draught']:.3f}, "
              f"wetted volume {rep['wetted_volume']:.6f}")
        print(f"  free-surface condition satisfied to "
              f"{rep['plane_residual']:.3e}")
        loads = wetted_loads(fs, rho=1000.0, gravity=False)
        side = loads["force"][fs["real"], 1].sum()
        print(f"  side force {side:.4f} N, "
              f"C_side {side / (0.5 * 1000.0 * fs['s_ref']):.5f}")
        x = np.linspace(-2.0, 3.0, 61)
        y = np.linspace(-1.5, 1.5, 41)
        eta = wave_elevation(fs, x, y)
        cs = ax.contourf(x, y, eta.T, levels=21, cmap="RdBu_r")
        ax.plot(points[:, 0, 0], points[:, 0, 1], "k-", lw=1.0)
        ax.set_title(f"{cond}: elevation, max |eta| {np.abs(eta).max():.2e} m")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
