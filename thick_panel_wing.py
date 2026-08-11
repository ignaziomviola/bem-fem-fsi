"""Thick-wing source-doublet panel code with a free wake, steady and unsteady.

This module holds the physics blocks and the STEADY driver. The time-dependent
driver is unsteady_wing.py, which composes these same blocks: the only parts of
this file that exist for it are the wake time-stepping primitives (convect,
shed, truncate), the motion arguments of update_points (point_velocities,
rigid) and the dphi_dt argument of get_loads and pressure_fields. Nothing here
branches on whether the flow is steady.

Extends panel_wing.py to wings with thickness: constant-strength source and
doublet panels on the closed wing surface, internal Dirichlet boundary
condition (Morino), Morino Kutta condition, and the free-wake filament
relaxation machinery shared with panel_wing.py. The onset flow is uniform or
a sheared profile U(z) along x, as before.

Physics: docs/DESIGN.md. Architecture: docs/ARCHITECTURE.md. Usage: README.md.
Conventions that pin every sign (the unit checks in test_thick_panel_wing.py
are the authority; Katz & Plotkin's doublet has the OPPOSITE sign):

- mesh 'points' (nwrap+1, nspan+1, 3), i wrapping lower-surface TE -> LE ->
  upper-surface TE, j across the span; sharp closed TE; pinched tips;
- panel h = i*nspan + j with corners Q1 = points[i, j], Q2 = points[i+1, j],
  Q3 = points[i+1, j+1], Q4 = points[i, j+1], anticlockwise about the outward
  normal n = (Q3 - Q1) x (Q4 - Q2), normalised (no orientation flip anywhere);
- solid angle Omega(P) = integral of n.(P - Q)/r^3 dS, positive on the +n side;
  doublet potential phi = mu * Omega / (4 pi); mu = phi_outer - phi_inner;
- a constant-doublet panel equals a vortex ring of circulation mu along its
  perimeter traversed in REVERSED corner order Q1 -> Q4 -> Q3 -> Q2;
- sigma = -n . u_rel with u_rel the onset (later: relative) velocity at the
  panel centroid; the own-panel doublet coefficient is the interior limit -1/2.

The wake note of the architecture document applies: wake_segments mirrors the
closure invariant of `filament_segments` in the thin lifting-surface code (one
FAR_FIELD leg per filament along the normalised local onset at the last node;
no far-end spanwise return segment in the velocity view) rather than reusing
it, because the wake is stored as arrays rather than per-filament dicts.

Library use (no prompts, prints or figures outside main(); matplotlib is
imported lazily inside the plotting functions, so headless runs need nothing):
    import thick_panel_wing as tpw
    state = tpw.init_state(points, onset, u_ref)
    state = tpw.solve(state)
    loads = tpw.get_loads(state)

For a time history use the driver instead:
    import unsteady_wing as usw
    history, state = usw.time_march(points, onset, usw.heave(points, 0.1, 2.0),
                                    dt=0.05, nsteps=200)
"""

import numpy as np

from freewake_kernels import (segment_velocities, induced_velocity,
                              load_velocity_profile, make_onset, pitch_mesh,
                              initial_nodes, prompt)

# ------------------------------------------------------------- 1 constants

RHO = 1.0                     # default density; loads take rho as a parameter
NWAKE = 50                    # steady wake rows per strip
WAKE_LENGTH_SPANS = 2.0       # steady wake length in span extents
FAR_FIELD = 1.0e4             # length of the closing straight leg/quad
MAX_ITER = 30                 # wake relaxation iterations
OMEGA = 0.7                   # under-relaxation factor for node positions
TOL = 1.0e-3                  # convergence: max near-field displacement [chords]
CORE_WING = 1.0e-6            # vortex core radius for surface ring segments
RC_WAKE_FRACTION = 0.2        # wake core radius as fraction of spanwise spacing
WARP_WARN = 0.1               # panel warp warning threshold [sqrt(area)]


# --------------------------------------------------------------- 3 kernels

def _corner_frames(corners):
    """Mean-plane frames derived from corner arrays alone (image-wrapper ready).

    Returns origin (K,3), n_hat (K,3), t1 (K,3), t2 (K,3), xi/eta (K,4) local
    projected corner coordinates. Degenerate (pinched-tip triangle) panels get
    a valid frame from their nonzero diagonals.
    """
    c1, c2, c3, c4 = (corners[:, k, :] for k in range(4))
    n_raw = np.cross(c3 - c1, c4 - c2)
    n_norm = np.linalg.norm(n_raw, axis=1, keepdims=True)
    n_hat = n_raw / np.maximum(n_norm, 1e-30)
    origin = corners.mean(axis=1)
    e_w = 0.5 * (c2 + c3) - 0.5 * (c1 + c4)
    t1 = e_w - np.sum(e_w * n_hat, axis=1, keepdims=True) * n_hat
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
    # fallback for panels whose wrap edge vanishes: use the first diagonal
    bad = (t1_norm[:, 0] < 1e-14)
    if np.any(bad):
        alt = c3[bad] - c1[bad]
        alt -= np.sum(alt * n_hat[bad], axis=1, keepdims=True) * n_hat[bad]
        t1[bad] = alt
        t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
    t1 = t1 / np.maximum(t1_norm, 1e-30)
    t2 = np.cross(n_hat, t1)
    rel = corners - origin[:, None, :]
    xi = np.einsum("kvc,kc->kv", rel, t1)
    eta = np.einsum("kvc,kc->kv", rel, t2)
    return origin, n_hat, t1, t2, xi, eta


def _solid_angle_tri(pts, tri):
    """Signed van Oosterom-Strackee solid angle of triangles at points.

    pts (m,3), tri (K,3,3) -> (m,K). Negative on the side from which the
    vertex order appears anticlockwise (the +n side for our corner order).
    Degenerate triangles return zero.
    """
    a = tri[None, :, :, :] - pts[:, None, None, :]
    a1, a2, a3 = a[:, :, 0, :], a[:, :, 1, :], a[:, :, 2, :]
    n1 = np.linalg.norm(a1, axis=2)
    n2 = np.linalg.norm(a2, axis=2)
    n3 = np.linalg.norm(a3, axis=2)
    num = np.einsum("mkc,mkc->mk", a1, np.cross(a2, a3))
    den = (n1 * n2 * n3 + np.einsum("mkc,mkc->mk", a1, a2) * n3
           + np.einsum("mkc,mkc->mk", a2, a3) * n1
           + np.einsum("mkc,mkc->mk", a3, a1) * n2)
    return 2.0 * np.arctan2(num, den)


def solid_angle(pts, corners, chunk=300):
    """Panel solid angle Omega(P), positive on the +n side. (M,3),(K,4,3)->(M,K).

    Average of the two diagonal splits (Q1-Q3 and Q2-Q4). A single fixed split
    is not mirror-covariant - mirroring a panel swaps its corners onto the
    other diagonal, which on warped panels breaks the left-right symmetry of
    the influence matrices at O(warp). The average is exactly mirror-symmetric,
    deterministic and iteration-fixed (the design requirements), and remains a
    valid potential: it is the mean of two closed spanning surfaces, so the
    closed-body row-sum identity holds exactly and the perimeter-ring velocity
    equivalence is unchanged. In-plane evaluation returns the principal value.
    """
    tri = [corners[:, idx, :] for idx in
           ((0, 1, 2), (0, 2, 3), (1, 2, 3), (1, 3, 0))]
    out = np.empty((len(pts), len(corners)))
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        out[s:s + chunk] = -0.5 * sum(_solid_angle_tri(p, t) for t in tri)
    return out


def doublet_potential_matrix(pts, corners, far_diag=None, chunk=300):
    """Unit-strength constant-doublet potentials. (M,3),(K,4,3) -> (M,K).

    phi = Omega/(4 pi) with the sign convention above; NO -1/2 diagonal here
    (assembly adds it). far_diag switches to the point-dipole form beyond
    far_diag panel diagonals (used for per-iteration wake work only).
    """
    phi = solid_angle(pts, corners, chunk) / (4.0 * np.pi)
    if far_diag is not None:
        centre = corners.mean(axis=1)
        d1 = corners[:, 2, :] - corners[:, 0, :]
        d2 = corners[:, 3, :] - corners[:, 1, :]
        vec_a = 0.5 * np.cross(d1, d2)
        diag = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
        rho_v = pts[:, None, :] - centre[None, :, :]
        r2 = np.sum(rho_v * rho_v, axis=2)
        far = r2 > (far_diag * diag[None, :]) ** 2
        dip = np.einsum("mkc,kc->mk", rho_v, vec_a) / np.maximum(r2, 1e-30) ** 1.5
        phi = np.where(far, dip / (4.0 * np.pi), phi)
    return phi


def _source_edge_terms(pts, corners):
    """Shared geometry of the Hess source kernels.

    Returns (x, y, z) local field coordinates (m,K), edge logs L (m,K,4),
    signed in-plane edge distances p (m,K,4), edge unit data, frames.
    """
    origin, n_hat, t1, t2, xi, eta = _corner_frames(corners)
    rel = pts[:, None, :] - origin[None, :, :]
    x = np.einsum("mkc,kc->mk", rel, t1)
    y = np.einsum("mkc,kc->mk", rel, t2)
    z = np.einsum("mkc,kc->mk", rel, n_hat)
    nxt = (1, 2, 3, 0)
    xi2, eta2 = xi[:, nxt], eta[:, nxt]
    dxi, deta = xi2 - xi, eta2 - eta
    d = np.hypot(dxi, deta)                                   # (K,4)
    scale = np.maximum(d.max(axis=1, keepdims=True), 1e-30)
    live = d > 1e-12 * scale                                  # degenerate edges off
    r1 = np.sqrt((x[:, :, None] - xi[None, :, :]) ** 2
                 + (y[:, :, None] - eta[None, :, :]) ** 2
                 + z[:, :, None] ** 2)                        # (m,K,4)
    r2 = np.sqrt((x[:, :, None] - xi2[None, :, :]) ** 2
                 + (y[:, :, None] - eta2[None, :, :]) ** 2
                 + z[:, :, None] ** 2)
    num = r1 + r2 + d[None, :, :]
    den = np.maximum(r1 + r2 - d[None, :, :], 1e-10 * d[None, :, :] + 1e-30)
    big_l = np.where(live[None, :, :], np.log(np.maximum(num, 1e-30) / den), 0.0)
    d_safe = np.maximum(d, 1e-30)
    p = ((y[:, :, None] - eta[None, :, :]) * dxi[None, :, :]
         - (x[:, :, None] - xi[None, :, :]) * deta[None, :, :]) / d_safe[None, :, :]
    return x, y, z, big_l, p, dxi, deta, d_safe, live, n_hat, t1, t2, origin


def source_potential_matrix(pts, corners, chunk=300):
    """Unit-strength constant-source potentials (Hess). (M,3),(K,4,3) -> (M,K).

    Mean-plane projection; exact everywhere (the body block is assembled once).
    Far field -A/(4 pi r); finite on the panel itself (the own-panel entry).
    """
    proj = _projected_corners(corners)
    out = np.empty((len(pts), len(corners)))
    tri1 = proj[:, (0, 1, 2), :]
    tri2 = proj[:, (0, 2, 3), :]
    for s in range(0, len(pts), chunk):
        p_blk = pts[s:s + chunk]
        (x, y, z, big_l, p, dxi, deta, d_safe, live,
         n_hat, t1, t2, origin) = _source_edge_terms(p_blk, corners)
        omega = -(_solid_angle_tri(p_blk, tri1) + _solid_angle_tri(p_blk, tri2))
        out[s:s + chunk] = -(np.sum(p * big_l, axis=2) - z * omega) / (4.0 * np.pi)
    return out


def _projected_corners(corners):
    """Corners projected onto the mean plane (the source layer's surface)."""
    origin, n_hat, t1, t2, xi, eta = _corner_frames(corners)
    return (origin[:, None, :] + xi[:, :, None] * t1[:, None, :]
            + eta[:, :, None] * t2[:, None, :])


def source_velocity(pts, corners, sigma, chunk=300, far_diag=5.0):
    """Velocity of the source layer with strengths sigma. (M,3) out.

    Hess flat-panel velocity with the point-source form beyond far_diag panel
    diagonals. Edge-singular terms are clamped (wake nodes sit exactly on TE
    panel edges), so on-perimeter evaluations stay finite.
    """
    proj = _projected_corners(corners)
    tri1 = proj[:, (0, 1, 2), :]
    tri2 = proj[:, (0, 2, 3), :]
    centre = corners.mean(axis=1)
    d1 = corners[:, 2, :] - corners[:, 0, :]
    d2 = corners[:, 3, :] - corners[:, 1, :]
    area = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
    diag = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
    v = np.zeros((len(pts), 3))
    for s in range(0, len(pts), chunk):
        p_blk = pts[s:s + chunk]
        rho_v = p_blk[:, None, :] - centre[None, :, :]
        r2 = np.sum(rho_v * rho_v, axis=2)
        far = r2 > (far_diag * diag[None, :]) ** 2
        # far field: point sources of strength sigma*A
        w = sigma * area / (4.0 * np.pi)
        v_far = rho_v * (w[None, :] / np.maximum(r2, 1e-30) ** 1.5)[:, :, None]
        v_blk = np.where(far[:, :, None], v_far, 0.0).sum(axis=1)
        # near field: exact kernel on the not-far columns
        near_cols = np.where(~far.all(axis=0))[0]
        if len(near_cols):
            sub = corners[near_cols]
            (x, y, z, big_l, p, dxi, deta, d_safe, live,
             n_hat, t1, t2, origin) = _source_edge_terms(p_blk, sub)
            omega = -(_solid_angle_tri(p_blk, tri1[near_cols])
                      + _solid_angle_tri(p_blk, tri2[near_cols]))
            u1 = np.sum((deta / d_safe)[None, :, :] * big_l, axis=2)
            u2 = np.sum((-dxi / d_safe)[None, :, :] * big_l, axis=2)
            u_vec = (u1[:, :, None] * t1[None, :, :]
                     + u2[:, :, None] * t2[None, :, :]
                     + omega[:, :, None] * n_hat[None, :, :]) / (4.0 * np.pi)
            here = ~far[:, near_cols]
            w_near = np.where(here, sigma[near_cols][None, :], 0.0)
            v_blk += np.einsum("mk,mkc->mc", w_near, u_vec)
        v[s:s + chunk] = v_blk
    return v


# -------------------------------------------------------------- 4 geometry

def load_mesh(path):
    """Load and validate a 'thick-wrap-1' mesh. Returns points."""
    data = np.load(path)
    if "points" not in data:
        raise ValueError(f"{path} must contain 'points' with shape "
                         f"(nwrap+1, nspan+1, 3)")
    points = np.asarray(data["points"], dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"{path} must contain 'points' with shape "
                         f"(nwrap+1, nspan+1, 3); got {points.shape}")
    if "mesh_format" not in data:
        raise ValueError(
            f"{path} has no 'mesh_format' key: this looks like a camber-surface "
            f"mesh for panel_wing.py; thick_panel_wing.py needs a wrapped-surface "
            f"mesh with mesh_format='thick-wrap-1' (see make_thick_sample_inputs.py)")
    fmt = str(data["mesh_format"])
    if fmt != "thick-wrap-1":
        raise ValueError(f"unsupported mesh_format '{fmt}'; this code reads "
                         f"'thick-wrap-1'")
    return validate_mesh(points)


def validate_mesh(points, full=True):
    """Design validation rules 4-9; welds sub-tolerance TE gaps. Returns points."""
    nwrap, nspan = points.shape[0] - 1, points.shape[1] - 1
    if nwrap < 6 or nwrap % 2 or nspan < 2:
        raise ValueError(f"need nwrap >= 6 (even) and nspan >= 2; "
                         f"got nwrap={nwrap}, nspan={nspan}")
    d_box = np.linalg.norm(points.reshape(-1, 3).max(axis=0)
                           - points.reshape(-1, 3).min(axis=0))
    gap = np.linalg.norm(points[0] - points[-1], axis=1)
    if gap.max() > 1e-6 * d_box:
        j_bad = int(np.argmax(gap))
        raise ValueError(
            f"trailing edge not closed: gap {gap.max():.3e} at j={j_bad} "
            f"(points[0,j] must equal points[-1,j]; standard NACA four-digit "
            f"sections (last coefficient -0.1015) have an open TE of about "
            f"0.21% chord and need the closed-TE coefficient -0.1036)")
    if gap.max() > 0.0:
        te = 0.5 * (points[0] + points[-1])
        points = points.copy()
        points[0], points[-1] = te, te
        import warnings
        warnings.warn(f"welded trailing-edge gap of {gap.max():.3e}")
    for jt in (0, nspan):
        pinch = np.linalg.norm(points[:, jt] - points[::-1, jt], axis=1)
        if pinch.max() > 1e-9 * d_box:
            i_bad = int(np.argmax(pinch))
            raise ValueError(
                f"tip station j={jt} not collapsed: mismatch {pinch.max():.3e} "
                f"at i={i_bad}; tip sections must be pinched onto the camber "
                f"line (points[i,jt] == points[nwrap-i,jt])")
    if full:
        pan = build_panels(points)
        if np.any(pan["areas"] <= 1e-12 * d_box ** 2):
            h = int(np.argmin(pan["areas"]))
            raise ValueError(f"degenerate panel at h={h} (i={h // nspan}, "
                             f"j={h % nspan}): area {pan['areas'][h]:.3e}")
        vol = signed_volume(pan)
        if vol <= 0.0:
            raise ValueError(
                f"wrap direction reversed (signed volume {vol:.3e} <= 0): index "
                f"i must run lower-surface TE -> LE -> upper-surface TE with j "
                f"increasing towards +y; fix with points = points[::-1, :, :]")
        _check_bowtie(pan)
    return points


def _check_bowtie(pan):
    """Rule 9: both corner triangles must agree with the diagonal normal."""
    c = pan["corners"]
    n1 = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    n2 = np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0])
    dot1 = np.einsum("kc,kc->k", n1, pan["normals"])
    dot2 = np.einsum("kc,kc->k", n2, pan["normals"])
    a1 = np.linalg.norm(n1, axis=1)
    a2 = np.linalg.norm(n2, axis=1)
    live1 = a1 > 1e-12 * pan["areas"].max()
    live2 = a2 > 1e-12 * pan["areas"].max()
    bad = (live1 & (dot1 < 0.0)) | (live2 & (dot2 < 0.0))
    if np.any(bad):
        h = int(np.where(bad)[0][0])
        raise ValueError(f"crossed (bowtie) panel at h={h} (i={h // pan['nspan']}, "
                         f"j={h % pan['nspan']})")


def build_topology(nwrap, nspan):
    """Index-only half of the panels dict; pure function of the two integers."""
    n = nwrap * nspan
    i_of = np.repeat(np.arange(nwrap), nspan)
    j_of = np.tile(np.arange(nspan), nwrap)
    h = np.arange(n)
    wrap_prev = np.where(i_of > 0, h - nspan, -1)
    wrap_next = np.where(i_of < nwrap - 1, h + nspan, -1)
    span_prev = np.where(j_of > 0, h - 1, -1)
    span_next = np.where(j_of < nspan - 1, h + 1, -1)
    node = np.arange((nwrap + 1) * (nspan + 1)).reshape(nwrap + 1, nspan + 1)
    panel_nodes = np.stack([node[:-1, :-1].ravel(), node[1:, :-1].ravel(),
                            node[1:, 1:].ravel(), node[:-1, 1:].ravel()],
                           axis=1)
    weld_map = node.copy()
    weld_map[nwrap, :] = weld_map[0, :]                       # TE seam
    for jt in (0, nspan):                                     # tip pinch pairs
        for i in range(nwrap // 2 + 1, nwrap + 1):
            weld_map[i, jt] = weld_map[nwrap - i, jt]
    tip_flag = (j_of == 0) | (j_of == nspan - 1)
    return {
        "nwrap": nwrap, "nspan": nspan,
        "panel_nodes": panel_nodes, "weld_map": weld_map.ravel(),
        "k_up": (nwrap - 1) * nspan + np.arange(nspan),
        "k_low": np.arange(nspan),
        "wrap_prev": wrap_prev, "wrap_next": wrap_next,
        "span_prev": span_prev, "span_next": span_next,
        "tip_flag": tip_flag,
    }


def build_metric(points, topo):
    """Metric half of the panels dict; pure, no orientation flip anywhere."""
    nwrap, nspan = topo["nwrap"], topo["nspan"]
    flat = points.reshape(-1, 3)
    corners = flat[topo["panel_nodes"]]
    c1, c2, c3, c4 = (corners[:, k, :] for k in range(4))
    n_raw = np.cross(c3 - c1, c4 - c2)
    areas = 0.5 * np.linalg.norm(n_raw, axis=1)
    normals = n_raw / np.maximum(2.0 * areas[:, None], 1e-30)
    # area-weighted triangle centroids, averaged over BOTH diagonal splits so
    # the collocation points are mirror-symmetric on warped panels
    def _split_centroid(pa, pb, pc, pd):
        a1 = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
        a2 = 0.5 * np.linalg.norm(np.cross(pc - pa, pd - pa), axis=1)
        g1 = (pa + pb + pc) / 3.0
        g2 = (pa + pc + pd) / 3.0
        w = np.maximum(a1 + a2, 1e-30)
        return (a1[:, None] * g1 + a2[:, None] * g2) / w[:, None]

    centroids = 0.5 * (_split_centroid(c1, c2, c3, c4)
                       + _split_centroid(c2, c3, c4, c1))
    origin, n_hat, t1, t2, xi, eta = _corner_frames(corners)
    frames = np.stack([t1, t2, normals], axis=1)
    rel = corners - centroids[:, None, :]
    warp = np.abs(np.einsum("kvc,kc->kv", rel, normals)).max(axis=1)
    warp = warp / np.maximum(np.sqrt(areas), 1e-30)
    ds_wrap = np.linalg.norm(0.5 * (c2 + c3) - 0.5 * (c1 + c4), axis=1)
    pan = dict(topo)
    pan.update({
        "points": points, "corners": corners, "centroids": centroids,
        "normals": normals, "areas": areas, "frames": frames,
        "ds_wrap": ds_wrap, "warp": warp,
        "d_box": float(np.linalg.norm(flat.max(axis=0) - flat.min(axis=0))),
    })
    pan["dual"] = _gradient_dual(pan)
    return pan


def centroid_weights(corners):
    """Barycentric weights of the collocation point on its corners. (K,4,3)->(K,4)

    build_metric places the collocation point at the area-weighted mean of the
    triangle centroids of BOTH diagonal splits; the same operation is linear in
    the corner data, and these are its weights. Applying them to nodal
    VELOCITIES therefore samples the body motion at the collocation point with
    the same operator that placed it, which is exact for any field linear over
    the panel and so for a rigid-body velocity, rotation included. Kept
    separate from build_metric rather than folded into it so that the verified
    steady centroid path is untouched, bit for bit; the unit test asserts that
    the weights reproduce panels['centroids'] to round-off.
    """
    c = [corners[:, k, :] for k in range(4)]

    def _one(order):
        pa, pb, pc, pd = (c[k] for k in order)
        a1 = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
        a2 = 0.5 * np.linalg.norm(np.cross(pc - pa, pd - pa), axis=1)
        den = 3.0 * np.maximum(a1 + a2, 1e-30)
        w = np.zeros((len(corners), 4))
        w[:, order[0]] = (a1 + a2) / den
        w[:, order[1]] = a1 / den
        w[:, order[2]] = (a1 + a2) / den
        w[:, order[3]] = a2 / den
        return w

    return 0.5 * (_one((0, 1, 2, 3)) + _one((1, 2, 3, 0)))


def sample_at_centroids(pan, node_field):
    """Nodal field (nwrap+1, nspan+1, 3) sampled at the collocation points."""
    flat = np.asarray(node_field, dtype=float).reshape(-1, 3)
    per_corner = flat[pan["panel_nodes"]]                      # (N, 4, 3)
    return np.einsum("kv,kvc->kc", centroid_weights(pan["corners"]), per_corner)


def _chain_direction(centroids, prev_idx, next_idx):
    """Effective differencing directions of the _directional_derivative stencils.

    For a linear field f = g.x the stencil returns exactly g.e with e the
    weighted vector below (NOT the normalised chord between the outer
    neighbours - the weights are swapped on curved, non-uniform chains):
    central e = (hm*u+ + hp*u-)/(hp+hm); one-sided 3-point
    e = ((2h1+h2)*u1 - h1*u2)/(h1+h2); two-point fallback e = u1.
    """
    c = centroids
    e = np.zeros_like(c)
    both = (prev_idx >= 0) & (next_idx >= 0)
    if np.any(both):
        h = np.where(both)[0]
        dp = c[next_idx[h]] - c[h]
        dm = c[h] - c[prev_idx[h]]
        hp = np.linalg.norm(dp, axis=1, keepdims=True)
        hm = np.linalg.norm(dm, axis=1, keepdims=True)
        e[h] = (hm * dp / np.maximum(hp, 1e-30)
                + hp * dm / np.maximum(hm, 1e-30)) / np.maximum(hp + hm, 1e-30)
    for sel, idx, sign in (((prev_idx < 0) & (next_idx >= 0), next_idx, 1.0),
                           ((next_idx < 0) & (prev_idx >= 0), prev_idx, -1.0)):
        if not np.any(sel):
            continue
        h = np.where(sel)[0]
        n1 = idx[h]
        n2 = idx[n1]
        ok = n2 >= 0
        if np.any(ok):
            hh, o1, o2 = h[ok], n1[ok], n2[ok]
            d1v = c[o1] - c[hh]
            d2v = c[o2] - c[o1]
            h1 = np.linalg.norm(d1v, axis=1, keepdims=True)
            h2 = np.linalg.norm(d2v, axis=1, keepdims=True)
            e[hh] = sign * ((2.0 * h1 + h2) * d1v / np.maximum(h1, 1e-30)
                            - h1 * d2v / np.maximum(h2, 1e-30)) \
                / np.maximum(h1 + h2, 1e-30)
        if np.any(~ok):
            hh, o1 = h[~ok], n1[~ok]
            d1v = c[o1] - c[hh]
            e[hh] = sign * d1v / np.maximum(
                np.linalg.norm(d1v, axis=1, keepdims=True), 1e-30)
    return e


def _gradient_dual(pan):
    """Dual basis (N,2,3): d_i . e_j = delta_ij in the tangent plane.

    The wrap direction is the panel's own unit tangent t_c (the arc-length
    derivative's direction; centroid chords degenerate across an
    under-resolved leading edge). The span direction is the effective
    stencil direction, tangent-projected, so g = D1*d1 + D2*d2 reproduces
    linear fields exactly up to the O(curvature times spacing) normal
    component the surface data cannot see.
    """
    nrm = pan["normals"]
    e1 = pan["frames"][:, 0, :].copy()
    e2 = _chain_direction(pan["centroids"], pan["span_prev"], pan["span_next"])
    e2 -= np.einsum("kc,kc->k", e2, nrm)[:, None] * nrm
    g11 = np.einsum("kc,kc->k", e1, e1)
    g12 = np.einsum("kc,kc->k", e1, e2)
    g22 = np.einsum("kc,kc->k", e2, e2)
    det = np.maximum(g11 * g22 - g12 ** 2, 1e-12)
    d1 = (g22[:, None] * e1 - g12[:, None] * e2) / det[:, None]
    d2 = (g11[:, None] * e2 - g12[:, None] * e1) / det[:, None]
    return np.stack([d1, d2], axis=1)


def build_panels(points):
    """Full panels dict; pure function of the node array, re-callable at will."""
    nwrap, nspan = points.shape[0] - 1, points.shape[1] - 1
    return build_metric(points, build_topology(nwrap, nspan))


def signed_volume(pan):
    """Enclosed volume by the divergence theorem; positive for outward normals."""
    return float(np.sum(pan["areas"] * np.einsum(
        "kc,kc->k", pan["centroids"], pan["normals"])) / 3.0)


def te_nodes(points):
    """Welded trailing-edge node row (nspan+1, 3)."""
    return 0.5 * (points[0] + points[-1])


def section_chords(points):
    """Straight TE-to-farthest-point section chords (nspan+1,)."""
    te = te_nodes(points)
    return np.linalg.norm(points - te[None, :, :], axis=2).max(axis=0)


def reference_area(points):
    """Projected planform area: sum of mid-strip chord times strip width."""
    chords = section_chords(points)
    y_sec = points[..., 1].mean(axis=0)
    c_strip = 0.5 * (chords[:-1] + chords[1:])
    return float(np.sum(c_strip * np.abs(np.diff(y_sec))))


# ------------------------------------------------------------------ 5 wake

def init_wake(te, onset, steps, n_bound, closure, core):
    """Straight filaments along the local onset from the TE nodes.

    te (nspan+1, 3); steps (nrows,) caller data (the steady driver passes
    full(NWAKE, WAKE_LENGTH_SPANS*span_extent/NWAKE)); handles empty steps.
    """
    cols = [initial_nodes(te[j], [], steps, onset) for j in range(len(te))]
    return {
        "nodes": np.stack(cols, axis=1),      # (nrows+1, nspan+1, 3)
        "mu": np.zeros((len(steps), len(te) - 1)),
        "n_bound": int(n_bound), "steps": np.asarray(steps, dtype=float),
        "closure": bool(closure), "core": float(core),
    }


def wake_quads(wake, onset):
    """Doublet-quad corners (nquads, 4, 3), strip-major rows first.

    Quad (r, j): W1 = nodes[r, j], W2 = nodes[r+1, j], W3 = nodes[r+1, j+1],
    W4 = nodes[r, j+1] (normal continuing the upper surface; fixed W1-W3
    diagonal = the design's deterministic split). One FAR_FIELD closure quad
    per strip appended iff wake['closure']. Order: strip j rows 0..nrows-1,
    then strip j closure; j-major.
    """
    nodes = wake["nodes"]
    nrows = nodes.shape[0] - 1
    nspan = nodes.shape[1] - 1
    quads = []
    if wake["closure"]:
        end = nodes[-1]                                        # (nspan+1, 3)
        v_end = onset(end)
        d_end = v_end / np.maximum(
            np.linalg.norm(v_end, axis=1, keepdims=True), 1e-12)
        far = end + FAR_FIELD * d_end
    for j in range(nspan):
        for r in range(nrows):
            quads.append([nodes[r, j], nodes[r + 1, j],
                          nodes[r + 1, j + 1], nodes[r, j + 1]])
        if wake["closure"]:
            quads.append([nodes[-1, j], far[j], far[j + 1], nodes[-1, j + 1]])
    if not quads:
        return np.zeros((0, 4, 3))       # zero-row wake: the unsteady start
    return np.array(quads)


def wake_segments(wake, onset):
    """Vortex segments of the wake for Biot-Savart: (p1, p2, strengths, cores).

    Trailing segments at span node j, rows r -> r+1 (downstream) carry
    G = mu[r, j-1] - mu[r, j] (out-of-range mu = 0). Spanwise inter-row
    segments at row r (+y) carry G = mu[r, j] - mu[r-1, j], identically zero
    in steady and skipped then; the row-0 (TE) spanwise segment is ALWAYS
    dropped (analytic Morino cancellation with the two body TE ring edges).
    closure=True appends one FAR_FIELD leg per filament along the normalised
    local onset at the last node (mirroring panel_wing.filament_segments; no
    far-end spanwise return segment); closure=False ends at the last spanwise
    row - the starting vortex, emitted normatively.
    """
    nodes = wake["nodes"]
    mu = wake["mu"]
    nrows, nspan = mu.shape
    mu_pad = np.zeros((nrows, nspan + 2))
    mu_pad[:, 1:-1] = mu
    p1, p2, g = [], [], []
    # trailing segments, all span nodes at once per row
    for r in range(nrows):
        p1.append(nodes[r])
        p2.append(nodes[r + 1])
        g.append(mu_pad[r, :-1] - mu_pad[r, 1:])
    # spanwise inter-row segments (unsteady only) and the starting vortex
    row_g = np.diff(mu, axis=0)                                # mu[r] - mu[r-1]
    if np.any(row_g != 0.0):
        for r in range(1, nrows):
            gg = row_g[r - 1]
            live = gg != 0.0
            if np.any(live):
                p1.append(nodes[r, :-1][live])
                p2.append(nodes[r, 1:][live])
                g.append(gg[live])
    if wake["closure"]:
        end = nodes[-1]
        v_end = onset(end)
        d_end = v_end / np.maximum(
            np.linalg.norm(v_end, axis=1, keepdims=True), 1e-12)
        p1.append(end)
        p2.append(end + FAR_FIELD * d_end)
        g.append(mu_pad[-1, :-1] - mu_pad[-1, 1:])
    elif nrows:
        p1.append(nodes[-1, :-1])
        p2.append(nodes[-1, 1:])
        g.append(-mu[-1])                                      # starting vortex
    p1 = np.concatenate(p1) if p1 else np.zeros((0, 3))
    p2 = np.concatenate(p2) if p2 else np.zeros((0, 3))
    g = np.concatenate(g) if g else np.zeros(0)
    live = g != 0.0
    return (p1[live], p2[live], g[live],
            np.full(int(live.sum()), wake["core"]))


def body_segments(pan, mu):
    """Surface ring segments: reversed corner order Q1->Q4->Q3->Q2, G = mu_h.

    The TE-lying edges of the upper (i = nwrap-1) and lower (i = 0) TE panel
    rows are omitted - the other two legs of the analytic Morino drop.
    """
    c = pan["corners"]
    nspan = pan["nspan"]
    n = len(c)
    i_of = np.arange(n) // nspan
    ring = ((0, 3), (3, 2), (2, 1), (1, 0))                    # reversed order
    p1, p2, owner = [], [], []
    for a, b in ring:
        keep = np.ones(n, dtype=bool)
        if (a, b) == (0, 3):
            keep &= i_of != 0                                  # lower TE edge
        if (a, b) == (2, 1):
            keep &= i_of != pan["nwrap"] - 1                   # upper TE edge
        p1.append(c[keep, a])
        p2.append(c[keep, b])
        owner.append(np.where(keep)[0])
    p1 = np.concatenate(p1)
    p2 = np.concatenate(p2)
    owner = np.concatenate(owner)
    return p1, p2, mu[owner], np.full(len(p1), CORE_WING)


def set_bound_strengths(wake, mu, k_up, k_low):
    """Slave rows [0, n_bound) of the wake to the current TE jump (in place).

    n_bound is clamped to the number of rows present, so a zero-row wake (the
    state the time driver starts in, before its first shed) is a no-op rather
    than a silent slice miss.
    """
    n_b = min(int(wake["n_bound"]), wake["mu"].shape[0])
    if n_b:
        wake["mu"][:n_b, :] = mu[k_up] - mu[k_low]


def relax_wake_step(wake, v_nodes, omega, x_near):
    """One under-relaxed streamline remarch over wake['steps'].

    v_nodes ((nrows+1)*(nspan+1), 3) is evaluated EXTERNALLY at
    wake['nodes'].reshape(-1, 3) - the block owns no kernel. Returns the
    maximum near-field (x < x_near) node displacement. Row 0 stays pinned.
    """
    nodes = wake["nodes"]
    nrows1, nspan1 = nodes.shape[0], nodes.shape[1]
    v = v_nodes.reshape(nrows1, nspan1, 3)
    vn = np.linalg.norm(v, axis=2, keepdims=True)
    v_hat = v / np.maximum(vn, 1e-12)
    new = nodes.copy()
    for r, length in enumerate(wake["steps"]):
        new[r + 1] = new[r] + length * v_hat[r]
    new = nodes + omega * (new - nodes)
    near = nodes[..., 0] < x_near
    delta = float(np.max(np.linalg.norm((new - nodes)[near], axis=-1),
                         initial=0.0))
    wake["nodes"] = new
    return delta


def convect(wake, v_nodes, dt):
    """Displace every wake node except row 0 by dt * v (in place).

    The time-marching sibling of relax_wake_step: same nodes array, same
    externally evaluated velocities ((nrows+1)*(nspan+1), 3 at
    wake['nodes'].reshape(-1, 3)), same pinned row 0, but an explicit Euler
    step in physical time instead of a streamline remarch over wake['steps'].
    Wake nodes are material points of the fluid, so no under-relaxation is
    applied: the transient IS the answer. Returns the maximum displacement.
    """
    nodes = wake["nodes"]
    v = np.asarray(v_nodes, dtype=float).reshape(nodes.shape)
    step = dt * v[1:]
    nodes[1:] += step
    wake["nodes"] = nodes
    return float(np.max(np.linalg.norm(step, axis=-1), initial=0.0))


def te_convection_velocity(state):
    """Inertial fluid velocity at the trailing-edge nodes. (nspan+1, 3)

    This is the velocity that convects the newly shed vorticity, and it must
    NOT be taken from wake_node_velocity at row 0. The trailing-edge nodes lie
    exactly on the perimeter rings and on the source-panel edges, where the
    Biot-Savart and Hess kernels are singular; what comes back there is the
    regularised value, which depends on the core radius and on the edge clamp,
    and on a wing at incidence it is a near-cancellation - measured at 0.08 U
    where the answer is U. The Kutta condition makes the velocity at a sharp
    trailing edge finite and equal on both sides, so the mean of the upper and
    lower surface velocities of the two adjacent panel rows is the well-posed
    measure. Body motion is added back because the surface velocity is
    relative to the moving surface while the wake lives in the inertial frame.
    Strip values are averaged onto the nodes; the tips take their single
    neighbour.
    """
    pan = state["panels"]
    v_surf = surface_velocity(pan, state["mu"], state["u_rel"])
    k_up, k_low = pan["k_up"], pan["k_low"]
    v_strip = 0.5 * (v_surf[k_up] + v_surf[k_low])             # (nspan, 3)
    u_body = state.get("u_body")
    if u_body is not None:
        v_strip = v_strip + 0.5 * (u_body[k_up] + u_body[k_low])
    nspan = pan["nspan"]
    v_node = np.empty((nspan + 1, 3))
    v_node[1:-1] = 0.5 * (v_strip[:-1] + v_strip[1:])
    v_node[0], v_node[-1] = v_strip[0], v_strip[-1]
    return v_node


def shed(wake, v_nodes, dt, v_shed=None):
    """Convect the wake by one step and prepend the newly shed row (in place).

    Shedding and convection are one operation because the wake nodes are
    material points: the fluid particle sitting at the trailing edge at t is at
    te + dt*v at t + dt, and it is that particle which becomes the downstream
    edge of the new strip. Row r of the old wake becomes row r+1 of the new
    one, displaced by dt*v; the new row 0 is a placeholder that update_points
    re-pins onto the moved trailing edge. mu gains a leading zero row, filled
    by set_bound_strengths after the solve (n_bound = 1), so the history rows
    keep the strengths they were shed with - Kelvin's theorem, discretely.

    No shortening fraction is applied to the new strip. The frequently quoted
    0.2-0.3 shortening of the latest wake panel is a device for schemes that
    place that panel by hand; here its downstream edge is a material point and
    there is nothing to choose. Row spacing alone does not move the load: on a
    straight uniform sheet C_L changes by 0.02% between dt*U = 1.6 c and 0.1 c
    (docs/UNSTEADY.md), so the steady limit cannot pin such a convention and
    the indicial response is what would have to.

    v_nodes is evaluated at the CURRENT nodes (the previous time level) - the
    scheme is explicit, as every free-wake code of this class is. v_shed
    (nspan+1, 3) overrides the velocity of the departing trailing-edge row and
    should be te_convection_velocity(state): row 0 of v_nodes sits on the
    singular panel edges and is not usable there.
    """
    nodes = wake["nodes"]
    v = np.asarray(v_nodes, dtype=float).reshape(nodes.shape).copy()
    if v_shed is not None:
        v[0] = np.asarray(v_shed, dtype=float)
    wake["nodes"] = np.concatenate([nodes[:1], nodes], axis=0)
    wake["mu"] = np.concatenate([np.zeros((1, wake["mu"].shape[1])), wake["mu"]],
                                axis=0)
    wake["steps"] = np.concatenate([[0.0], np.asarray(wake["steps"], dtype=float)])
    v_full = np.concatenate([v[:1], v], axis=0)
    convect(wake, v_full.reshape(-1, 3), dt)
    wake["steps"][0] = float(np.linalg.norm(
        wake["nodes"][1] - wake["nodes"][0], axis=1).mean())
    return wake


def truncate(wake, nmax, close=True):
    """Keep the newest nmax rows, closing the cut at infinity (in place).

    A bare slice would leave the last kept row's spanwise segment acting as a
    starting vortex of the full strip strength, metres behind the wing instead
    of wherever the real one has drifted to. Setting closure=True instead
    replaces the discarded tail by the FAR_FIELD closing quad of the same
    strength, which both wake views already build, so the sheet stays closed
    and only the true starting vortex - by then far downstream - is given up.
    Truncating is therefore valid once the starting vortex no longer matters,
    and invalid during the indicial transient it is the whole point of.
    """
    nrows = wake["mu"].shape[0]
    if nmax is None or nrows <= nmax:
        return wake
    wake["nodes"] = wake["nodes"][:nmax + 1].copy()
    wake["mu"] = wake["mu"][:nmax].copy()
    wake["steps"] = wake["steps"][:nmax].copy()
    if close:
        wake["closure"] = True
    return wake


def wake_node_velocity(state):
    """Total advection velocity at the wake nodes: onset + sources + rings.

    The only composition point of the three terms.
    """
    pan, wake = state["panels"], state["wake"]
    pts = wake["nodes"].reshape(-1, 3)
    v = state["onset"](pts)
    v += source_velocity(pts, pan["corners"], state["sigma"])
    bp1, bp2, bg, bc = body_segments(pan, state["mu"])
    wp1, wp2, wg, wc = wake_segments(wake, state["onset"])
    p1 = np.concatenate([bp1, wp1])
    p2 = np.concatenate([bp2, wp2])
    g = np.concatenate([bg, wg])
    core = np.concatenate([bc, wc])
    return v + induced_velocity(pts, p1, p2, g, core)


# -------------------------------------------------------------- 6 assembly

def source_strengths(normals, u_rel):
    """sigma = -n . u_rel per panel; never sees the onset closure."""
    return -np.einsum("kc,kc->k", normals, u_rel)


def assemble_body(pan, chunk=300):
    """Geometry-only operator blocks {'D', 'S'}; -1/2 hard-wired onto D here."""
    d_blk = doublet_potential_matrix(pan["centroids"], pan["corners"],
                                     chunk=chunk)
    np.fill_diagonal(d_blk, -0.5)                 # analytic interior limit
    s_blk = source_potential_matrix(pan["centroids"], pan["corners"],
                                    chunk=chunk)
    return {"D": d_blk, "S": s_blk}


def wake_influence(colloc, wake, onset):
    """(P_bound (M, nspan), phi_known (M,)) for the current wake geometry.

    P_bound[:, j]: potential per unit strip doublet, rows [0, n_bound) of
    strip j (closure quad included when flagged). phi_known: frozen-row
    potentials times the stored mu - identically zero in steady. The general
    row split is implemented now so the unsteady extension is a no-op here.
    """
    nrows = wake["nodes"].shape[0] - 1
    nspan = wake["mu"].shape[1]
    if nrows == 0:                       # zero-row wake: the unsteady start
        return np.zeros((len(colloc), nspan)), np.zeros(len(colloc))
    quads = wake_quads(wake, onset)
    per_strip = nrows + (1 if wake["closure"] else 0)
    phi_q = doublet_potential_matrix(colloc, quads)            # (M, nquads)
    n_b = wake["n_bound"]
    p_bound = np.zeros((len(colloc), nspan))
    phi_known = np.zeros(len(colloc))
    for j in range(nspan):
        base = j * per_strip
        cols = phi_q[:, base:base + per_strip]
        # bound rows (and the closure quad, which continues the bound strip
        # in steady; with n_bound < nrows the closure belongs to the frozen tail)
        if n_b == nrows:
            p_bound[:, j] = cols.sum(axis=1)
        else:
            p_bound[:, j] = cols[:, :n_b].sum(axis=1)
            mu_tail = wake["mu"][n_b:, j]
            phi_known += cols[:, n_b:nrows] @ mu_tail
            if wake["closure"]:
                phi_known += cols[:, -1] * wake["mu"][-1, j]
    return p_bound, phi_known


def fold_column(a_mat, rhs, b_col, targets, const):
    """General affine-constraint fold, in place.

    A dependent strength equals sum(coeff * unknown[k]) + const, with
    influence column b_col: a_mat[:, k] += coeff * b_col per (k, coeff);
    rhs -= const * b_col.
    """
    for k, coeff in targets:
        a_mat[:, k] += coeff * b_col
    if const != 0.0:
        rhs -= const * b_col


def assemble_system(d_blk, s_blk, p_bound, phi_known, k_up, k_low,
                    unknown_type, mu_fixed, sigma_fixed):
    """Masked Dirichlet system (a_mat, rhs); N_sys = N always.

    Column k: D[:, k] if unknown_type[k] == 0 (mu unknown) else S[:, k]
    (sigma unknown). Every known strength accumulates into rhs with the
    opposite sign: rhs = -S_unmasked @ sigma_fixed - D_masked @ mu_fixed
    - phi_known - P_bound @ (known part of the TE jumps). Wake strip columns
    fold into k_up (+) and k_low (-) when those panels' mu is unknown, else
    their known mu feeds the rhs. With unknown_type all zero and mu_fixed
    zero this reproduces A mu = -B sigma with the folded wake bit for bit.
    P_bound=None (non-lifting body) skips the fold.
    """
    n = d_blk.shape[0]
    unknown_type = np.asarray(unknown_type)
    mask = unknown_type == 1
    a_mat = d_blk.copy()
    if np.any(mask):
        a_mat[:, mask] = s_blk[:, mask]
        rhs = -(s_blk[:, ~mask] @ np.asarray(sigma_fixed)[~mask])
        rhs -= d_blk[:, mask] @ np.asarray(mu_fixed)[mask]
    else:                       # bit-exact pass-through (all-mu unknowns)
        rhs = -(s_blk @ np.asarray(sigma_fixed))
    if phi_known is not None:
        rhs = rhs - phi_known
    if p_bound is not None:
        for j in range(p_bound.shape[1]):
            for k, coeff in ((k_up[j], +1.0), (k_low[j], -1.0)):
                if unknown_type[k] == 0:
                    fold_column(a_mat, rhs, coeff * p_bound[:, j],
                                [(k, 1.0)], 0.0)
                else:
                    rhs -= coeff * p_bound[:, j] * mu_fixed[k]
    return a_mat, rhs


# ------------------------------------------------------------ 7 solver API

def init_state(points, onset, u_ref=1.0, config=None):
    """Build a solver state. No prompts, no I/O.

    config keys (all optional): lifting (default True), nwake, wake_length_spans,
    n_bound, closure, wake (pass None to defer wake creation to the driver).
    """
    config = dict(config or {})
    pan = build_panels(points)
    chords = section_chords(points)
    u_body = np.zeros((len(pan["areas"]), 3))
    if config.get("point_velocities") is not None:
        u_body = sample_at_centroids(pan, config["point_velocities"])
    state = {
        "u_ref": float(u_ref), "onset": onset,
        "lifting": bool(config.get("lifting", True)),
        "panels": pan,
        "u_body": u_body,
        "u_rel": onset(pan["centroids"]) - u_body,
        "chord_mean": float(chords.mean()),
        "span_extent": float(np.ptp(points[..., 1])),
        "s_ref": reference_area(points),
        "x_near": float(points[..., 0].max() + np.ptp(points[..., 1])),
        "blocks": None, "blocks_version": -1,
        "mu": None, "sigma": None, "wake": None,
        "geom_version": 0, "wake_version": 0,
        "geom_invariants": _rigid_invariants(pan),
        "residuals": {"wake_disp_chords": np.inf, "iterations": 0},
    }
    if state["lifting"] and "wake" not in config:
        nwake = int(config.get("nwake", NWAKE))
        length = float(config.get("wake_length_spans", WAKE_LENGTH_SPANS))
        steps = np.full(nwake, length * state["span_extent"] / max(nwake, 1))
        rc = RC_WAKE_FRACTION * state["span_extent"] / pan["nspan"]
        state["wake"] = init_wake(te_nodes(points), onset, steps,
                                  n_bound=int(config.get("n_bound", nwake)),
                                  closure=bool(config.get("closure", True)),
                                  core=rc)
    elif "wake" in config:
        state["wake"] = config["wake"]
    return state


def _rigid_invariants(pan):
    """Scalars a rigid motion cannot change: enclosed volume and wetted area."""
    return (signed_volume(pan), float(pan["areas"].sum()))


def update_points(state, points, point_velocities=None, validate="fast",
                  rigid=False):
    """The single geometry chokepoint: same wrap topology, absolute coordinates.

    Rebuilds the metric (topology untouched), recomputes u_rel/s_ref/x_near,
    drops the operator blocks, re-pins wake row 0 to the moved TE; mu, sigma
    and the wake persist as warm starts. validate='fast': O(N) repairs and
    hard checks; 'full' reruns the load_mesh rules.

    point_velocities (nwrap+1, nspan+1, 3) is the velocity of the mesh nodes.
    It is sampled at the collocation points with the operator that placed them
    (centroid_weights) and subtracted from the onset, so u_rel is the velocity
    of the fluid RELATIVE to the moving surface and sigma = -n.u_rel enforces
    tangency on it. None means a stationary surface and leaves u_body zero.
    The wake is unaffected: it lives in the inertial frame, and the source and
    ring velocities that advect it are already absolute.

    rigid=True asserts that the motion since the last update is a rotation and
    a translation, under which the doublet and source blocks are invariant
    (verified to 1.3e-13 in the unit test), so they and their version stamp are
    kept and the O(N^2) reassembly - 0.63 s at 960 panels, the dominant cost of
    a time step - is skipped. The claim is checked against the enclosed volume
    and the wetted area, which no rigid motion changes.
    """
    pan = state["panels"]
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    if points.shape != (nwrap + 1, nspan + 1, 3):
        raise ValueError(f"update_points needs shape {(nwrap + 1, nspan + 1, 3)} "
                         f"(same wrap topology); got {points.shape}")
    points = np.asarray(points, dtype=float).copy()
    te = 0.5 * (points[0] + points[-1])                        # re-weld
    points[0], points[-1] = te, te
    for jt in (0, nspan):                                      # re-pin tips
        avg = 0.5 * (points[:, jt] + points[::-1, jt])
        points[:, jt] = avg
    if validate == "full":
        points = validate_mesh(points)
    topo = {k: pan[k] for k in ("nwrap", "nspan", "panel_nodes", "weld_map",
                                "k_up", "k_low", "wrap_prev", "wrap_next",
                                "span_prev", "span_next", "tip_flag")}
    new_pan = build_metric(points, topo)
    if validate == "fast":
        d_box = new_pan["d_box"]
        if np.any(new_pan["areas"] <= 1e-12 * d_box ** 2):
            raise ValueError(f"degenerate panel after update "
                             f"(geom_version {state['geom_version'] + 1})")
        if signed_volume(new_pan) <= 0.0:
            raise ValueError(f"inverted geometry after update "
                             f"(geom_version {state['geom_version'] + 1})")
        _check_bowtie(new_pan)
    if rigid:
        vol, area = _rigid_invariants(new_pan)
        vol0, area0 = state["geom_invariants"]
        drift = max(abs(vol - vol0) / max(abs(vol0), 1e-30),
                    abs(area - area0) / max(area0, 1e-30))
        if drift > 1e-9:
            raise ValueError(
                f"update_points(rigid=True) but the geometry deformed: enclosed "
                f"volume and wetted area drifted by {drift:.2e} (tolerance "
                f"1e-9). Call with rigid=False, which reassembles the blocks.")
    state["panels"] = new_pan
    u_body = np.zeros((len(new_pan["areas"]), 3))
    if point_velocities is not None:
        if np.shape(point_velocities) != points.shape:
            raise ValueError(f"point_velocities must have shape {points.shape}; "
                             f"got {np.shape(point_velocities)}")
        u_body = sample_at_centroids(new_pan, point_velocities)
    state["u_body"] = u_body
    state["u_rel"] = state["onset"](new_pan["centroids"]) - u_body
    chords = section_chords(points)
    state["chord_mean"] = float(chords.mean())
    state["span_extent"] = float(np.ptp(points[..., 1]))
    state["s_ref"] = reference_area(points)
    state["x_near"] = float(points[..., 0].max() + state["span_extent"])
    state["geom_version"] += 1
    if rigid and state["blocks"] is not None:
        state["blocks_version"] = state["geom_version"]    # blocks still valid
    else:
        state["blocks"] = None
        state["geom_invariants"] = _rigid_invariants(new_pan)
    if state["wake"] is not None:
        state["wake"]["nodes"][0] = te_nodes(points)
    return state


def interface(state):
    """Mesh data a structural driver needs to build its transfer operators."""
    pan = state["panels"]
    return {"nodes": pan["points"].reshape(-1, 3).copy(),
            "panel_nodes": pan["panel_nodes"].copy(),
            "weld_map": pan["weld_map"].copy(),
            "tip_flag": pan["tip_flag"].copy()}


def solve_once(state):
    """The shared step kernel: sigma -> wake influence -> fold -> LU -> sync.

    Asserts the operator blocks match the geometry. Shared by the steady
    driver and the future time driver; the cavity driver reuses it for its
    wetted baseline and composes assemble_system directly thereafter.
    """
    assert state["blocks"] is not None and \
        state["blocks_version"] == state["geom_version"], \
        "operator blocks are stale; call solve() which reassembles"
    pan = state["panels"]
    sigma = source_strengths(pan["normals"], state["u_rel"])
    if state["wake"] is not None:
        p_bound, phi_known = wake_influence(pan["centroids"], state["wake"],
                                            state["onset"])
    else:
        p_bound, phi_known = None, None
    n = len(sigma)
    a_mat, rhs = assemble_system(state["blocks"]["D"], state["blocks"]["S"],
                                 p_bound, phi_known, pan["k_up"], pan["k_low"],
                                 np.zeros(n, dtype=np.int8), np.zeros(n), sigma)
    mu = np.linalg.solve(a_mat, rhs)
    state["mu"], state["sigma"] = mu, sigma
    if state["wake"] is not None:
        set_bound_strengths(state["wake"], mu, pan["k_up"], pan["k_low"])
    return state


def solve(state, wake="relax", max_wake_iter=MAX_ITER, warm_start=True,
          verbose=False, callback=None):
    """Steady driver. wake='relax' | 'frozen' | 'none'.

    'frozen': one solve on the current wake geometry (the inner-coupling fast
    path). Deterministic and idempotent at fixed state and geometry.
    Diagnostics only via verbose/callback - never unconditional prints.
    Note: a driver that composes image blocks into state['blocks'] must also
    own the relaxation loop - wake_node_velocity composes exactly three terms
    and would advect the wake WITHOUT image terms.
    """
    if state["blocks"] is None or state["blocks_version"] != state["geom_version"]:
        state["blocks"] = assemble_body(state["panels"])
        state["blocks_version"] = state["geom_version"]
    if wake == "none" or state["wake"] is None:
        return solve_once(state)
    if not warm_start:
        old = state["wake"]
        rc = old["core"]
        state["wake"] = init_wake(te_nodes(state["panels"]["points"]),
                                  state["onset"], old["steps"],
                                  old["n_bound"], old["closure"], rc)
    if wake == "frozen":
        return solve_once(state)
    for it in range(max_wake_iter):
        solve_once(state)
        v = wake_node_velocity(state)
        delta = relax_wake_step(state["wake"], v, OMEGA, state["x_near"])
        state["wake_version"] += 1
        state["residuals"].update(wake_disp_chords=delta / state["chord_mean"],
                                  iterations=it + 1)
        if verbose:
            print(f"  wake iteration {it + 1:2d}: max near-field node "
                  f"displacement {delta / state['chord_mean']:.2e} c")
        if callback:
            callback(it, delta)
        if delta < TOL * state["chord_mean"]:
            break
    return solve_once(state)


# ----------------------------------------------------------------- 8 loads

def _directional_derivative(f, centroids, prev_idx, next_idx, steps=None):
    """Second-order derivative of f along a neighbour chain (one-sided at -1).

    Spacings are straight-line centroid distances; when the per-panel extents
    `steps` are given, the spacing is max(chord, (steps[a] + steps[b])/2).
    The arc floor is essential along the wrap: at a thin, under-resolved
    leading edge the chain folds back on itself and centroid chords collapse
    while the surface distance stays O(panel size). The chord floor matters
    where collapsed panels (sphere poles, pinched tips) make the extents
    underestimate the true centroid separation.
    """
    n = len(f)
    d = np.zeros(n)
    c = centroids

    def _h(a, b):
        chord = np.linalg.norm(c[b] - c[a], axis=1)
        if steps is None:
            return chord
        return np.maximum(chord, 0.5 * (steps[a] + steps[b]))

    both = (prev_idx >= 0) & (next_idx >= 0)
    if np.any(both):
        h = np.where(both)[0]
        hm = _h(prev_idx[h], h)
        hp = _h(h, next_idx[h])
        d[h] = (hm ** 2 * f[next_idx[h]] + (hp ** 2 - hm ** 2) * f[h]
                - hp ** 2 * f[prev_idx[h]]) / (hp * hm * (hp + hm))
    for sel, idx, sign in (((prev_idx < 0) & (next_idx >= 0), next_idx, 1.0),
                           ((next_idx < 0) & (prev_idx >= 0), prev_idx, -1.0)):
        if not np.any(sel):
            continue
        h = np.where(sel)[0]
        n1 = idx[h]
        n2 = idx[n1]
        ok = n2 >= 0
        if np.any(ok):
            hh, o1, o2 = h[ok], n1[ok], n2[ok]
            h1 = _h(hh, o1)
            h2 = _h(o1, o2)
            d[hh] = sign * (-(2 * h1 + h2) / (h1 * (h1 + h2)) * f[hh]
                            + (h1 + h2) / (h1 * h2) * f[o1]
                            - h1 / (h2 * (h1 + h2)) * f[o2])
        if np.any(~ok):
            hh, o1 = h[~ok], n1[~ok]
            d[hh] = sign * (f[o1] - f[hh]) / _h(hh, o1)
    return d


def surface_gradient(pan, f):
    """Tangential surface gradient of a per-panel scalar (N,) -> (N, 3).

    Central stencils along wrap (continuous through the LE) and span,
    one-sided at the TE seam and the tips; dual-basis conversion; tangent
    projection.
    """
    d1 = _directional_derivative(f, pan["centroids"], pan["wrap_prev"],
                                 pan["wrap_next"], steps=pan["ds_wrap"])
    d2 = _directional_derivative(f, pan["centroids"],
                                 pan["span_prev"], pan["span_next"])
    g = d1[:, None] * pan["dual"][:, 0, :] + d2[:, None] * pan["dual"][:, 1, :]
    g -= np.einsum("kc,kc->k", g, pan["normals"])[:, None] * pan["normals"]
    return g


def surface_velocity(pan, mu, u_rel):
    """External surface velocity V = (I - nn^T) u_rel + grad_s(mu)."""
    n = pan["normals"]
    u_t = u_rel - np.einsum("kc,kc->k", u_rel, n)[:, None] * n
    return u_t + surface_gradient(pan, mu)


def pressure_fields(pan, mu, u_rel, u_ref, dphi_dt=None, rho=1.0, p_ref=None):
    """Surface velocity, pressure coefficient and gauge pressure.

    p_gauge = 0.5*rho*(|u_rel|^2 - |V|^2) - rho*dphi_dt + p_ref(z_c);
    the streamline-local Bernoulli convention (|u_rel| = U(z_c) in steady).
    dphi_dt=None means zeros - the DRIVER owns mu_prev and the finite
    difference. p_ref: callable z -> reference pressure (None = zero); cp is
    defined from the p_ref=None path and is rho-free.
    """
    v_surf = surface_velocity(pan, mu, u_rel)
    q_loc = np.einsum("kc,kc->k", u_rel, u_rel)
    v2 = np.einsum("kc,kc->k", v_surf, v_surf)
    base = q_loc - v2
    if dphi_dt is not None:
        base = base - 2.0 * np.asarray(dphi_dt)
    p_gauge = 0.5 * rho * base
    if p_ref is not None:
        p_gauge = p_gauge + p_ref(pan["centroids"][:, 2])
    cp = base / u_ref ** 2
    return {"v_surf": v_surf, "cp": cp, "p_gauge": p_gauge}


def integrate_loads(pan, p_gauge, rho, u_ref, s_ref, x_ref=None):
    """Pressure integration: per-panel forces, resultants, CL, CD_pressure, CM."""
    force_panels = -p_gauge[:, None] * pan["areas"][:, None] * pan["normals"]
    force = force_panels.sum(axis=0)
    points = pan["points"]
    if x_ref is None:
        jm = pan["nspan"] // 2
        te = te_nodes(points)[jm]
        sec = points[:, jm, :]
        le = sec[int(np.argmax(np.linalg.norm(sec - te, axis=1)))]
        x_ref = le + 0.25 * (te - le)
    moment = np.cross(pan["centroids"] - x_ref, force_panels).sum(axis=0)
    q_ref = 0.5 * rho * u_ref ** 2 * s_ref
    chords = section_chords(points)
    c_ref = float(chords.mean())
    return {"force_panels": force_panels, "force": force, "moment": moment,
            "CL": force[2] / q_ref, "CD_pressure": force[0] / q_ref,
            "CM": moment[1] / (q_ref * c_ref), "x_ref": np.asarray(x_ref)}


def get_loads(state, rho=1.0, p_ref=None, dphi_dt=None):
    """One-call load bundle on the flat panel index (for coupling and output).

    dphi_dt (N,) is the rate of change of the doublet strength following the
    material panel, dmu/dt, which the DRIVER owns: panel h denotes the same
    material panel at every time level, so the difference is well defined, and
    it is what carries the added mass. None means the steady path.
    """
    pan = state["panels"]
    fields = pressure_fields(pan, state["mu"], state["u_rel"], state["u_ref"],
                             dphi_dt=dphi_dt, rho=rho, p_ref=p_ref)
    res = integrate_loads(pan, fields["p_gauge"], rho, state["u_ref"],
                          state["s_ref"])
    return {"mu": state["mu"].copy(), "sigma": state["sigma"].copy(),
            "v_surf": fields["v_surf"], "cp": fields["cp"],
            "p_gauge": fields["p_gauge"], "dp": fields["p_gauge"],
            "force": res["force_panels"], "areas": pan["areas"].copy(),
            "normals": pan["normals"].copy(),
            "centroids": pan["centroids"].copy(),
            "corners": pan["corners"].copy(), "quality": pan["warp"].copy(),
            "tip_flag": pan["tip_flag"].copy(), "resultants": res}


def spanwise_loading(pan, p_gauge, u_ref, rho=1.0):
    """Per-strip section lift coefficient from pressure integration."""
    nspan = pan["nspan"]
    fz = (-p_gauge * pan["areas"] * pan["normals"][:, 2]).reshape(-1, nspan)
    fz_strip = fz.sum(axis=0)
    points = pan["points"]
    chords = section_chords(points)
    c_strip = 0.5 * (chords[:-1] + chords[1:])
    y_sec = points[..., 1].mean(axis=0)
    dy = np.abs(np.diff(y_sec))
    y_strip = 0.5 * (y_sec[:-1] + y_sec[1:])
    q = 0.5 * rho * u_ref ** 2
    return y_strip, fz_strip / (q * c_strip * np.maximum(dy, 1e-30))


def leakage(state, offset=0.05):
    """Dirichlet leakage diagnostic: (max, rms) normal velocity just outside.

    Evaluated at the four panel-edge midpoints displaced outward by
    offset*sqrt(area) along the normal (the perimeter itself carries the
    ring and source edge singularities).
    """
    pan = state["panels"]
    c = pan["corners"]
    mids = 0.5 * (c + np.roll(c, -1, axis=1))                  # (N,4,3)
    lift = (offset * np.sqrt(pan["areas"]))[:, None, None] * \
        pan["normals"][:, None, :]
    pts = (mids + lift).reshape(-1, 3)
    nrm = np.repeat(pan["normals"], 4, axis=0)
    v = state["onset"](pts)
    v += source_velocity(pts, pan["corners"], state["sigma"])
    bp1, bp2, bg, bc = body_segments(pan, state["mu"])
    if state["wake"] is not None:
        wp1, wp2, wg, wc = wake_segments(state["wake"], state["onset"])
        bp1 = np.concatenate([bp1, wp1])
        bp2 = np.concatenate([bp2, wp2])
        bg = np.concatenate([bg, wg])
        bc = np.concatenate([bc, wc])
    v += induced_velocity(pts, bp1, bp2, bg, bc)
    vn = np.einsum("kc,kc->k", v, nrm)
    speed = max(float(np.linalg.norm(state["u_rel"], axis=1).max()), 1e-30)
    return float(np.abs(vn).max()) / speed, \
        float(np.sqrt(np.mean(vn ** 2))) / speed


def trefftz_drag(wake, onset, rho, u_ref, s_ref, x_plane):
    """Trefftz-plane induced drag from the relaxed wake (steady diagnostic).

    Operates on (wake, onset) alone. Returns cdi_2d (regularised
    two-dimensional point vortices at the filament crossings), cdi_3d
    (induced_velocity over the wake segments including the closure legs),
    the trace, and the far-field Kutta-Joukowski lift diagnostic.
    """
    nodes = wake["nodes"]
    mu = wake["mu"]
    nrows, nspan = mu.shape
    crossings, rows = [], []
    for j in range(nspan + 1):
        x = nodes[:, j, 0]
        idx = np.where((x[:-1] <= x_plane) & (x[1:] > x_plane))[0]
        if len(idx) == 0:
            raise ValueError(f"wake does not reach the Trefftz plane "
                             f"x = {x_plane:.3f}")
        r = int(idx[0])
        t = (x_plane - x[r]) / max(x[r + 1] - x[r], 1e-30)
        crossings.append(nodes[r, j] + t * (nodes[r + 1, j] - nodes[r, j]))
        rows.append(min(r, nrows - 1))
    crossings = np.array(crossings)                            # (nspan+1, 3)
    mu_pad = np.zeros((nrows, nspan + 2))
    mu_pad[:, 1:-1] = mu
    g_fil = np.array([mu_pad[rows[j], j] - mu_pad[rows[j], j + 1]
                      for j in range(nspan + 1)])
    seg_t = crossings[1:] - crossings[:-1]
    seg_len = np.linalg.norm(seg_t[:, 1:], axis=1)             # (y,z) length
    mids = 0.5 * (crossings[:-1] + crossings[1:])
    t_hat = seg_t[:, 1:] / np.maximum(seg_len, 1e-30)[:, None]  # (ty, tz)
    n_s = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)        # x_hat x t_hat
    mu_strip = np.array([mu[rows[j], j] for j in range(nspan)])
    # 2d mode
    dy = mids[:, None, 1] - crossings[None, :, 1]
    dz = mids[:, None, 2] - crossings[None, :, 2]
    r2 = dy ** 2 + dz ** 2 + wake["core"] ** 2
    v2d_y = np.sum(g_fil[None, :] * (-dz) / (2.0 * np.pi * r2), axis=1)
    v2d_z = np.sum(g_fil[None, :] * dy / (2.0 * np.pi * r2), axis=1)
    vn_2d = v2d_y * n_s[:, 0] + v2d_z * n_s[:, 1]
    # 3d mode
    wp1, wp2, wg, wc = wake_segments(wake, onset)
    v3d = induced_velocity(mids, wp1, wp2, wg, wc)
    vn_3d = v3d[:, 1] * n_s[:, 0] + v3d[:, 2] * n_s[:, 1]
    q_ref = 0.5 * rho * u_ref ** 2 * s_ref
    di_2d = -0.5 * rho * np.sum(mu_strip * vn_2d * seg_len)
    di_3d = -0.5 * rho * np.sum(mu_strip * vn_3d * seg_len)
    u_mid = onset(mids)[:, 0]
    kj_lift = rho * np.sum(u_mid * mu_strip * (crossings[1:, 1]
                                               - crossings[:-1, 1]))
    return {"cdi_2d": di_2d / q_ref, "cdi_3d": di_3d / q_ref,
            "trace": crossings, "kj_lift": kj_lift,
            "cl_trefftz": kj_lift / q_ref}


# -------------------------------------------------------------- 9 plotting

def plot_results(state, loads=None, sections=(0.0, 0.5, 0.85), save=None):
    """Result figure: loading, wake views, Cp sections, Cp map. Returns Figure."""
    import matplotlib.pyplot as plt
    pan = state["panels"]
    points = pan["points"]
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    if loads is None:
        loads = get_loads(state)
    span_extent = state["span_extent"]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(gs[0, 0])
    y_sec, cl_span = spanwise_loading(pan, loads["p_gauge"], state["u_ref"])
    ax.plot(2.0 * y_sec / span_extent, cl_span, "o-")
    ax.set_xlabel(r"$2y/b$")
    ax.set_ylabel(r"$c_l(y)$")
    ax.set_title("Spanwise loading")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    if state["wake"] is not None:
        nodes = state["wake"]["nodes"]
        for j in range(nodes.shape[1]):
            tip = j in (0, nodes.shape[1] - 1)
            ax.plot(nodes[:, j, 0], nodes[:, j, 1], nodes[:, j, 2],
                    color="tab:red" if tip else "tab:blue",
                    lw=1.2 if tip else 0.6)
    step = max(nwrap // 12, 1)
    for i in range(0, nwrap + 1, step):
        ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2],
                color="k", lw=0.5)
    ax.set_title("Relaxed wake")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(gs[0, 2])
    if state["wake"] is not None:
        nodes = state["wake"]["nodes"]
        for j in range(nodes.shape[1]):
            ax.plot(nodes[:, j, 1], nodes[:, j, 2], lw=0.6, color="tab:blue")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title("Wake, rear view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    cp2d = loads["cp"].reshape(nwrap, nspan)
    y_strip = 0.5 * (points[..., 1].mean(axis=0)[:-1]
                     + points[..., 1].mean(axis=0)[1:])
    for eta in sections:
        j = int(np.argmin(np.abs(2.0 * y_strip / span_extent - eta)))
        xc = 0.5 * (pan["centroids"].reshape(nwrap, nspan, 3)[:, j, 0])
        xc = pan["centroids"].reshape(nwrap, nspan, 3)[:, j, 0]
        ax.plot(xc, -cp2d[:, j], ".-", ms=3, lw=0.8,
                label=f"$2y/b$ = {2.0 * y_strip[j] / span_extent:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$-C_p$")
    ax.set_title("Chordwise pressure")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    if state["wake"] is not None:
        nodes = state["wake"]["nodes"]
        for j in range(nodes.shape[1]):
            ax.plot(nodes[:, j, 0], nodes[:, j, 1], lw=0.6, color="tab:blue")
    outline = np.vstack([points[0, :, :2]])
    ax.plot(points[0, :, 0], points[0, :, 1], "k", lw=1.5)
    i_le = nwrap // 2
    ax.plot(points[i_le, :, 0], points[i_le, :, 1], "k", lw=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Top view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    jm = nspan // 2
    sec = points[:, jm]
    ax.plot(sec[:, 0], sec[:, 2], "k-", lw=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Root section")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    upper = cp2d[nwrap // 2:, :]
    im = ax.imshow(upper, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(r"Upper-surface $C_p$ map")
    ax.set_xlabel("span index")
    ax.set_ylabel("wrap index (LE at bottom)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    res = loads["resultants"]
    lines = [f"CL = {res['CL']:.4f}",
             f"CD_pressure = {res['CD_pressure']:.5f}",
             f"CM = {res['CM']:.4f}",
             f"S_ref = {state['s_ref']:.4f}",
             f"U_ref = {state['u_ref']}"]
    ax.text(0.05, 0.9, "\n".join(lines), va="top", family="monospace")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def plot_sphere_cp(state, loads=None, save=None):
    """Computed against analytic Cp along the sphere meridian. Returns Figure."""
    import matplotlib.pyplot as plt
    pan = state["panels"]
    if loads is None:
        loads = get_loads(state)
    c = pan["centroids"]
    centre = c.mean(axis=0)
    rel = c - centre
    theta = np.arccos(np.clip(rel[:, 0] /
                              np.maximum(np.linalg.norm(rel, axis=1), 1e-30),
                              -1.0, 1.0))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.degrees(theta), loads["cp"], ".", ms=3, label="panel method")
    tt = np.linspace(0.0, np.pi, 200)
    ax.plot(np.degrees(tt), 1.0 - 2.25 * np.sin(tt) ** 2, "-",
            label=r"$1 - \frac{9}{4}\sin^2\theta$")
    ax.set_xlabel(r"$\theta$ from the onset axis [deg]")
    ax.set_ylabel(r"$C_p$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if save:
        fig.savefig(save, dpi=150)
    return fig


# ------------------------------------------------------------------ 10 CLI

def main():
    mesh_path = prompt("Wing mesh (.npz, thick-wrap-1)", "thick_wing_mesh.npz")
    profile_path = prompt("Velocity profile CSV (z, U)", "velocity_profile.csv")
    u_ref = float(prompt("Reference speed U_ref", "1.0"))
    alpha_deg = float(prompt("Additional pitch angle [deg]", "0.0"))

    points = load_mesh(mesh_path)
    if abs(alpha_deg) > 0.0:
        points = pitch_mesh(points, np.radians(alpha_deg))
        points = validate_mesh(points)
    z_table, u_table = load_velocity_profile(profile_path)
    onset = make_onset(z_table, u_table)

    state = init_state(points, onset, u_ref)
    pan = state["panels"]
    print(f"\nMesh: {pan['nwrap']} wrap x {pan['nspan']} spanwise panels, "
          f"span extent {state['span_extent']:.3f}, "
          f"S_ref (projected planform) {state['s_ref']:.4f}")
    if pan["warp"].max() > WARP_WARN:
        print(f"warning: max panel warp {pan['warp'].max():.3f} sqrt(area) "
              f"(tip-strip pressures are noisy)")

    print("Relaxing the wake...")
    state = solve(state, verbose=True)
    loads = get_loads(state)
    res = loads["resultants"]

    lk_max, lk_rms = leakage(state)
    print(f"\nCL (pressure integration): {res['CL']:.4f}")
    print(f"CD (pressure integration, unreliable at low order): "
          f"{res['CD_pressure']:.5f}")
    print(f"CM about the root quarter chord: {res['CM']:.4f}")
    print(f"Dirichlet leakage (max, rms normal velocity / onset): "
          f"{lk_max:.2e}, {lk_rms:.2e}")

    x_te = float(pan["points"][0, :, 0].max())
    b = state["span_extent"]
    sheared = np.ptp(u_table) > 1e-12
    cdis = []
    for frac in (0.8, 1.0):
        tz = trefftz_drag(state["wake"], onset, RHO, u_ref, state["s_ref"],
                          x_te + frac * b)
        cdis.append(tz["cdi_3d"])
        print(f"Trefftz plane at {frac:.1f} b: CDi = {tz['cdi_3d']:.5f} "
              f"(2d mode {tz['cdi_2d']:.5f}, KJ lift CL_T = "
              f"{tz['cl_trefftz']:.4f})")
    note = "; indicative only under shear" if sheared else ""
    print(f"CDi (Trefftz): {np.mean(cdis):.5f} +/- "
          f"{0.5 * abs(cdis[1] - cdis[0]):.5f}{note}")

    import matplotlib.pyplot as plt
    plot_results(state, loads)
    plt.show()


if __name__ == "__main__":
    main()
