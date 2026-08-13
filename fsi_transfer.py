"""Load and motion transfer between the panel mesh and the structural mesh.

The two meshes are not required to match, and on a solid foil they nearly do:
the structural surface is the wetted surface everywhere except over the
outermost spanwise strip at each tip, where the fluid mesh is pinched and the
structure is not (fem_mesh.solid_foil_mesh). The operator built here handles
both cases with the same code, and reports the offset so the non-matching
region is visible rather than assumed away.

Three properties are wanted from a transfer pair, and this module gets all
three by construction rather than by tolerance.

**Conservation of force and moment when the panel loads are lumped.** The
fluid returns a force per PANEL, at the collocation point; the structure wants
forces at nodes. `thick_panel_wing.centroid_weights` already returns the
weights of the linear map that PLACED that collocation point, and they are a
partition of unity that reproduces the centroid exactly. Scattering a panel
force to its four corners with those same weights therefore preserves the total
force and the total moment about any point to round-off. No new kernel: the
operator is the adjoint of `sample_at_centroids`, which is what the fluid
itself uses to sample the body velocity at the collocation points, so the
discrete power of the tractions against the transferred nodal velocities is
exactly the power the fluid sees.

**Conservation of virtual work across the non-matching interface.** Motion goes
forward through one operator H and loads come back through H^T. The two sides
then agree on the work identically, which is what makes the coupled energy
balance meaningful and what a conservative partitioned scheme is defined by.

**Exactness for rigid-body motion.** H interpolates with the bilinear shape
functions of the structural surface face the fluid node projects onto. The
isoparametric map of a bilinear quad is itself bilinear, so a displacement
field linear in x - a translation, a rotation, or any combination - is
represented exactly. A transfer that failed this would put spurious strain into
a structure that is merely being carried along, which is the classic way a
partitioned code develops a slow drift nobody can find.

Welded fluid nodes (the trailing-edge seam and the mirrored tip-pinch pairs)
take the row of their canonical node through `weld_map`, so duplicates receive
identical displacements and `update_points` never has to average a
disagreement into existence, while their loads still accumulate.
"""

import numpy as np

import thick_panel_wing as tpw

NEWTON_PROJECT = 12         # iterations of the isoparametric inverse map
CANDIDATES = 8              # nearest faces tested per fluid node


# ---------------------------------------------------------- 1 projection

def _bilinear(xi, eta):
    """Shape functions of a 4-node quad at (P,) natural coordinates. -> (P, 4)"""
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return 0.25 * (1.0 + xi[:, None] * signs[None, :, 0]) \
        * (1.0 + eta[:, None] * signs[None, :, 1])


def _bilinear_derivatives(xi, eta):
    """d N / d xi and d N / d eta. -> two (P, 4) arrays"""
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    d_xi = 0.25 * signs[None, :, 0] * (1.0 + eta[:, None] * signs[None, :, 1])
    d_eta = 0.25 * signs[None, :, 1] * (1.0 + xi[:, None] * signs[None, :, 0])
    return d_xi, d_eta


def project_to_quads(points, corners, clamp=True):
    """Closest point on each bilinear quad, in natural coordinates.

    points (P, 3), corners (P, 4, 3) - one quad per point, already paired.
    Returns (weights (P, 4), projected (P, 3), distance (P,)).

    Newton on the two orthogonality conditions r . x_xi = r . x_eta = 0 with
    r = x(xi, eta) - X. The second derivative of a bilinear map is the single
    constant twist vector, so the Jacobian is exact and the iteration converges
    in a handful of steps; clamping to the reference square afterwards keeps a
    fluid node that overhangs the structure attached to the nearest edge rather
    than extrapolating a shape function outside its element.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    quads = np.asarray(corners, dtype=float)
    xi = np.zeros(len(pts))
    eta = np.zeros(len(pts))
    twist = 0.25 * (quads[:, 0] - quads[:, 1] + quads[:, 2] - quads[:, 3])
    for _ in range(NEWTON_PROJECT):
        n_f = _bilinear(xi, eta)
        d_xi, d_eta = _bilinear_derivatives(xi, eta)
        pos = np.einsum("pa,pac->pc", n_f, quads)
        x_xi = np.einsum("pa,pac->pc", d_xi, quads)
        x_eta = np.einsum("pa,pac->pc", d_eta, quads)
        res = pos - pts
        f_1 = np.einsum("pc,pc->p", res, x_xi)
        f_2 = np.einsum("pc,pc->p", res, x_eta)
        j_11 = np.einsum("pc,pc->p", x_xi, x_xi)
        j_22 = np.einsum("pc,pc->p", x_eta, x_eta)
        j_12 = np.einsum("pc,pc->p", x_xi, x_eta) \
            + np.einsum("pc,pc->p", res, twist)
        det = j_11 * j_22 - j_12 ** 2
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        xi = xi - (j_22 * f_1 - j_12 * f_2) / det
        eta = eta - (j_11 * f_2 - j_12 * f_1) / det
        if clamp:
            xi = np.clip(xi, -1.0, 1.0)
            eta = np.clip(eta, -1.0, 1.0)
    n_f = _bilinear(xi, eta)
    pos = np.einsum("pa,pac->pc", n_f, quads)
    return n_f, pos, np.linalg.norm(pos - pts, axis=1)


# ------------------------------------------------------------ 2 operator

def build_transfer(fluid_state, model, candidates=CANDIDATES, iface=None):
    """Interpolation operator from structural nodes to fluid mesh nodes.

    Built ONCE, on the two reference configurations, and used unchanged for
    every displacement and every coupling iteration: that is what makes the
    coupling residual a fixed linear function of the structural state, which
    quasi-Newton acceleration requires.

    iface overrides `tpw.interface(fluid_state)` and the mesh shape taken from
    it, for the case where the structure faces only PART of the fluid mesh. The
    doubled mesh of a surface-piercing strut is that case: its image half has no
    structural counterpart, so the transfer is built on the immersed half alone
    and the image geometry is slaved by mirroring, which is a fixed linear map
    and so leaves the build-once property intact. `vent_mesh.half_interface`
    supplies it, and supplies a weld map that welds the deep tip only -
    `build_topology` welds both end stations, which on the half mesh would weld
    the upper- and lower-surface waterline nodes to each other and collapse the
    open root. fluid_state is then unused except as the default.

    Returns a dict with `rows` (M, 4) structural node indices, `weights`
    (M, 4), the fluid mesh `shape`, the `weld_map`, and the offset diagnostic.
    """
    iface = tpw.interface(fluid_state) if iface is None else iface
    shape = iface.get("shape", fluid_state["panels"]["points"].shape)
    x_f = iface["nodes"]
    weld = iface["weld_map"]
    faces = model["faces"]
    if not len(faces):
        raise ValueError("the structural model has no surface faces to "
                         "transfer through")
    x_s = model["nodes"]
    quads = x_s[faces]                                    # (Nf, 4, 3)
    centres = quads.mean(axis=1)
    canonical = np.unique(weld)
    d_2 = np.einsum("mc,mc->m", x_f[canonical], x_f[canonical])[:, None] \
        + np.einsum("fc,fc->f", centres, centres)[None, :] \
        - 2.0 * x_f[canonical] @ centres.T
    k = min(int(candidates), len(faces))
    near = np.argpartition(d_2, k - 1, axis=1)[:, :k]     # (Mc, k)

    best_w = np.zeros((len(canonical), 4))
    best_r = np.zeros((len(canonical), 4), dtype=np.int64)
    best_d = np.full(len(canonical), np.inf)
    best_p = np.zeros((len(canonical), 3))
    for col in range(k):
        f_id = near[:, col]
        w_col, pos, dist = project_to_quads(x_f[canonical], quads[f_id])
        take = dist < best_d
        best_d = np.where(take, dist, best_d)
        best_w[take] = w_col[take]
        best_r[take] = faces[f_id][take]
        best_p[take] = pos[take]

    rows = np.zeros((len(x_f), 4), dtype=np.int64)
    weights = np.zeros((len(x_f), 4))
    projected = np.zeros((len(x_f), 3))
    lookup = np.zeros(len(x_f), dtype=np.int64)
    lookup[canonical] = np.arange(len(canonical))
    src = lookup[weld]                                    # welded rows share one
    rows[:] = best_r[src]
    weights[:] = best_w[src]
    projected[:] = best_p[src]

    scale = float(np.linalg.norm(np.ptp(x_f, axis=0)))
    return {"rows": rows, "weights": weights, "weld_map": weld,
            "shape": shape,
            "n_struct": len(x_s), "offset": projected - x_f,
            "offset_max": float(np.abs(best_d).max()),
            "offset_rel": float(np.abs(best_d).max() / max(scale, 1e-30))}


def to_fluid(transfer, field):
    """Structural nodal field (Ns, 3) -> fluid nodal field (M, 3). H f."""
    val = np.asarray(field, dtype=float).reshape(-1, 3)
    return np.einsum("ma,mac->mc", transfer["weights"], val[transfer["rows"]])


def to_structure(transfer, field):
    """Fluid nodal field (M, 3) -> structural nodal field (Ns, 3). H^T f.

    The transpose of `to_fluid`, which is what makes the virtual work identical
    on the two sides. Welded fluid nodes carry the same row, so their loads
    accumulate onto the one structural footprint.
    """
    val = np.asarray(field, dtype=float).reshape(-1, 3)
    out = np.zeros((transfer["n_struct"], 3))
    contrib = transfer["weights"][:, :, None] * val[:, None, :]
    np.add.at(out, transfer["rows"].ravel(), contrib.reshape(-1, 3))
    return out


# ------------------------------------------------------- 3 the two directions

def displaced_points(transfer, reference_points, u_struct):
    """Absolute fluid mesh points for update_points. (nwrap+1, nspan+1, 3)"""
    ref = np.asarray(reference_points, dtype=float)
    return ref + to_fluid(transfer, u_struct).reshape(transfer["shape"])


def point_velocities(transfer, v_struct):
    """Fluid mesh nodal velocities for update_points. (nwrap+1, nspan+1, 3)

    Through the SAME operator as the displacement, which is what keeps u_rel
    consistent with the surface the tangency condition is imposed on.
    """
    return to_fluid(transfer, v_struct).reshape(transfer["shape"])


def panel_forces_to_nodes(pan, force_panels):
    """Panel forces (N, 3) -> fluid nodal forces (M, 3), conservatively.

    The weights are the fluid's own `centroid_weights`, a partition of unity
    reproducing the collocation point, so the total force and the total moment
    about any point are preserved to round-off. Evaluated on the CURRENT
    corners, because that is where the force acts.
    """
    weights = tpw.centroid_weights(pan["corners"])                # (N, 4)
    force = np.asarray(force_panels, dtype=float).reshape(-1, 3)
    n_nodes = pan["points"].reshape(-1, 3).shape[0]
    out = np.zeros((n_nodes, 3))
    contrib = weights[:, :, None] * force[:, None, :]
    np.add.at(out, pan["panel_nodes"].ravel(), contrib.reshape(-1, 3))
    return out


def structural_forces(transfer, pan, force_panels):
    """The whole load path in one call: panels -> fluid nodes -> structure."""
    return to_structure(transfer, panel_forces_to_nodes(pan, force_panels))


# --------------------------------------------------------- 4 diagnostics

def conservation_report(transfer, pan, force_panels, u_struct=None):
    """What the transfer did to the resultants and to the virtual work.

    Every entry is a difference that should be at round-off when the two
    surfaces coincide. The moment entry is the one that is NOT zero on a
    genuinely non-matching interface, because the load then acts at the
    projected point rather than at the fluid node; `offset_max` bounds it, and
    reporting both is the honest alternative to a tolerance that hides it.
    """
    force = np.asarray(force_panels, dtype=float).reshape(-1, 3)
    f_nodes = panel_forces_to_nodes(pan, force)
    f_struct = to_structure(transfer, f_nodes)
    x_f = pan["points"].reshape(-1, 3)
    scale = max(np.abs(force).sum(), 1e-300)
    lever = max(float(np.linalg.norm(np.ptp(x_f, axis=0))), 1e-300)
    out = {
        "force_panels": force.sum(axis=0),
        "force_lumped": f_nodes.sum(axis=0),
        "force_struct": f_struct.sum(axis=0),
        "moment_panels": np.cross(pan["centroids"], force).sum(axis=0),
        "moment_lumped": np.cross(x_f, f_nodes).sum(axis=0),
        "offset_max": transfer["offset_max"],
    }
    out["force_error_lump"] = float(np.abs(out["force_lumped"]
                                           - out["force_panels"]).max() / scale)
    out["force_error_transfer"] = float(np.abs(out["force_struct"]
                                               - out["force_lumped"]).max() / scale)
    out["moment_error_lump"] = float(np.abs(out["moment_lumped"]
                                            - out["moment_panels"]).max()
                                     / (scale * lever))
    if u_struct is not None:
        u_s = np.asarray(u_struct, dtype=float).reshape(-1, 3)
        u_f = to_fluid(transfer, u_s)
        work_f = float(np.einsum("mc,mc->", f_nodes, u_f))
        work_s = float(np.einsum("mc,mc->", f_struct, u_s))
        out["work_fluid"], out["work_struct"] = work_f, work_s
        out["work_error"] = abs(work_f - work_s) / max(abs(work_f), 1e-300)
    return out
