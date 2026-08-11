"""Three-dimensional finite element solid, for coupling to the panel method.

Ten dependency-ordered sections, the layout of thick_panel_wing.py, and the
same three discipline rules: nothing imports downward, every block is a pure
function over explicit arguments with all mutable data in an explicit state
dictionary so that two states coexist freely, and prompts, prints and figures
appear only under main().

    1 constants     Gauss rule, Newton tolerances, Jacobian floor
    2 imports       numpy; fem_materials
    3 shape         hex8 shape functions, derivatives, the 2x2x2 rule
    4 element       Jacobians, B matrices, incompatible modes, k_e, m_e, f_int
    5 model         build_model, validation, surface extraction, mass properties
    6 assembly      stiffness, mass, damping, internal force, energies
    7 constraints   clamps, prescribed values, grounded springs, free-DOF mask
    8 linear alg    Cholesky factor and cached triangular solves; eigenproblem
    9 solvers       solve_static (Newton), modes, step_dynamic (generalised-alpha)
    10 plotting/CLI lazy matplotlib, returns Figure, never show()

Three decisions are worth stating before the code, because each looks like an
inefficiency and none is.

The internal force is ALWAYS the integral of B^T sigma, never K u, even though
the two agree exactly for the linear elastic material of version one and a unit
test asserts that they do. Writing it the other way would put the constitutive
law back inside the element and would make a nonlinear material a rewrite
rather than a new class in fem_materials.py.

The operators are dense and factored once. That is the same contract the fluid
side states for its O(N^2) assembly and O(N^3) solve, and at the sizes these two
solvers are used together at - a few thousand structural degrees of freedom
against a thousand panels - the fluid dominates by an order of magnitude. scipy
is deliberately absent, upstream and here, so the triangular solves are written
out; each row is one vectorised dot product, which costs about ten milliseconds
at three thousand degrees of freedom against ten seconds for the factorisation
it is reused from.

step_dynamic is a PURE function of the committed level and the new load: it
returns the next level and never mutates the state. A strongly coupled time
step re-solves the same time level many times with different loads, and an
integrator that has already advanced its own state cannot be asked twice.
"""

import numpy as np

import fem_materials as fmat

# ------------------------------------------------------------- 1 constants

GAUSS_1D = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))   # 2-point rule, weights 1
TOL_NEWTON = 1e-8           # relative residual for the Newton loops. Not tighter:
                            # f_int is an integral of B^T sigma with large
                            # cancellations, so on a slender stiff structure the
                            # attainable relative residual bottoms out near 1e-9
                            # and a tighter target only spins the loop
MAX_NEWTON = 20
STAGNATION = 0.9            # a Newton step that fails to reduce the residual by
                            # this factor has reached round-off; stop rather than
                            # iterate to the cap
JAC_MIN = 1e-12             # relative floor on det J at a Gauss point
RHO_INF = 1.0               # generalised-alpha default: no numerical dissipation

# hex8 corner coordinates in the reference cube, bottom face first, each face
# anticlockwise about +zeta
NODE_XI = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])

# the six faces, each listed anticlockwise seen from OUTSIDE the element
FACE_NODES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


# --------------------------------------------------------------- 3 shape

def gauss_points():
    """The 2x2x2 rule: (points (8, 3), weights (8,)). Deterministic order."""
    g = np.array(GAUSS_1D)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts, np.ones(len(pts))


def shape_functions(xi):
    """Trilinear shape functions at (P, 3) natural coordinates. -> (P, 8)"""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    return 0.125 * np.prod(1.0 + xi[:, None, :] * NODE_XI[None, :, :], axis=2)


def shape_derivatives(xi):
    """d N_a / d xi_j at (P, 3) natural coordinates. -> (P, 8, 3)"""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    fac = 1.0 + xi[:, None, :] * NODE_XI[None, :, :]              # (P, 8, 3)
    out = np.empty_like(fac)
    for j in range(3):
        other = [k for k in range(3) if k != j]
        out[:, :, j] = 0.125 * NODE_XI[None, :, j] * fac[:, :, other[0]] \
            * fac[:, :, other[1]]
    return out


def bubble_derivatives(xi):
    """d M_i / d xi_j for the three Wilson bubbles M_i = 1 - xi_i^2. -> (P, 3, 3)

    Diagonal by construction, which is why the incompatible modes cost so
    little; the substance is entirely in where their gradient is evaluated
    (see element_arrays).
    """
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    out = np.zeros((len(xi), 3, 3))
    for i in range(3):
        out[:, i, i] = -2.0 * xi[:, i]
    return out


def _strain_operator(grad):
    """Gradient (..., n, 3) -> Voigt B (..., 6, 3n), engineering shear.

    Rows follow the module convention [11, 22, 33, 2*12, 2*23, 2*31]; column
    3a + c is the c component of node (or mode) a.
    """
    shape = grad.shape[:-2]
    n_a = grad.shape[-2]
    b_mat = np.zeros(shape + (6, 3 * n_a))
    for c in range(3):
        b_mat[..., c, c::3] = grad[..., :, c]
    b_mat[..., 3, 0::3] = grad[..., :, 1]        # 2 e12
    b_mat[..., 3, 1::3] = grad[..., :, 0]
    b_mat[..., 4, 1::3] = grad[..., :, 2]        # 2 e23
    b_mat[..., 4, 2::3] = grad[..., :, 1]
    b_mat[..., 5, 0::3] = grad[..., :, 2]        # 2 e31
    b_mat[..., 5, 2::3] = grad[..., :, 0]
    return b_mat


# -------------------------------------------------------------- 4 element

def element_arrays(nodes, elements, incompatible=True):
    """Everything about the reference configuration, computed once.

    Returns a dict with the compatible strain operator B (Ne, G, 6, 24), the
    incompatible operator Ba (Ne, G, 6, 9) or None, the integration measure
    w*detJ (Ne, G), the shape functions N (G, 8) and the element volumes (Ne,).

    The incompatible gradient is evaluated with the Jacobian at the element
    CENTRE and scaled by detJ0/detJ - the Taylor, Beresford and Wilson
    correction. Without it the element is more flexible but fails the patch
    test on a distorted mesh, which is the one thing an element may not do;
    with it the bubbles vanish identically under a linear displacement field
    and the patch test is exact to round-off. Both statements are gated.
    """
    xyz = np.asarray(nodes, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)
    coords = xyz[conn]                                            # (Ne, 8, 3)
    pts, wts = gauss_points()
    d_n = shape_derivatives(pts)                                  # (G, 8, 3)
    n_f = shape_functions(pts)                                    # (G, 8)
    jac = np.einsum("gai,naj->ngij", d_n, coords)                 # (Ne, G, 3, 3)
    det = np.linalg.det(jac)
    scale = np.abs(det).max() if det.size else 1.0
    bad = det <= JAC_MIN * max(scale, 1e-300)
    if np.any(bad):
        n_bad = int(np.argmax(bad.any(axis=1)))
        raise ValueError(
            f"element {n_bad} has a non-positive Jacobian at a Gauss point "
            f"(min det {det[n_bad].min():.3e}): the node ordering must be the "
            f"bottom face anticlockwise about +zeta then the top face in the "
            f"same order, and the element must not be inverted or fully "
            f"collapsed")
    inv_j = np.linalg.inv(jac)
    # jac[i, j] is dx_j/dxi_i, so dN/dx = jac^-1 dN/dxi contracts the inverse on
    # its SECOND index. On a rectangular grid the Jacobian is diagonal and the
    # other contraction gives the same answer, which is why only a distorted
    # patch test catches it.
    grad = np.einsum("gai,ngji->ngaj", d_n, inv_j)                # (Ne, G, 8, 3)
    wdet = wts[None, :] * det
    out = {"B": _strain_operator(grad), "wdet": wdet, "N": n_f,
           "dNdx": grad, "volume": wdet.sum(axis=1), "Ba": None,
           "gauss": (pts, wts)}
    if incompatible:
        d_m = bubble_derivatives(pts)                             # (G, 3, 3)
        d_n0 = shape_derivatives(np.zeros((1, 3)))                # centre
        jac0 = np.einsum("gai,naj->nij", d_n0, coords)            # (Ne, 3, 3)
        det0 = np.linalg.det(jac0)
        inv0 = np.linalg.inv(jac0)
        grad_m = np.einsum("gik,njk->ngij", d_m, inv0)            # (Ne, G, 3, 3)
        grad_m = grad_m * (det0[:, None] / det)[:, :, None, None]
        out["Ba"] = _strain_operator(grad_m)                      # (Ne, G, 6, 9)
    return out


def constant_operators(model):
    """Cached per-element tangent and condensed bubble inverse.

    Both are functions of the reference geometry and the material alone while
    the tangent is constant, which is every material in version one. Caching
    them is what makes `internal_force` - called two or three times per time
    step, and again for every coupling subiteration - a handful of einsums
    rather than a reassembly. A material with a state invalidates the cache and
    the assembly falls back to recomputing, which is the correct behaviour and
    the reason this is a lazily filled cache rather than a build-time field.
    """
    cache = model["cache"]
    if cache.get("d_all") is None:
        d_all = element_tangents(model)
        cache["d_all"] = d_all
        cache["k_aa_inv"] = element_stiffness(model, d_all)[1]
    return cache["d_all"], cache["k_aa_inv"]


def element_tangents(model):
    """Per-element (6, 6) tangent modulus for the constant-tangent materials.

    Materials that carry state return their tangent from `response` instead;
    this path exists so the elastic assembly does not call into the material
    once per Gauss point for an answer it already knows. The two paths are
    asserted to agree in the tests, so the optimisation cannot change an answer.
    """
    n_e = len(model["elements"])
    d_all = np.empty((n_e, 6, 6))
    for m_id, mat in enumerate(model["materials"]):
        sel = np.where(model["mat_id"] == m_id)[0]
        if not len(sel):
            continue
        if not getattr(mat, "constant_tangent", False):
            raise ValueError(f"material '{mat.name}' has no constant tangent; "
                             f"assemble through internal_force instead")
        if model["orientation"] is None:
            d_all[sel] = mat.tangent()
        else:
            for n in sel:
                d_all[n] = mat.tangent(model["orientation"][n])
    return d_all


def element_stiffness(model, d_all):
    """Condensed element stiffness (Ne, 24, 24) for a given tangent per element.

    With incompatible modes the element system is [[Kuu, Kua], [Kau, Kaa]] on
    (u, alpha) with no external work on alpha, so alpha condenses out exactly:
    Ke = Kuu - Kua Kaa^-1 Kau. The condensation is at element level and is
    algebraically exact, not an approximation.
    """
    cache = model["cache"]
    b_mat, wdet = cache["B"], cache["wdet"]
    db = np.einsum("nkl,nglj->ngkj", d_all, b_mat)
    k_uu = np.einsum("ngki,ngkj,ng->nij", b_mat, db, wdet)
    if cache["Ba"] is None:
        return k_uu, None
    b_a = cache["Ba"]
    dba = np.einsum("nkl,nglj->ngkj", d_all, b_a)
    k_ua = np.einsum("ngki,ngkj,ng->nij", b_mat, dba, wdet)
    k_aa = np.einsum("ngki,ngkj,ng->nij", b_a, dba, wdet)
    k_aa_inv = np.linalg.inv(k_aa)
    return k_uu - k_ua @ k_aa_inv @ k_ua.transpose(0, 2, 1), k_aa_inv


def element_mass(model, lumped=False):
    """Consistent (or HRZ-lumped) element mass (Ne, 24, 24)."""
    cache = model["cache"]
    n_f, wdet = cache["N"], cache["wdet"]
    rho = np.array([model["materials"][m].rho for m in model["mat_id"]])
    m_ab = np.einsum("ga,gb,ng->nab", n_f, n_f, wdet * rho[:, None])
    m_e = np.einsum("nab,cd->nacbd", m_ab, np.eye(3)).reshape(len(m_ab), 24, 24)
    if not lumped:
        return m_e
    diag = np.einsum("nii->ni", m_e)
    total = (rho * cache["volume"])[:, None]
    scaled = diag * (3.0 * total / np.maximum(diag.sum(axis=1, keepdims=True),
                                              1e-300))
    out = np.zeros_like(m_e)
    idx = np.arange(24)
    out[:, idx, idx] = scaled
    return out


def element_alpha(model, u_e, d_all, k_aa_inv):
    """Internal incompatible amplitudes (Ne, 9) from the nodal displacements.

    Kaa alpha = -Kau u, the element-level equilibrium of the bubbles, which for
    a linear material is one solve. A material with a state would need a local
    Newton iteration here and nowhere else, which is what keeps the nonlinear
    extension local.
    """
    if k_aa_inv is None:
        return None
    cache = model["cache"]
    dba = np.einsum("nkl,nglj->ngkj", d_all, cache["Ba"])
    k_au = np.einsum("ngki,ngkj,ng->nij", dba, cache["B"], cache["wdet"])
    return -np.einsum("nij,njk,nk->ni", k_aa_inv, k_au, u_e)


def element_strain(model, u_e, alpha=None):
    """Voigt strain at every Gauss point. (Ne, G, 6)"""
    cache = model["cache"]
    eps = np.einsum("ngki,ni->ngk", cache["B"], u_e)
    if alpha is not None and cache["Ba"] is not None:
        eps = eps + np.einsum("ngki,ni->ngk", cache["Ba"], alpha)
    return eps


# ---------------------------------------------------------------- 5 model

def build_model(nodes, elements, materials, mat_id=None, orientation=None,
                incompatible=True):
    """Topology, material and reference-configuration cache. Built once.

    materials is a Material or a list of them; mat_id (Ne,) selects one per
    element, so a layered or spatially varying material needs no new structure.
    orientation (Ne, 3, 3) rotates the material axes into the global frame,
    None meaning aligned - the hook a fibre-reinforced lay-up uses.
    """
    xyz = np.asarray(nodes, dtype=float)
    conn = np.asarray(elements, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"nodes must be (Ns, 3); got {xyz.shape}")
    if conn.ndim != 2 or conn.shape[1] != 8:
        raise ValueError(f"elements must be (Ne, 8) hex8 connectivity; "
                         f"got {conn.shape}")
    if conn.min() < 0 or conn.max() >= len(xyz):
        raise ValueError("element connectivity indexes a node that does not exist")
    if not isinstance(materials, (list, tuple)):
        materials = [materials]
    mat_id = np.zeros(len(conn), dtype=np.int64) if mat_id is None \
        else np.asarray(mat_id, dtype=np.int64)
    if len(mat_id) != len(conn) or mat_id.max(initial=0) >= len(materials):
        raise ValueError("mat_id must be (Ne,) and index the materials list")
    if orientation is not None:
        orientation = np.asarray(orientation, dtype=float)
        if orientation.shape == (3, 3):
            orientation = np.broadcast_to(orientation, (len(conn), 3, 3)).copy()
        if orientation.shape != (len(conn), 3, 3):
            raise ValueError(f"orientation must be (3, 3) or (Ne, 3, 3); "
                             f"got {orientation.shape}")
    model = {"nodes": xyz, "elements": conn, "etype": "hex8",
             "materials": list(materials), "mat_id": mat_id,
             "orientation": orientation, "incompatible": bool(incompatible)}
    model["cache"] = element_arrays(xyz, conn, incompatible)
    model["faces"], model["face_elem"] = surface_faces(conn)
    return model


def surface_faces(elements):
    """Outer boundary of a hex mesh: (Nf, 4) node quads and their element ids.

    A face shared by two elements is interior and drops out. Collapsed faces
    (the wedge-degenerate elements at a foil leading and trailing edge have
    them) carry a repeated node and are recognised by their node SET, so they
    pair correctly instead of appearing twice on the boundary.
    """
    conn = np.asarray(elements, dtype=np.int64)
    seen = {}
    for n, elem in enumerate(conn):
        for f_local in FACE_NODES:
            quad = elem[f_local]
            key = tuple(sorted(set(int(v) for v in quad)))
            if len(key) < 3:                      # fully collapsed: no area
                continue
            if key in seen:
                seen[key] = None                  # interior
            else:
                seen[key] = (quad.copy(), n)
    faces = [v for v in seen.values() if v is not None]
    if not faces:
        return np.zeros((0, 4), dtype=np.int64), np.zeros(0, dtype=np.int64)
    quads = np.array([f[0] for f in faces], dtype=np.int64)
    owners = np.array([f[1] for f in faces], dtype=np.int64)
    return quads, owners


def mass_properties(model):
    """Total mass, centre of mass and inertia tensor about it. Analytic gate.

    Integrated with the element rule on the density field, so it is exact for
    a mesh of parallelepipeds and converges otherwise; the box gate compares it
    with the closed form.
    """
    cache = model["cache"]
    rho = np.array([model["materials"][m].rho for m in model["mat_id"]])
    coords = model["nodes"][model["elements"]]
    x_g = np.einsum("ga,nac->ngc", cache["N"], coords)            # (Ne, G, 3)
    dm = cache["wdet"] * rho[:, None]                             # (Ne, G)
    mass = float(dm.sum())
    com = np.einsum("ng,ngc->c", dm, x_g) / max(mass, 1e-300)
    rel = x_g - com
    r2 = np.einsum("ngc,ngc->ng", rel, rel)
    inertia = np.einsum("ng,ng,ij->ij", dm, r2, np.eye(3)) \
        - np.einsum("ng,ngi,ngj->ij", dm, rel, rel)
    return {"mass": mass, "centre": com, "inertia": inertia,
            "volume": float(cache["volume"].sum())}


# ------------------------------------------------------------- 6 assembly

def dof_map(elements):
    """Element degree-of-freedom indices. (Ne, 24), dof = 3*node + component."""
    conn = np.asarray(elements, dtype=np.int64)
    return (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), 24)


def _scatter(k_e, edof, ndof):
    """Element matrices (Ne, 24, 24) -> dense global (ndof, ndof)."""
    rows = np.repeat(edof[:, :, None], 24, axis=2)
    cols = np.repeat(edof[:, None, :], 24, axis=1)
    flat = (rows * ndof + cols).ravel()
    return np.bincount(flat, weights=k_e.ravel(),
                       minlength=ndof * ndof).reshape(ndof, ndof)


def assemble_stiffness(model):
    """Global tangent stiffness for the constant-tangent path. (ndof, ndof)"""
    d_all = element_tangents(model)
    k_e, _ = element_stiffness(model, d_all)
    ndof = 3 * len(model["nodes"])
    return _scatter(k_e, dof_map(model["elements"]), ndof)


def assemble_mass(model, lumped=False):
    """Global mass matrix. (ndof, ndof)"""
    m_e = element_mass(model, lumped)
    ndof = 3 * len(model["nodes"])
    return _scatter(m_e, dof_map(model["elements"]), ndof)


def assemble_rate_damping(model):
    """Global damping from the materials' rate tangent, zero if none has one.

    A Kelvin-Voigt material returns tau*D as its rate tangent, so this is
    exactly tau*K - stiffness-proportional damping arrived at through the
    constitutive law rather than fitted to it.
    """
    mats = model["materials"]
    if all(m.response(np.zeros((1, 6)))["rate_tangent"] is None for m in mats):
        return None
    n_e = len(model["elements"])
    d_all = np.zeros((n_e, 6, 6))
    for m_id, mat in enumerate(mats):
        sel = np.where(model["mat_id"] == m_id)[0]
        if not len(sel):
            continue
        rate = mat.response(np.zeros((1, 6)))["rate_tangent"]
        if rate is None:
            continue
        base = np.asarray(rate)[0]
        if model["orientation"] is None:
            d_all[sel] = base
        else:
            for n in sel:
                d_all[n] = fmat.rotate_tangent(base, model["orientation"][n])
    c_e, _ = element_stiffness(model, d_all)
    ndof = 3 * len(model["nodes"])
    return _scatter(c_e, dof_map(model["elements"]), ndof)


def internal_force(model, u, v=None, history=None, dt=None):
    """Integral of B^T sigma over the model, with the material in the loop.

    Returns (f_int (Ns, 3), updated history, stress (Ne, G, 6)). This is the
    only route by which stress reaches the equilibrium equations; K u is never
    used for it, even where the two agree. The incompatible amplitudes are
    resolved first from the element-level equilibrium of the bubbles.
    """
    conn = model["elements"]
    edof = dof_map(conn)
    u_flat = np.asarray(u, dtype=float).reshape(-1)
    u_e = u_flat[edof]                                            # (Ne, 24)
    d_all, k_aa_inv = constant_operators(model)
    alpha = element_alpha(model, u_e, d_all, k_aa_inv)
    eps = element_strain(model, u_e, alpha)
    rate = None
    if v is not None:
        v_e = np.asarray(v, dtype=float).reshape(-1)[edof]
        alpha_v = element_alpha(model, v_e, d_all, k_aa_inv)
        rate = element_strain(model, v_e, alpha_v)
    n_e, n_g = eps.shape[0], eps.shape[1]
    stress = np.empty_like(eps)
    new_hist = None if history is None else np.array(history, dtype=float)
    for m_id, mat in enumerate(model["materials"]):
        sel = np.where(model["mat_id"] == m_id)[0]
        if not len(sel):
            continue
        orient = None if model["orientation"] is None \
            else model["orientation"][sel[0]]
        flat_eps = eps[sel].reshape(-1, 6)
        flat_rate = None if rate is None else rate[sel].reshape(-1, 6)
        flat_hist = None if history is None else \
            np.asarray(history)[sel].reshape(len(sel) * n_g, -1)
        res = mat.response(flat_eps, strain_rate=flat_rate, history=flat_hist,
                           dt=dt, orientation=orient)
        stress[sel] = res["stress"].reshape(len(sel), n_g, 6)
        if new_hist is not None and mat.n_history:
            new_hist[sel] = res["history"].reshape(len(sel), n_g, -1)
    f_e = np.einsum("ngki,ngk,ng->ni", model["cache"]["B"], stress,
                    model["cache"]["wdet"])
    ndof = 3 * len(model["nodes"])
    f_int = np.bincount(edof.ravel(), weights=f_e.ravel(), minlength=ndof)
    return f_int.reshape(-1, 3), new_hist, stress


def strain_energy(model, u):
    """0.5 * integral of sigma . epsilon. Scalar."""
    _, _, stress = internal_force(model, u)
    conn = model["elements"]
    u_e = np.asarray(u, dtype=float).reshape(-1)[dof_map(conn)]
    d_all, k_aa_inv = constant_operators(model)
    eps = element_strain(model, u_e, element_alpha(model, u_e, d_all, k_aa_inv))
    return 0.5 * float(np.einsum("ngk,ngk,ng->", stress, eps,
                                 model["cache"]["wdet"]))


def kinetic_energy(sstate, v=None):
    """0.5 * v^T M v. Scalar."""
    vel = sstate["v"] if v is None else v
    flat = np.asarray(vel, dtype=float).reshape(-1)
    return 0.5 * float(flat @ (sstate["M"] @ flat))


# ---------------------------------------------------------- 7 constraints

def init_state(model, rayleigh=(0.0, 0.0), lumped_mass=False):
    """Structural state: level, constraints and operators. No I/O, no prints."""
    n_s = len(model["nodes"])
    n_g = model["cache"]["wdet"].shape[1]
    n_hist = max((m.n_history for m in model["materials"]), default=0)
    return {"model": model,
            "u": np.zeros((n_s, 3)), "v": np.zeros((n_s, 3)),
            "a": np.zeros((n_s, 3)), "f": np.zeros((n_s, 3)),
            "history": np.zeros((len(model["elements"]), n_g, n_hist)),
            "fixed": np.zeros(3 * n_s, dtype=bool),
            "spring": np.zeros(3 * n_s),
            "rayleigh": (float(rayleigh[0]), float(rayleigh[1])),
            "lumped_mass": bool(lumped_mass),
            "K": None, "M": None, "C": None, "C_mat": None,
            "factor": None, "factor_key": None,
            "operators_version": 0, "constraint_version": 0,
            "residuals": {"newton": 0, "resid": np.inf}}


def clamp(sstate, node_ids, components=(0, 1, 2)):
    """Fix the given components of the given nodes. In place; returns sstate."""
    nodes = np.atleast_1d(np.asarray(node_ids, dtype=np.int64))
    for c in components:
        sstate["fixed"][3 * nodes + c] = True
    sstate["constraint_version"] += 1
    sstate["factor"] = None
    return sstate


def add_spring(sstate, node_ids, components, stiffness):
    """Grounded linear springs on the given degrees of freedom. In place.

    Enough to build an elastically supported wing - the plunge and pitch
    springs of a typical section are grounded translational springs on a root
    rib - without a second element type. Energy 0.5*k*u^2 per degree of
    freedom, so the springs add to the diagonal of K and to nothing else.
    """
    nodes = np.atleast_1d(np.asarray(node_ids, dtype=np.int64))
    for c in np.atleast_1d(components):
        sstate["spring"][3 * nodes + int(c)] += float(stiffness)
    sstate["constraint_version"] += 1
    sstate["factor"] = None
    return sstate


def free_mask(sstate):
    """(ndof,) boolean: True where the degree of freedom is unknown."""
    return ~sstate["fixed"]


def assemble_operators(sstate):
    """Fill K, M and C for the current model and constraints. Returns sstate.

    K carries the grounded springs on its diagonal.

    The two damping paths are kept apart on purpose and the distinction is not
    cosmetic. Rayleigh damping is a modelling device with no constitutive
    content, so it appears in the equilibrium equation directly as C. A
    material's rate tangent is the constitutive law, so its stress reaches
    equilibrium through `internal_force(v=...)` like any other stress, and the
    matrix assembled here (C_mat) is used ONLY to build the effective dynamic
    operator, where it is a tangent. Adding it to C as well would count the
    viscous stress twice, which is the obvious mistake this split prevents.
    """
    model = sstate["model"]
    k_mat = assemble_stiffness(model)
    idx = np.arange(len(k_mat))
    k_mat[idx, idx] += sstate["spring"]
    m_mat = assemble_mass(model, sstate["lumped_mass"])
    a_0, a_1 = sstate["rayleigh"]
    sstate["K"], sstate["M"] = k_mat, m_mat
    sstate["C"] = a_0 * m_mat + a_1 * k_mat
    rate = assemble_rate_damping(model)
    sstate["C_mat"] = np.zeros_like(k_mat) if rate is None else rate
    sstate["operators_version"] += 1
    sstate["factor"] = None
    return sstate


# -------------------------------------------------------- 8 linear algebra

def cholesky_factor(a_mat, what="operator"):
    """Lower Cholesky factor, with a diagnostic instead of a LinAlgError."""
    try:
        return np.linalg.cholesky(a_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"the reduced {what} is not positive definite: the model is under-"
            f"constrained (a rigid-body mode or a detached node survives the "
            f"boundary conditions) or a material is not positive definite"
        ) from exc


def forward_substitute(low, rhs):
    """Solve L y = b for lower-triangular L. b is (n,) or (n, m)."""
    b_mat = np.array(rhs, dtype=float)
    single = b_mat.ndim == 1
    if single:
        b_mat = b_mat[:, None]
    y = np.empty_like(b_mat)
    for i in range(len(low)):
        y[i] = (b_mat[i] - low[i, :i] @ y[:i]) / low[i, i]
    return y[:, 0] if single else y


def backward_substitute(low, rhs):
    """Solve L^T x = y for lower-triangular L. y is (n,) or (n, m)."""
    y_mat = np.array(rhs, dtype=float)
    single = y_mat.ndim == 1
    if single:
        y_mat = y_mat[:, None]
    x = np.empty_like(y_mat)
    n = len(low)
    for i in range(n - 1, -1, -1):
        x[i] = (y_mat[i] - low[i + 1:, i] @ x[i + 1:]) / low[i, i]
    return x[:, 0] if single else x


def cholesky_solve(low, rhs):
    """Solve A x = b from the factor of A. O(n^2), reusing an O(n^3) factor."""
    return backward_substitute(low, forward_substitute(low, rhs))


def generalised_modes(k_mat, m_mat, nmodes=None):
    """Solve K phi = omega^2 M phi. Returns (omega (m,), phi (n, m)).

    Reduced to a symmetric standard problem through the Cholesky factor of M,
    which is what keeps numpy.linalg.eigh - the only symmetric eigensolver
    available without scipy - applicable and the eigenvalues real.
    """
    low = cholesky_factor(m_mat, "mass matrix")
    tmp = forward_substitute(low, k_mat)
    a_mat = forward_substitute(low, tmp.T).T
    a_mat = 0.5 * (a_mat + a_mat.T)
    vals, vecs = np.linalg.eigh(a_mat)
    phi = backward_substitute(low, vecs)
    omega = np.sqrt(np.maximum(vals, 0.0))
    if nmodes is not None:
        omega, phi = omega[:nmodes], phi[:, :nmodes]
    return omega, phi


# ----------------------------------------------------------- 9 solvers

def _reduce(mat, free):
    return mat[np.ix_(free, free)]


def solve_static(sstate, f_ext, tol=TOL_NEWTON, max_iter=MAX_NEWTON,
                 u0=None):
    """Newton on r = f_ext - f_int(u). One iteration when the material is linear.

    Returns a level dict {u, v, a, history}; sstate is not mutated, so the same
    state serves repeated coupling iterations. The loop is here rather than a
    single linear solve because it is the entire difference between a code that
    extends to inelastic materials and one that does not: an elastic-plastic
    material changes `internal_force` and nothing else.
    """
    model = sstate["model"]
    free = free_mask(sstate)
    if sstate["K"] is None:
        assemble_operators(sstate)
    k_ff = _reduce(sstate["K"], free)
    low = cholesky_factor(k_ff, "stiffness matrix")
    u = np.zeros_like(sstate["u"]) if u0 is None else np.array(u0, dtype=float)
    f_ext = np.asarray(f_ext, dtype=float).reshape(-1, 3)
    hist = sstate["history"]
    # measured against the FIRST residual, not against the applied force: a
    # free-vibration or self-equilibrated case has no external force and a
    # scale taken from it alone collapses to zero, which silently runs every
    # step to the iteration cap instead of converging
    scale, n_it, resid = None, 0, np.inf
    for n_it in range(1, max_iter + 1):
        f_int, hist_new, _ = internal_force(model, u)
        f_int = f_int + (sstate["spring"] * u.reshape(-1)).reshape(-1, 3)
        r = (f_ext - f_int).reshape(-1)
        r[~free] = 0.0
        if scale is None:
            scale = max(np.linalg.norm(r),
                        np.linalg.norm(f_ext.reshape(-1)[free]), 1e-300)
        previous, resid = resid, np.linalg.norm(r) / scale
        if resid < tol or resid > STAGNATION * previous:
            break
        du = np.zeros(len(r))
        du[free] = cholesky_solve(low, r[free])
        u = u + du.reshape(-1, 3)
    sstate["residuals"] = {"newton": n_it, "resid": float(resid)}
    return {"u": u, "v": np.zeros_like(u), "a": np.zeros_like(u),
            "history": hist if hist_new is None else hist_new}


def modes(sstate, nmodes=6):
    """Free-vibration frequencies (Hz) and mode shapes of the CONSTRAINED model.

    Returns (freq (m,), shapes (Ns, 3, m)) with shapes mass-normalised and the
    constrained degrees of freedom filled with zeros.
    """
    if sstate["K"] is None:
        assemble_operators(sstate)
    free = free_mask(sstate)
    omega, phi_f = generalised_modes(_reduce(sstate["K"], free),
                                     _reduce(sstate["M"], free), nmodes)
    phi = np.zeros((len(free), phi_f.shape[1]))
    phi[free] = phi_f
    return omega / (2.0 * np.pi), phi.reshape(-1, 3, phi_f.shape[1])


def integrator_parameters(rho_inf=RHO_INF, alpha_m=None, alpha_f=None):
    """Chung and Hulbert generalised-alpha parameters from the spectral radius.

    rho_inf = 1 gives alpha_m = alpha_f = 1/2, the midpoint rule: second order
    and non-dissipative. Classical Newmark average acceleration is alpha_m =
    alpha_f = 0, beta = 1/4, gamma = 1/2, obtained by passing both explicitly.
    Lower rho_inf damps the unresolved high modes, which is what a coupled run
    on a stiff mesh generally wants; the price is second-order accuracy only in
    the limit, and the verification tabulates the measured decay against the
    theoretical spectral radius.
    """
    if alpha_m is None or alpha_f is None:
        rho = float(rho_inf)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho_inf must lie in [0, 1]; got {rho}")
        alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
        alpha_f = rho / (rho + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    gamma = 0.5 - alpha_m + alpha_f
    return {"alpha_m": float(alpha_m), "alpha_f": float(alpha_f),
            "beta": float(beta), "gamma": float(gamma)}


def _effective_operator(sstate, dt, par):
    """Cached factor of the generalised-alpha effective stiffness."""
    key = (round(float(dt), 15), par["alpha_m"], par["alpha_f"], par["beta"],
           par["gamma"], sstate["operators_version"], sstate["constraint_version"])
    if sstate["factor"] is not None and sstate["factor_key"] == key:
        return sstate["factor"]
    free = free_mask(sstate)
    # the effective operator is written in stiffness units, so it is the true
    # tangent d(residual)/d(du) with du = beta dt^2 a_{n+1} the increment solved
    # for; solving for the acceleration directly with this operator would be
    # wrong by exactly that factor
    c_m = (1.0 - par["alpha_m"]) / (par["beta"] * dt ** 2)
    c_c = (1.0 - par["alpha_f"]) * par["gamma"] / (par["beta"] * dt)
    c_k = 1.0 - par["alpha_f"]
    damping = _reduce(sstate["C"], free) + _reduce(sstate["C_mat"], free)
    eff = c_m * _reduce(sstate["M"], free) + c_c * damping \
        + c_k * _reduce(sstate["K"], free)
    low = cholesky_factor(eff, "effective dynamic operator")
    sstate["factor"], sstate["factor_key"] = low, key
    return low


def step_dynamic(sstate, f_new, dt, par=None, f_old=None, tol=TOL_NEWTON,
                 max_iter=MAX_NEWTON):
    """One generalised-alpha step. PURE: returns the next level, mutates nothing.

    Chung and Hulbert equilibrium at the intermediate level,

        M a_{n+1-am} + C v_{n+1-af} + f_int(u_{n+1-af}, v_{n+1-af}) = f_{n+1-af},

    with the Newmark update between the levels. The internal force comes from
    the material through `internal_force`, not from K u, so a nonlinear or
    inelastic material needs no change here; the loop below then simply takes
    more than the single iteration a linear material takes. C is the Rayleigh
    damping only - a material's viscous stress is already inside f_int.

    Purity is what a strongly coupled step needs: the same time level is
    re-solved several times with different loads, and an integrator that has
    advanced its own state cannot be asked twice. The driver commits the level
    when the coupling converges.
    """
    if sstate["K"] is None:
        assemble_operators(sstate)
    par = par or integrator_parameters()
    free = free_mask(sstate)
    low = _effective_operator(sstate, dt, par)
    a_m, a_f = par["alpha_m"], par["alpha_f"]
    beta, gamma = par["beta"], par["gamma"]
    u_n, v_n, a_n = (sstate[k].reshape(-1) for k in ("u", "v", "a"))
    f_n = (sstate["f"] if f_old is None else np.asarray(f_old)).reshape(-1)
    f_1 = np.asarray(f_new, dtype=float).reshape(-1)
    f_ext = (1.0 - a_f) * f_1 + a_f * f_n

    # predictor: a_{n+1} = 0, so u and v follow from level n alone
    u_p = u_n + dt * v_n + dt ** 2 * (0.5 - beta) * a_n
    v_p = v_n + dt * (1.0 - gamma) * a_n
    du = np.zeros_like(u_n)                        # du = beta dt^2 a_{n+1}
    hist = sstate["history"]
    # the reference is the predictor residual, not the applied force: in free
    # vibration the latter is zero and every step would run to the cap
    scale, resid = None, np.inf
    for _ in range(max_iter):
        acc = du / (beta * dt ** 2)
        u_1, v_1 = u_p + du, v_p + gamma * dt * acc
        u_int = (1.0 - a_f) * u_1 + a_f * u_n
        v_int = (1.0 - a_f) * v_1 + a_f * v_n
        a_int = (1.0 - a_m) * acc + a_m * a_n
        f_int, hist_new, _ = internal_force(sstate["model"],
                                            u_int.reshape(-1, 3),
                                            v=v_int.reshape(-1, 3),
                                            history=sstate["history"], dt=dt)
        f_int = f_int.reshape(-1) + sstate["spring"] * u_int
        res = f_ext - sstate["M"] @ a_int - sstate["C"] @ v_int - f_int
        res[~free] = 0.0
        hist = sstate["history"] if hist_new is None else hist_new
        if scale is None:
            scale = max(np.linalg.norm(res), np.linalg.norm(f_ext[free]), 1e-300)
        previous, resid = resid, np.linalg.norm(res) / scale
        if resid < tol or resid > STAGNATION * previous:
            break
        step = np.zeros_like(du)
        step[free] = cholesky_solve(low, res[free])
        du = du + step
    acc = du / (beta * dt ** 2)
    return {"u": (u_p + du).reshape(-1, 3),
            "v": (v_p + gamma * dt * acc).reshape(-1, 3),
            "a": acc.reshape(-1, 3), "history": hist}


def initial_acceleration(sstate, f_ext):
    """a_0 from M a = f - C v - f_int(u), the consistent start of a march."""
    if sstate["K"] is None:
        assemble_operators(sstate)
    free = free_mask(sstate)
    model = sstate["model"]
    f_int, _, _ = internal_force(model, sstate["u"], v=sstate["v"])
    f_int = f_int + (sstate["spring"] * sstate["u"].reshape(-1)).reshape(-1, 3)
    rhs = (np.asarray(f_ext, dtype=float) - f_int).reshape(-1) \
        - sstate["C"] @ sstate["v"].reshape(-1)
    rhs[~free] = 0.0
    low = cholesky_factor(_reduce(sstate["M"], free), "mass matrix")
    acc = np.zeros(len(rhs))
    acc[free] = cholesky_solve(low, rhs[free])
    return acc.reshape(-1, 3)


def commit(sstate, level, f_ext=None):
    """Install a level as the state of record. In place; returns sstate."""
    sstate["u"] = np.array(level["u"], dtype=float)
    sstate["v"] = np.array(level["v"], dtype=float)
    sstate["a"] = np.array(level["a"], dtype=float)
    if level.get("history") is not None:
        sstate["history"] = np.array(level["history"], dtype=float)
    if f_ext is not None:
        sstate["f"] = np.asarray(f_ext, dtype=float).reshape(-1, 3)
    return sstate


# ------------------------------------------------------ 10 plotting and CLI

def plot_deformed(model, u, scale=1.0, save=None):
    """Wireframe of the surface faces, undeformed and deformed. Returns Figure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    quads = model["faces"]
    for disp, colour, alpha, label in ((0.0, "0.7", 0.15, "reference"),
                                       (scale, "tab:blue", 0.45, "deformed")):
        pts = model["nodes"] + disp * np.asarray(u, dtype=float)
        poly = Poly3DCollection(pts[quads], facecolor=colour, edgecolor="k",
                                linewidths=0.2, alpha=alpha)
        ax.add_collection3d(poly)
        ax.plot([], [], color=colour, label=label)
    pts = model["nodes"] + scale * np.asarray(u, dtype=float)
    span = np.ptp(pts, axis=0).max()
    mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    for setter, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setter(m - 0.6 * span, m + 0.6 * span)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.legend(loc="upper right")
    ax.set_title(f"deformation, magnified {scale:g}x")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    return fig


def main():
    """Cantilever demonstration: tip deflection and the first three frequencies."""
    import fem_mesh

    print("fem_solid.py - three-dimensional finite element solid\n")
    length = float(input("cantilever length [1.0]: ") or 1.0)
    width = float(input("width [0.1]: ") or 0.1)
    thick = float(input("thickness [0.02]: ") or 0.02)
    e_mod = float(input("Young's modulus [70e9]: ") or 70e9)
    rho = float(input("density [2700]: ") or 2700.0)
    tip = float(input("tip load, z [-100.0]: ") or -100.0)
    n_l = int(input("elements along the length [16]: ") or 16)

    mesh = fem_mesh.box_mesh(length, width, thick, n_l, 2, 2)
    material = fem_materials_isotropic(e_mod, 0.3, rho)
    model = build_model(mesh["nodes"], mesh["elements"], material)
    sstate = init_state(model)
    clamp(sstate, mesh["node_sets"]["x_min"])
    assemble_operators(sstate)

    f_ext = np.zeros_like(sstate["u"])
    tip_nodes = mesh["node_sets"]["x_max"]
    f_ext[tip_nodes, 2] = tip / len(tip_nodes)
    level = solve_static(sstate, f_ext)
    commit(sstate, level, f_ext)

    inertia = width * thick ** 3 / 12.0
    beam = tip * length ** 3 / (3.0 * e_mod * inertia)
    print(f"\ntip deflection      {sstate['u'][tip_nodes, 2].mean():+.6e} m")
    print(f"Euler-Bernoulli     {beam:+.6e} m")
    freq, _ = modes(sstate, 3)
    coeff = (1.875104, 4.694091, 7.854757)
    area = width * thick
    for n, (f, b) in enumerate(zip(freq, coeff), start=1):
        exact = b ** 2 / (2.0 * np.pi) * np.sqrt(e_mod * inertia
                                                 / (rho * area * length ** 4))
        print(f"mode {n}: {f:9.4f} Hz   Euler-Bernoulli {exact:9.4f} Hz")

    import matplotlib.pyplot as plt
    plot_deformed(model, sstate["u"], scale=0.2 * length
                  / max(abs(sstate["u"]).max(), 1e-30))
    plt.show()


def fem_materials_isotropic(e_mod, nu, rho):
    """Convenience shim so main() reads without a second import alias."""
    return fmat.IsotropicElastic(e_mod, nu, rho)


if __name__ == "__main__":
    main()
