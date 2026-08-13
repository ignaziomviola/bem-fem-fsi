"""The surface-piercing strut: its doubled mesh, and the mirror maps.

A surface-piercing strut has an open section at the waterline, and the vendored
solver cannot represent one: `validate_mesh` requires both span-end stations to
be collapsed onto the camber line, and `update_points` re-pins them
unconditionally on every call. The free surface therefore enters as an exact
NEGATIVE IMAGE realised by mesh doubling - the immersed strut plus its mirror as
one legal closed thick-wrap-1 body, with the free surface at mid-span and a
FULL-CHORD section there, which is where the cavity is longest and where a
single pinched foil of span h would have had no chord at all.

Seven dependency-ordered sections, and the same three discipline rules: nothing
imports downward, every block is a pure function over explicit arguments with all
mutable data in an explicit state dictionary, and prompts, prints and figures
appear only under main().

    1 constants     the depth axis, and the mirror reflection
    2 imports       numpy; thick_panel_wing, make_thick_sample_inputs, fem_mesh
    3 the strut     strut_mesh, and the symmetrisation that makes it exact
    4 mirror maps   the index algebra, built once
    5 half and full the two directions, for points and for vectors
    6 the wake      the projection that keeps the image an image
    7 the structure the immersed solid, clamped at the waterline

This is a builders-only module, like fem_mesh: nothing in the solve path imports
it, and the maps it builds travel in the `vent` state as integer arrays, so
`vent_solve` needs no import of it.

Four decisions are worth stating before the code.

**The immersed half is authoritative, always.** The image is a mathematical
device, not a piece of fluid: everything about it is computed from the real half,
never solved for and never convected. `symmetrise_points` and `symmetrise_wake`
both write the image from the real half and never the other way.

**The symmetry is enforced, not inherited.** `thick_wing_mesh` with unit taper
and no sweep or twist IS the doubled strut, and the yaw rotation is about the
span axis and does not touch it, so almost everything is symmetric already. But
`np.linspace(-a, a, n+1)` is bit-exactly antisymmetric only when n is a power of
two - it is not at n = 12, 20 or 24 - so the last bits of the spanwise
coordinate must be overwritten from the real half. For a prismatic strut that
overwrite changes nothing but those bits, and it makes the property hold for any
station count and any spanwise spacing rather than for a lucky few.

**The waterline station is NOT a pinched tip, and `fem_mesh` had to be told.**
On the immersed half the far station is the open root at the waterline. The
vendored-style rib rebuild replaces the half-thickness at both ends with the
neighbour's, which for an untapered strut is bit-exact and therefore a latent
trap rather than a live bug; `foil_ribs` now takes which ends to rebuild.

**The wake cannot be mirrored by convecting it.** With phi antisymmetric the
perturbation velocity obeys v'(Rx) = -R v'(x) while the onset obeys v(Rx) =
+R v(x), so mirror-consistent advection would need v' = 0. Left to `convect`,
the image wake drifts off the mirror position and the free-surface condition is
lost silently and progressively over a march. Section 6 projects it instead, and
the drift it discards is a physically meaningful diagnostic - see the docstring
of `symmetrise_wake`.
"""

# ------------------------------------------------------------- 1 constants

import numpy as np

import thick_panel_wing as tpw
import make_thick_sample_inputs as mts
import fem_mesh

DEPTH_AXIS = 1                 # y. The span of a surface-piercing strut is
                               # vertical, so the spanwise axis IS the depth
                               # axis, and a spanwise strip is a depth station.
                               # This is what keeps every vendored spanwise
                               # routine valid and correctly interpreted:
                               # `spanwise_loading` returns the depth loading,
                               # and `integrate_loads`' CM about y is already the
                               # YAWING moment the paper reports.
REFLECT = np.array([1.0, -1.0, 1.0])       # R = diag(1, -1, 1), det = -1


# --------------------------------------------------------------- 3 the strut

def symmetrise_points(points, y_fs=0.0, n_half=None):
    """Overwrite the image half from the mirror of the immersed half. -> points

    The immersed half is authoritative. For a prismatic strut this changes only
    the last bits of the spanwise coordinate, and it makes exact mirror symmetry
    hold for any station count rather than only for a power of two.
    """
    points = np.array(points, dtype=float)
    nspan = points.shape[1] - 1
    if nspan % 2:
        raise ValueError(
            f"a doubled strut needs an even spanwise station count so one node "
            f"row lies on the free surface; got nspan={nspan}. With an odd count "
            f"a panel straddles the plane and has no exact mirror partner.")
    n_half = nspan // 2 if n_half is None else n_half
    for j in range(n_half):
        src = points[:, j, :]
        points[:, nspan - j, 0] = src[:, 0]
        points[:, nspan - j, 1] = 2.0 * y_fs - src[:, 1]
        points[:, nspan - j, 2] = src[:, 2]
    points[:, n_half, DEPTH_AXIS] = y_fs           # the waterline row, exactly
    return points


def symmetry_residual(points, y_fs=0.0):
    """How far the mesh is from mirror-symmetric about y = y_fs. -> float

    Zero, exactly, for anything `symmetrise_points` has touched. Reported rather
    than tolerated: it is the premise the whole image rests on.
    """
    points = np.asarray(points, dtype=float)
    nspan = points.shape[1] - 1
    mirrored = points[:, ::-1, :] * REFLECT
    mirrored[..., DEPTH_AXIS] += 2.0 * y_fs
    scale = max(float(np.abs(points).max()), 1e-300)
    del nspan
    return float(np.abs(points - mirrored).max() / scale)


def strut_mesh(n_c=8, nspan_half=8, h=1.0, chord=1.0, thickness=0.12,
               camber=0.0, alpha_deg=0.0, y_fs=0.0, image=True):
    """Wrapped surface of a surface-piercing strut. -> (2*n_c+1, nspan+1, 3)

    image=True (the default) returns the DOUBLED body: the strut immersed to
    depth h in y in [y_fs - h, y_fs], plus its mirror above the free surface, as
    one closed mesh with pinched far tips and a full-chord section at y = y_fs.
    nspan = 2*nspan_half.

    image=False returns the immersed strut alone as a closed foil of span h with
    both tips pinched, spanning y in [y_fs - h, y_fs]. There is then no free
    surface at all: the waterline behaves as a second tip, which forces the
    loading to zero there as the image does, but the chord vanishes there too.
    It is a cheap fixture and a fallback, not a model of a free surface.

    The strut is prismatic - unit taper, no sweep, no twist - and yawed by
    alpha_deg about the span axis, which for a vertical span is the incidence.
    Composed from the vendored builder and then symmetrised; no geometry is
    written here.
    """
    if h <= 0.0 or chord <= 0.0:
        raise ValueError(f"immersion and chord must be positive; got h={h}, "
                         f"chord={chord}")
    nspan = 2 * nspan_half if image else nspan_half
    points = mts.thick_wing_mesh(n_c=n_c, nspan=nspan,
                                 span=(2.0 * h if image else h),
                                 root_chord=chord, taper=1.0, sweep_deg=0.0,
                                 twist_deg=0.0, camber=camber,
                                 thickness=thickness)
    if image:
        points = symmetrise_points(points, y_fs=0.0)
        points[..., DEPTH_AXIS] += y_fs
    else:                                   # shift so the waterline is the top
        points[..., DEPTH_AXIS] += y_fs - 0.5 * h
    points = tpw.pitch_mesh(points, np.radians(alpha_deg))
    return tpw.validate_mesh(points)


# ------------------------------------------------------------ 4 mirror maps

def mirror_maps(shape_full, y_fs=0.0, image=True):
    """Index algebra of the reflection, built once. -> dict of int arrays

    node   (i, j) <-> (i, nspan - j)
    panel  (i, j) <-> (i, nspan - 1 - j)
    wake   node column j <-> nspan - j, strip j <-> nspan - 1 - j

    The immersed half is j <= nspan_half for nodes and j < nspan_half for panels.
    Because the reflection reverses the corner order AND has determinant -1, the
    mirrored panel's outward normal is exactly the reflection of the original's,
    so the doubled body is one closed body with consistent normals - which is why
    `validate_mesh` accepts it and why D[P(h), P(k)] = D[h, k].

    image=False builds the degenerate maps of a single immersed foil: every panel
    is real, and the mirror is the identity.
    """
    nwrap, nspan = shape_full[0] - 1, shape_full[1] - 1
    node = np.arange((nwrap + 1) * (nspan + 1)).reshape(nwrap + 1, nspan + 1)
    panel = np.arange(nwrap * nspan).reshape(nwrap, nspan)
    j_of = np.tile(np.arange(nspan), nwrap)
    if not image:
        return {"image": False, "y_fs": float(y_fs), "n_half": nspan,
                "waterline_j": nspan, "shape_full": tuple(shape_full),
                "shape_half": tuple(shape_full),
                "node_mirror": node.ravel().copy(),
                "panel_mirror": panel.ravel().copy(),
                "node_real": node.ravel().copy(),
                "real_mask": np.ones(nwrap * nspan, dtype=bool),
                "image_mask": np.zeros(nwrap * nspan, dtype=bool),
                "wake_col_mirror": np.arange(nspan + 1),
                "wake_strip_mirror": np.arange(nspan)}
    if nspan % 2:
        raise ValueError(f"a doubled strut needs an even nspan; got {nspan}")
    n_half = nspan // 2
    return {"image": True, "y_fs": float(y_fs), "n_half": n_half,
            "waterline_j": n_half, "shape_full": tuple(shape_full),
            "shape_half": (nwrap + 1, n_half + 1, 3),
            "node_mirror": node[:, ::-1].ravel().copy(),
            "panel_mirror": panel[:, ::-1].ravel().copy(),
            "node_real": node[:, :n_half + 1].ravel().copy(),
            "real_mask": j_of < n_half,
            "image_mask": j_of >= n_half,
            "wake_col_mirror": np.arange(nspan, -1, -1),
            "wake_strip_mirror": np.arange(nspan - 1, -1, -1)}


def mirror_scalar(field, maps, sign=-1.0):
    """Reflect a per-panel scalar onto its mirror partner. -> (N,)

    sign=-1 is the antisymmetric case, which is what the free-surface image
    needs: the image strength is MINUS the mirror of the real one. sign=+1 is the
    rigid-wall image, and the two differ in exactly this one sign.
    """
    return sign * np.asarray(field, dtype=float)[maps["panel_mirror"]]


# --------------------------------------------------------- 5 half and full

def half_points(points_full, maps):
    """The immersed half of a doubled points array. -> (nwrap+1, n_half+1, 3)"""
    return np.array(points_full[:, :maps["n_half"] + 1, :], dtype=float)


def complete_points(points_half, maps):
    """Mirror the immersed half up to the full doubled array. -> points_full

    The fixed affine map that slaves the image geometry to the deformed real
    half. It is linear in the half coordinates, which is what lets the transfer
    stay built once on the reference configurations and keeps the coupling
    residual a fixed function of the displacement.
    """
    points_half = np.asarray(points_half, dtype=float)
    if not maps["image"]:
        return np.array(points_half)
    nh = maps["n_half"]
    out = np.empty(maps["shape_full"], dtype=float)
    out[:, :nh + 1, :] = points_half
    for j in range(nh + 1, 2 * nh + 1):
        src = points_half[:, 2 * nh - j, :]
        out[:, j, 0] = src[:, 0]
        out[:, j, 1] = 2.0 * maps["y_fs"] - src[:, 1]
        out[:, j, 2] = src[:, 2]
    out[:, nh, DEPTH_AXIS] = maps["y_fs"]
    return out


def complete_vectors(v_half, maps):
    """Mirror a nodal vector field up to the full array. -> v_full

    A vector reflects with R = diag(1, -1, 1) and no offset, unlike a point.
    """
    v_half = np.asarray(v_half, dtype=float)
    if not maps["image"]:
        return np.array(v_half)
    nh = maps["n_half"]
    out = np.empty(maps["shape_full"], dtype=float)
    out[:, :nh + 1, :] = v_half
    for j in range(nh + 1, 2 * nh + 1):
        out[:, j, :] = v_half[:, 2 * nh - j, :] * REFLECT
    return out


def half_interface(points_half):
    """Interface data for a transfer built on the immersed half alone.

    The same three arrays `tpw.interface` returns, but for the half mesh and
    with a weld map that welds the trailing-edge seam and the DEEP tip pinch
    only. `build_topology` welds both end stations, which on the half mesh would
    weld the upper- and lower-surface waterline nodes to each other and collapse
    the open root. -> dict for fsi_transfer.build_transfer(iface=...)
    """
    points_half = np.asarray(points_half, dtype=float)
    nwrap, nspan_h = points_half.shape[0] - 1, points_half.shape[1] - 1
    node = np.arange((nwrap + 1) * (nspan_h + 1)).reshape(nwrap + 1, nspan_h + 1)
    weld = node.copy()
    weld[nwrap, :] = weld[0, :]                              # TE seam
    for i in range(nwrap // 2 + 1, nwrap + 1):               # the deep tip only
        weld[i, 0] = weld[nwrap - i, 0]
    j_of = np.tile(np.arange(nspan_h), nwrap)
    panel_nodes = np.stack([node[:-1, :-1].ravel(), node[1:, :-1].ravel(),
                            node[1:, 1:].ravel(), node[:-1, 1:].ravel()], axis=1)
    return {"nodes": points_half.reshape(-1, 3).copy(),
            "panel_nodes": panel_nodes, "weld_map": weld.ravel(),
            "tip_flag": j_of == 0, "shape": points_half.shape}


# ----------------------------------------------------------------- 6 the wake

def symmetrise_wake(wake, maps):
    """Project the wake onto the mirror-symmetric set. -> dict of diagnostics

    Called after every `shed`/`truncate` and inside any relaxation loop. The
    immersed half is authoritative: the image columns are overwritten from the
    mirror of the real ones, and the waterline column's depth coordinate is set
    to the plane exactly.

    Both operations have a physical reading, and the second is worth a
    diagnostic. On the plane, the antisymmetry of phi makes the streamwise and
    lift components of the perturbation velocity vanish while the depthwise
    component does not, so the perturbation velocity on the free surface is
    purely normal to it, and **the discarded drift is exactly the linearised wave
    elevation**. Report it, as the transfer reports its projection offset: it is
    the quantitative measure of how far below the model's valid Froude range a
    run is being pushed, and it is what the driver's low-Froude warning should
    key on rather than a rule of thumb.

    The doublet strengths are NOT projected. They are written from the body
    solution by `set_bound_strengths` and are antisymmetric already if the solve
    was; projecting them would hide the failure that `mu_residual` reports.
    """
    if wake is None or not maps["image"]:
        return {"drift_y": 0.0, "drift_max": 0.0, "mu_residual": 0.0}
    nodes = wake["nodes"]
    nh, y_fs = maps["n_half"], maps["y_fs"]
    drift = float(np.abs(nodes[:, nh, DEPTH_AXIS] - y_fs).max())
    nodes[:, nh, DEPTH_AXIS] = y_fs
    real = nodes[:, :nh, :]
    image = np.array(real[:, ::-1, :])
    image[..., DEPTH_AXIS] = 2.0 * y_fs - image[..., DEPTH_AXIS]
    before = float(np.abs(nodes[:, nh + 1:, :] - image).max())
    nodes[:, nh + 1:, :] = image
    mu = wake.get("mu")
    resid = 0.0
    if mu is not None and mu.size:
        scale = max(float(np.abs(mu).max()), 1e-300)
        resid = float(np.abs(mu + mu[:, maps["wake_strip_mirror"]]).max() / scale)
    return {"drift_y": drift, "drift_max": max(drift, before),
            "mu_residual": resid}


# ------------------------------------------------------------ 7 the structure

def strut_solid_mesh(points_half, n_thick=2, tip_fraction=1.0,
                     min_half_thickness=0.0):
    """Solid strut lofted across the immersed half. -> mesh dict

    The deep tip is a pinched wrap station and its half-thickness is rebuilt from
    its neighbour; the waterline station is the open root and keeps its own. Clamp
    `node_sets['tip_max']`, which for the immersed half IS the waterline.
    """
    return fem_mesh.solid_foil_mesh(points_half, n_thick=n_thick,
                                    tip_fraction=tip_fraction,
                                    min_half_thickness=min_half_thickness,
                                    pinched=(True, False))


# ------------------------------------------------------ 8 plotting and CLI

def main():
    """Build the doubled strut and report the properties the image rests on."""
    print("The surface-piercing strut: vent_mesh.py")
    for nspan_half in (4, 6, 8, 10, 12):
        points = strut_mesh(n_c=6, nspan_half=nspan_half, h=1.0, chord=1.0,
                            alpha_deg=8.0)
        maps = mirror_maps(points.shape)
        raw = mts.thick_wing_mesh(n_c=6, nspan=2 * nspan_half, span=2.0,
                                  root_chord=1.0, taper=1.0, sweep_deg=0.0,
                                  twist_deg=0.0, camber=0.0, thickness=0.12)
        half = half_points(points, maps)
        back = complete_points(half, maps)
        print(f"  nspan_half {nspan_half:3d}: panels {points.shape[0] - 1}"
              f"x{points.shape[1] - 1}   symmetry residual "
              f"{symmetry_residual(points):.1e}   as built by the vendored "
              f"builder {symmetry_residual(raw):.1e}   round trip "
              f"{np.abs(back - points).max():.1e}")
    print("  The vendored builder is symmetric to round-off but not always to")
    print("  the last bit: np.linspace(-a, a, n+1) is bit-exactly antisymmetric")
    print("  only when n is a power of two. Enforcing it costs nothing and makes")
    print("  the property hold for every station count.")

    points = strut_mesh(n_c=8, nspan_half=8, h=1.0, chord=1.0, alpha_deg=10.0)
    maps = mirror_maps(points.shape)
    pan = tpw.build_panels(points)
    y = pan["centroids"][:, DEPTH_AXIS]
    print(f"\n  immersed panels {int(maps['real_mask'].sum())}, image panels "
          f"{int(maps['image_mask'].sum())}, deepest y {y.min():+.3f}, "
          f"shallowest immersed y {y[maps['real_mask']].max():+.4f}")
    print(f"  enclosed volume {tpw.signed_volume(pan):+.6f} (must be positive), "
          f"wetted area {pan['areas'].sum():.4f}, immersed reference area "
          f"{0.5 * tpw.reference_area(points):.4f}")
    iface = half_interface(half_points(points, maps))
    print(f"  half interface: {iface['nodes'].shape[0]} nodes, "
          f"{len(np.unique(iface['weld_map']))} after welding the TE seam and "
          f"the deep tip only")
    mesh = strut_solid_mesh(half_points(points, maps))
    print(f"  immersed solid: {mesh['nodes'].shape[0]} nodes, "
          f"{mesh['elements'].shape[0]} elements, clamp set 'tip_max' has "
          f"{len(mesh['node_sets']['tip_max'])} nodes at the waterline")
    print("\nDone.")


if __name__ == "__main__":
    main()
