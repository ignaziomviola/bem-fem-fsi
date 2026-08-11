"""Structural meshes for fem_solid.py: boxes, plates and solid foils.

The builders are pure and return a dict; files are written only under main(),
the pattern make_thick_sample_inputs.py already follows on the fluid side.

    mesh = {"nodes": (Ns, 3), "elements": (Ne, 8), "node_sets": {name: ids}}

Two things about the foil builder are worth reading before it is used.

The wrap mesh the panel method reads pairs chordwise station i with station
nwrap - i: both lie at the same chordwise position by construction of
`make_thick_sample_inputs.thick_section`, one on the lower surface and one on
the upper. The segment between them is the section's vertical extent there, so
lofting across it gives a solid foil whose outer surface IS the wetted surface,
and the load transfer then has almost nothing to interpolate. The pairing
degenerates at both ends - the vertical extent of a section at its leading and
trailing edge is zero - so the first and last chordwise columns are wedges,
which is the geometry and not an artefact. Their Jacobians are checked at the
Gauss points like any other element's.

The fluid mesh pinches both tip stations onto the camber line, because the
aerodynamic body has to close. A structure lofted from those stations would
have zero volume there. The tip sections are therefore rebuilt with the
thickness of their inboard neighbour, so the structural surface departs from
the wetted surface over the outermost spanwise strip at each tip and nowhere
else. That is a genuine non-matching region, it is reported by the transfer's
offset diagnostic rather than hidden, and it is why the transfer operator is
built for non-matching meshes even though the rest of the two surfaces
coincide.
"""

import numpy as np


def _grid_nodes(x_s, y_s, z_s):
    """Structured node table over three coordinate vectors, index (i, j, k)."""
    xx, yy, zz = np.meshgrid(x_s, y_s, z_s, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _grid_elements(n_x, n_y, n_z):
    """Hex connectivity of a structured (n_x, n_y, n_z) element grid.

    Node ordering follows fem_solid.NODE_XI: the k face first, anticlockwise
    about +k, then the k+1 face in the same order, which makes the Jacobian
    positive for a right-handed (i, j, k) grid.
    """
    idx = np.arange((n_x + 1) * (n_y + 1) * (n_z + 1))
    idx = idx.reshape(n_x + 1, n_y + 1, n_z + 1)
    i, j, k = np.meshgrid(np.arange(n_x), np.arange(n_y), np.arange(n_z),
                          indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    return np.stack([idx[i, j, k], idx[i + 1, j, k],
                     idx[i + 1, j + 1, k], idx[i, j + 1, k],
                     idx[i, j, k + 1], idx[i + 1, j, k + 1],
                     idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]], axis=1)


def box_mesh(lx, ly, lz, nx, ny, nz, origin=(0.0, 0.0, 0.0)):
    """Structured box. The substrate of every analytical gate.

    node_sets: x_min, x_max, y_min, y_max, z_min, z_max (the six faces) and
    'all'. Element (i, j, k) is at flat index (i*ny + j)*nz + k.
    """
    x_s = origin[0] + np.linspace(0.0, lx, nx + 1)
    y_s = origin[1] + np.linspace(0.0, ly, ny + 1)
    z_s = origin[2] + np.linspace(0.0, lz, nz + 1)
    nodes = _grid_nodes(x_s, y_s, z_s)
    elements = _grid_elements(nx, ny, nz)
    sets = {}
    for axis, name, target in ((0, "x", (x_s[0], x_s[-1])),
                               (1, "y", (y_s[0], y_s[-1])),
                               (2, "z", (z_s[0], z_s[-1]))):
        tol = 1e-9 * max(lx, ly, lz)
        sets[f"{name}_min"] = np.where(np.abs(nodes[:, axis] - target[0]) < tol)[0]
        sets[f"{name}_max"] = np.where(np.abs(nodes[:, axis] - target[1]) < tol)[0]
    sets["all"] = np.arange(len(nodes))
    return {"nodes": nodes, "elements": elements, "node_sets": sets,
            "shape": (nx, ny, nz)}


def distort_interior(mesh, amplitude=0.2, seed=0):
    """Randomly displace the interior nodes of a box mesh. The patch test needs it.

    A patch test on a regular grid passes for elements that are wrong, because
    every Jacobian is constant; the distorted mesh is what makes the test
    discriminating. Boundary nodes stay put so the prescribed linear field can
    still be applied on them exactly.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in mesh.items()}
    nodes = out["nodes"] = mesh["nodes"].copy()
    n_x, n_y, n_z = mesh["shape"]
    spacing = np.array([np.ptp(nodes[:, c]) / max(n, 1)
                        for c, n in zip(range(3), (n_x, n_y, n_z))])
    grid = np.arange(len(nodes)).reshape(n_x + 1, n_y + 1, n_z + 1)
    inner = grid[1:-1, 1:-1, 1:-1].ravel()
    rng = np.random.default_rng(seed)
    nodes[inner] += amplitude * spacing * (rng.random((len(inner), 3)) - 0.5)
    return out


def plate_wing_mesh(chord, span, thickness, n_chord, n_span, n_thick=2,
                    x_offset=-0.25):
    """Flat rectangular plate wing in the fluid frame: x chord, y span, z up.

    x_offset places the leading edge, in chords, so the default puts the
    quarter-chord at the origin and matches make_thick_sample_inputs, whose
    sections are built about their own quarter chord.

    node_sets adds 'root' (the midspan station), 'tip_min', 'tip_max',
    'leading' and 'trailing'.
    """
    mesh = box_mesh(chord, span, thickness, n_chord, n_span, n_thick,
                    origin=(x_offset * chord, -0.5 * span, -0.5 * thickness))
    nodes = mesh["nodes"]
    tol = 1e-9 * max(chord, span)
    sets = mesh["node_sets"]
    sets["root"] = np.where(np.abs(nodes[:, 1]) < tol)[0]
    sets["tip_min"], sets["tip_max"] = sets["y_min"], sets["y_max"]
    sets["leading"], sets["trailing"] = sets["x_min"], sets["x_max"]
    return mesh


def foil_ribs(points, tip_fraction=1.0, min_half_thickness=0.0):
    """Lower and upper surface paired by chordwise station, tips rebuilt.

    Returns (lower, upper), both (n_c+1, nspan+1, 3), ordered from the LEADING
    edge (index 0) to the trailing edge, which is the order that makes the
    element Jacobian positive with x chordwise, y spanwise and z through the
    thickness.

    The tip stations of a wrap mesh are pinched, so their half-thickness is
    zero; it is replaced by tip_fraction times that of the neighbouring
    station. min_half_thickness, in units of the local chord, blunts the
    leading and trailing edge columns if their wedge elements are too ill
    conditioned for the material at hand; the default keeps the true geometry.
    """
    pts = np.asarray(points, dtype=float)
    nwrap = pts.shape[0] - 1
    nspan = pts.shape[1] - 1
    if nwrap % 2:
        raise ValueError(f"a wrap mesh has an even nwrap; got {nwrap}")
    n_c = nwrap // 2
    wrap = n_c - np.arange(n_c + 1)                    # LE first, TE last
    lower = pts[wrap]                                  # (n_c+1, nspan+1, 3)
    upper = pts[nwrap - wrap]
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    half[:, 0] = tip_fraction * half[:, 1]
    half[:, nspan] = tip_fraction * half[:, nspan - 1]
    if min_half_thickness > 0.0:
        chord = np.ptp(pts[..., 0], axis=0).max()
        floor = min_half_thickness * chord
        norm = np.linalg.norm(half, axis=2, keepdims=True)
        direction = np.where(norm > 1e-30, half / np.maximum(norm, 1e-30),
                             np.array([0.0, 0.0, 1.0]))
        half = np.where(norm < floor, floor * direction, half)
    return mid - half, mid + half


def solid_foil_mesh(points, n_thick=2, tip_fraction=1.0,
                    min_half_thickness=0.0):
    """Solid foil lofted across the paired wrap stations of a fluid mesh.

    Chordwise resolution is inherited from the wrap mesh (n_c = nwrap/2 element
    columns), spanwise likewise, and n_thick sets the layers through the
    thickness. Ribs whose two surfaces coincide - the leading and trailing edge
    - collapse to a single node per layer, so the end columns are wedges and
    carry no duplicated free nodes.

    node_sets: 'surface_lower', 'surface_upper', 'root' (midspan), 'tip_min',
    'tip_max', 'leading', 'trailing', 'all'.
    """
    lower, upper = foil_ribs(points, tip_fraction, min_half_thickness)
    n_r, n_j = lower.shape[0], lower.shape[1]
    n_k = int(n_thick)
    scale = np.linalg.norm(np.ptp(np.asarray(points).reshape(-1, 3), axis=0))
    collapsed = np.linalg.norm(upper - lower, axis=2) < 1e-12 * scale

    node_id = -np.ones((n_r, n_k + 1, n_j), dtype=np.int64)
    coords = []
    for r in range(n_r):
        for j in range(n_j):
            if collapsed[r, j]:
                node_id[r, :, j] = len(coords)
                coords.append(lower[r, j])
                continue
            for k in range(n_k + 1):
                node_id[r, k, j] = len(coords)
                coords.append(lower[r, j] + (k / n_k) * (upper[r, j] - lower[r, j]))
    nodes = np.array(coords)

    elements = []
    for r in range(n_r - 1):
        for j in range(n_j - 1):
            for k in range(n_k):
                elements.append([node_id[r, k, j], node_id[r + 1, k, j],
                                 node_id[r + 1, k, j + 1], node_id[r, k, j + 1],
                                 node_id[r, k + 1, j], node_id[r + 1, k + 1, j],
                                 node_id[r + 1, k + 1, j + 1],
                                 node_id[r, k + 1, j + 1]])
    elements = np.array(elements, dtype=np.int64)

    j_root = (n_j - 1) // 2
    sets = {"surface_lower": np.unique(node_id[:, 0, :]),
            "surface_upper": np.unique(node_id[:, n_k, :]),
            "root": np.unique(node_id[:, :, j_root]),
            "tip_min": np.unique(node_id[:, :, 0]),
            "tip_max": np.unique(node_id[:, :, n_j - 1]),
            "leading": np.unique(node_id[0, :, :]),
            "trailing": np.unique(node_id[n_r - 1, :, :]),
            "all": np.arange(len(nodes))}
    return {"nodes": nodes, "elements": elements, "node_sets": sets,
            "node_id": node_id, "shape": (n_r - 1, n_j - 1, n_k)}


def merge_meshes(*meshes):
    """Concatenate meshes, offsetting connectivity and node sets."""
    nodes, elements, sets = [], [], {}
    offset = 0
    for n, mesh in enumerate(meshes):
        nodes.append(mesh["nodes"])
        elements.append(mesh["elements"] + offset)
        for name, ids in mesh["node_sets"].items():
            sets[f"{n}:{name}"] = np.asarray(ids) + offset
        offset += len(mesh["nodes"])
    return {"nodes": np.concatenate(nodes), "elements": np.concatenate(elements),
            "node_sets": sets}


def main():
    """Report the meshes this module builds, and check them through fem_solid."""
    import fem_solid
    import make_thick_sample_inputs as mts

    box = box_mesh(1.0, 0.1, 0.02, 16, 2, 2)
    print(f"box_mesh          {len(box['nodes']):5d} nodes, "
          f"{len(box['elements']):5d} elements")

    plate = plate_wing_mesh(1.0, 8.0, 0.06, 12, 16, 2)
    print(f"plate_wing_mesh   {len(plate['nodes']):5d} nodes, "
          f"{len(plate['elements']):5d} elements")

    wrap = mts.thick_wing_mesh(n_c=12, nspan=12, span=8.0, root_chord=1.0,
                               taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                               camber=0.0, thickness=0.12)
    foil = solid_foil_mesh(wrap, n_thick=2)
    print(f"solid_foil_mesh   {len(foil['nodes']):5d} nodes, "
          f"{len(foil['elements']):5d} elements")

    for name, mesh in (("box", box), ("plate", plate), ("foil", foil)):
        cache = fem_solid.element_arrays(mesh["nodes"], mesh["elements"])
        print(f"  {name:6s} volume {cache['volume'].sum():.6f}, "
              f"min element volume {cache['volume'].min():.3e}, "
              f"surface faces {len(fem_solid.surface_faces(mesh['elements'])[0])}")


if __name__ == "__main__":
    main()
