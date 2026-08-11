"""Shared free-wake kernels and input helpers.

These seven functions are copied verbatim from `panel_wing.py` in the companion
repository https://github.com/ignaziomviola/free-wake-lifting-surface, which
holds the thin lifting-surface codes this one extends. They are duplicated
rather than imported so that a fresh clone of this repository runs on its own,
and they are collected here rather than inlined so that the duplication is
visible: any change to the Biot-Savart kernel or the regularisation must be
made in both places, and the doublet-ring unit check in
test_thick_panel_wing.py is what detects a divergence.

Unlike `panel_wing.py` this module imports numpy only, so importing the solver
creates no figure backend and headless runs need no MPLBACKEND setting.
"""

import numpy as np


def segment_velocities(points, p1, p2, core):
    """Unit-strength velocity induced by S straight vortex segments at M points.

    Returns an (M, S, 3) array. `core` is a per-segment absolute core radius
    (van Garrel regularisation) so the velocity vanishes smoothly on the axis.
    """
    r1 = points[:, None, :] - p1[None, :, :]
    r2 = points[:, None, :] - p2[None, :, :]
    r0 = p2 - p1
    cr = np.cross(r1, r2)
    cr2 = np.sum(cr * cr, axis=2)
    r0n2 = np.sum(r0 * r0, axis=1)[None, :]
    r1n = np.sqrt(np.sum(r1 * r1, axis=2))
    r2n = np.sqrt(np.sum(r2 * r2, axis=2))
    dot1 = np.sum(r0[None, :, :] * r1, axis=2)
    dot2 = np.sum(r0[None, :, :] * r2, axis=2)
    denom = 4.0 * np.pi * (cr2 + (core[None, :] ** 2) * r0n2)
    k = (dot1 / np.maximum(r1n, 1e-12) - dot2 / np.maximum(r2n, 1e-12))
    k /= np.maximum(denom, 1e-30)
    return k[:, :, None] * cr


def induced_velocity(points, p1, p2, strengths, core, chunk=300):
    """Total induced velocity at M points, chunked to bound memory use."""
    v = np.zeros_like(points)
    for s in range(0, len(points), chunk):
        block = segment_velocities(points[s:s + chunk], p1, p2, core)
        v[s:s + chunk] = np.einsum("msk,s->mk", block, strengths)
    return v


def load_velocity_profile(path):
    """Read a two-column 'z, U' CSV file; returns the sorted columns."""
    data = np.genfromtxt(path, delimiter=",", comments="#")
    data = np.atleast_2d(data)
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 1 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain rows of 'z, U' values")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def make_onset(z_table, u_table):
    """Onset-velocity function: (M,3) points -> (M,3) velocities."""
    def onset(points):
        pts = np.atleast_2d(points)
        v = np.zeros((len(pts), 3))
        v[:, 0] = np.interp(pts[:, 2], z_table, u_table)
        return v
    return onset


def pitch_mesh(points, alpha_rad):
    """Rotate the mesh nose-up by alpha about the y axis (through the origin)."""
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    out = points.copy()
    out[..., 0] = c * points[..., 0] + s * points[..., 2]
    out[..., 2] = -s * points[..., 0] + c * points[..., 2]
    return out


def initial_nodes(start, waypoints, steps, onset):
    """Initial filament: along the given surface path, then along the onset flow."""
    nodes = [start.copy()]
    p = start.copy()
    path = [w.copy() for w in waypoints]
    for length in steps:
        remaining = length
        while path and remaining > 1e-14:
            d = path[0] - p
            dist = np.linalg.norm(d)
            if dist > remaining:
                p = p + d * (remaining / dist)
                remaining = 0.0
            else:
                p = path.pop(0)
                remaining -= dist
        if remaining > 1e-14:
            v = onset(p[None, :])[0]
            p = p + v / max(np.linalg.norm(v), 1e-12) * remaining
        nodes.append(p.copy())
    return np.array(nodes)


def prompt(text, default):
    """Prompt on standard input, returning `default` when the reply is empty."""
    reply = input(f"{text} [{default}]: ").strip()
    return reply if reply else default
