"""Generate sample wrapped-surface meshes for thick_panel_wing.py.

Creates:
- naca0012_rect_mesh.npz      rectangular NACA 0012 wing, AR 8
- thick_wing_mesh.npz         tapered, swept, twisted, cambered wing, 12% thick
- naca0012_quasi2d_mesh.npz   AR 40 NACA 0012 wing for two-dimensional checks
- uniform_profile.csv         uniform onset, U = 1
- velocity_profile.csv        linear shear, U = 1 + 0.3 z

Mesh format 'thick-wrap-1': key 'points' of shape (nwrap+1, nspan+1, 3), index
i wrapping from the lower-surface trailing edge (i = 0) forward under the lower
surface, around the leading edge, back along the upper surface to the trailing
edge (i = nwrap); index j across the span. The trailing edge is sharp and closed
(points[0, j] == points[nwrap, j]) and both tip stations are pinched onto the
section camber line (points[i, jt] == points[nwrap - i, jt]).

The mesh-building functions are pure; files are written only under main().
"""

import numpy as np


def naca4_thickness(xi, thickness):
    """Closed-TE NACA four-digit half-thickness y_f/c at chordwise stations xi.

    The last coefficient is -0.1036 (not the standard -0.1015) so the five
    coefficients sum to zero and the trailing edge closes sharply.
    """
    xi = np.asarray(xi, dtype=float)
    return 5.0 * thickness * (0.2969 * np.sqrt(np.maximum(xi, 0.0))
                              - 0.1260 * xi - 0.3516 * xi ** 2
                              + 0.2843 * xi ** 3 - 0.1036 * xi ** 4)


def wrap_stations(n_c):
    """Cosine-clustered wrap coordinate xi_i, i = 0..2*n_c (TE -> LE -> TE)."""
    i = np.arange(2 * n_c + 1)
    return 0.5 * (1.0 + np.cos(np.pi * i / n_c))


def thick_section(n_c, camber, thickness, pinched=False):
    """Unit-chord section wrap: (x, z), each (2*n_c + 1,), TE -> LE -> TE.

    Thickness is added perpendicular to the parabolic camber line; the camber
    line and slope match generic_wing_mesh in make_sample_inputs.py. A pinched
    section (tips) collapses onto the camber line.
    """
    xi = wrap_stations(n_c)
    n_wrap = 2 * n_c
    yf = np.zeros_like(xi) if pinched else naca4_thickness(xi, thickness)
    zc = 4.0 * camber * xi * (1.0 - xi)
    th = np.arctan(4.0 * camber * (1.0 - 2.0 * xi))
    sgn = np.where(np.arange(n_wrap + 1) <= n_c, -1.0, 1.0)  # lower, then upper
    x = xi - sgn * yf * np.sin(th)
    z = zc + sgn * yf * np.cos(th)
    x[-1], z[-1] = x[0], z[0]                    # exact sharp closed TE
    if pinched:                                  # exact mirror about the LE
        x[n_c + 1:] = x[n_c - 1::-1]
        z[n_c + 1:] = z[n_c - 1::-1]
    return x, z


def thick_wing_mesh(n_c=24, nspan=20, span=8.0, root_chord=4.0 / 3.0, taper=0.5,
                    sweep_deg=15.0, twist_deg=-3.0, camber=0.02, thickness=0.12,
                    cosine_span=False):
    """Wrapped surface of a tapered, swept, linearly twisted, cambered wing.

    Twist rotates each section about its quarter-chord point; sweep shifts the
    leading edge by tan(sweep) * |y|; thickness scales with the local chord.
    Returns points of shape (2*n_c + 1, nspan + 1, 3).
    """
    nwrap = 2 * n_c
    if cosine_span:
        y_stations = -0.5 * span * np.cos(np.pi * np.arange(nspan + 1) / nspan)
    else:
        y_stations = np.linspace(-span / 2.0, span / 2.0, nspan + 1)
    points = np.zeros((nwrap + 1, nspan + 1, 3))
    tan_sweep = np.tan(np.radians(sweep_deg))
    for j, y in enumerate(y_stations):
        eta = abs(2.0 * y / span)
        chord = root_chord * (1.0 + (taper - 1.0) * eta)
        x_le = tan_sweep * abs(y)
        theta = np.radians(twist_deg) * eta
        xs, zs = thick_section(n_c, camber, thickness, pinched=j in (0, nspan))
        x_sec = (xs - 0.25) * chord
        z_sec = zs * chord
        x_rot = x_sec * np.cos(theta) + z_sec * np.sin(theta)
        z_rot = -x_sec * np.sin(theta) + z_sec * np.cos(theta)
        points[:, j, 0] = x_le + 0.25 * chord + x_rot
        points[:, j, 1] = y
        points[:, j, 2] = z_rot
    points[-1, :, :] = points[0, :, :]           # bit-exact TE closure
    for jt in (0, nspan):                        # bit-exact tip pinch
        for i in range(n_c + 1):
            points[nwrap - i, jt] = points[i, jt]
    return points


def save_mesh(name, points):
    np.savez(name, points=points, mesh_format=np.array("thick-wrap-1"))


def main():
    save_mesh("naca0012_rect_mesh.npz",
              thick_wing_mesh(n_c=24, nspan=20, span=8.0, root_chord=1.0,
                              taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                              camber=0.0, thickness=0.12))
    print("wrote naca0012_rect_mesh.npz (rectangular NACA 0012, AR 8, "
          "48 x 20 panels)")

    save_mesh("thick_wing_mesh.npz", thick_wing_mesh())
    print("wrote thick_wing_mesh.npz (taper 0.5, sweep 15 deg, twist -3 deg, "
          "camber 2%, thickness 12%, 48 x 20 panels)")

    save_mesh("naca0012_quasi2d_mesh.npz",
              thick_wing_mesh(n_c=32, nspan=12, span=40.0, root_chord=1.0,
                              taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                              camber=0.0, thickness=0.12, cosine_span=True))
    print("wrote naca0012_quasi2d_mesh.npz (rectangular NACA 0012, AR 40, "
          "64 x 12 panels, cosine span)")

    np.savetxt("uniform_profile.csv",
               np.column_stack([[-10.0, 10.0], [1.0, 1.0]]),
               delimiter=",", header="z, U", comments="# ")
    print("wrote uniform_profile.csv (U = 1)")

    z = np.linspace(-4.0, 4.0, 17)
    np.savetxt("velocity_profile.csv",
               np.column_stack([z, np.maximum(1.0 + 0.3 * z, 0.1)]),
               delimiter=",", header="z, U", comments="# ")
    print("wrote velocity_profile.csv (linear shear U = 1 + 0.3 z)")


if __name__ == "__main__":
    main()
