"""Pressure and loads on a ventilated, doubled surface-piercing strut.

The load half of the seam to the vendored solver. `vent_solve` owns the linear
system; this module owns what happens after it: the cavity pressure, the
resultants of the immersed half, and the path to structural nodal forces.

Five dependency-ordered sections, and the same three discipline rules: nothing
imports downward, every block is a pure function over explicit arguments, and
prompts, prints and figures appear only under main().

    1 constants     none of its own
    2 imports       numpy; thick_panel_wing, fsi_transfer, vent_section
    3 pressure      the cavity pressure, and the fractional blend
    4 resultants    the immersed half, on the immersed reference area
    5 structure     the load path, with the image half dropped
    6 loading       the depth loading, for comparison with the sectional model

Four decisions are worth stating before the code.

**p_ref must be None, and a caller passing it is an error rather than a
nuisance.** The cavity condition is stated relative to the LOCAL still-water
pressure, and the vendored `p_ref=None` path is already exactly that reference.
Passing a hydrostatic `p_ref` as well would count the hydrostatic field twice.
`p_ref` also only ever receives the z column of the centroids, which for a
vertical span is the lift axis and not depth, so it could not carry the field
even if it were wanted.

**The clip RAISES the pressure on the suction side.** A ventilated cavity is at
atmospheric pressure, which is above the suction pressure it replaces; that is
precisely why ventilation destroys lift. So the invariant is that the ventilated
pressure is never BELOW the wetted one, and the cavity is admitted only where the
margin Cp_cavity - Cp_wetted is positive. The opposite reading - that a cavity
lowers pressure - is a plausible-sounding inversion that would still reduce the
lift magnitude on some meshes, so `test_vent` asserts the direction with no
tolerance.

**Resultants come from the immersed half, on s_ref = h*c.** On the doubled mesh
the image half carries the negated loading, so `tpw.get_loads` returns a
near-cancellation - measured at about a tenth of the immersed CL, not zero,
because the pressure is quadratic in a velocity whose tangential part does not
simply mirror. `tpw.get_loads` must therefore never be called on a ventilating
state, and scaling its near-zero CL by two is a real bug wearing a plausible
face.

**The solved pressure and the pressure used for the load are both returned.**
With the Dirichlet closure the cavity pressure emerges from the solve, because
mu is prescribed there and the vendored surface gradient reproduces it; that
`cp_solved` equals -sigma_c on fully covered panels is an end-to-end test of the
dynamic condition, the mixed system and the pressure evaluation together, and it
would be a tautology if the load path overwrote it. The load uses `cp`, which
imposes the cavity pressure to the fractional degree of the closure panel so
that it is continuous in the cavity length; the two differ only on partially
covered panels.
"""

# ------------------------------------------------------------- 2 imports

import numpy as np

import thick_panel_wing as tpw
import fsi_transfer as fsx
import vent_cavity as vcv
import vent_section as vs


# ------------------------------------------------------------- 3 pressure

def cavity_cp(pan, vent):
    """The pressure coefficient inside the cavity, Cp = -sigma_c(z'). -> (N,)

    Referenced to the local still-water pressure, so this IS the cavity
    condition: for natural ventilation the cavity is at atmospheric pressure,
    dsigma = 0, and Cp_cavity = -(z'/h)(2/Fn_h^2) - zero at the waterline and
    increasingly negative with depth. A cavity is possible only where the wetted
    Cp lies BELOW this value.
    """
    depth = vcv.panel_depth(pan, vent["y_fs"])
    return -vs.sigma_cavity(depth, vent["h"], vent["fn_h"], vent["dsigma"])


def vent_pressure(pan, mu, u_rel, u_ref, vent, weight, dphi_dt=None, rho=1.0):
    """Pressure fields of a ventilated section. -> dict

    Keys: v_surf, cp_solved (what the solve produced, untouched), cp_cav (the
    cavity condition), cp (what the load uses), p_gauge, and raised (the
    per-panel pressure increase, which is never negative by construction).

    Note that cp_solved is NOT the wetted baseline once the cavity is in the
    system: with the Dirichlet closure the dynamic condition makes it equal the
    cavity pressure on the cavity. The baseline is a separate, wetted evaluation,
    and `solve_cavity` keeps it as cav['cp_baseline'] - the two were briefly
    aliased here, and the consequence was that the extent rule read the cavity's
    own imposed pressure, found a zero margin and collapsed the cavity every
    iteration.

    With closure='dirichlet' the solve already carries the cavity condition, so
    cp and cp_solved differ only on partially covered panels. With
    closure='clip' the solve is the wetted one and the cavity pressure is
    imposed here; that is auditable but not self-consistent, because the
    trailing-edge jump and hence the wake circulation stay wetted.
    """
    fields = tpw.pressure_fields(pan, mu, u_rel, u_ref, dphi_dt=dphi_dt, rho=rho)
    cp_solved = fields["cp"]
    cp_cav = cavity_cp(pan, vent)
    w = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    # The cavity pressure is a FLOOR, applied to the fractional degree of the
    # coverage: inside a cavity the pressure cannot fall below the cavity's own,
    # which is what a cavity is. Written as a floor rather than as a replacement so
    # that the ventilated pressure is never below the wetted one anywhere, by
    # construction and not by tolerance. The kink coincides with the closure point,
    # where the two pressures are equal and the cavity length is determined.
    cp = cp_solved + w * np.maximum(cp_cav - cp_solved, 0.0)
    p_gauge = 0.5 * rho * u_ref ** 2 * cp
    return {"v_surf": fields["v_surf"], "cp_solved": cp_solved,
            "cp_cav": cp_cav, "cp": cp, "p_gauge": p_gauge,
            "raised": cp - cp_solved}


def cavity_speed(pan, vent):
    """Total surface speed on the cavity, |V| = u sqrt(1 + sigma_c), (3.1). -> (N,)

    The dynamic condition: the pressure inside the cavity is known, so Bernoulli
    fixes the speed on its boundary. This is what `vent_solve` integrates to get
    the prescribed mu, and it is the whole content of the Dirichlet closure.
    """
    depth = vcv.panel_depth(pan, vent["y_fs"])
    sigma_c = vs.sigma_cavity(depth, vent["h"], vent["fn_h"], vent["dsigma"])
    return vent["u_ref"] * np.sqrt(np.maximum(1.0 + sigma_c, 0.0))


# ----------------------------------------------------------- 4 resultants

def immersed_loads(fluid, vent, p_gauge, rho=1.0):
    """Resultants of the immersed half alone, on s_ref = h*c. -> dict

    The image half's pressure is zeroed before integration, so nothing about it
    reaches the force, the moment or the structure. The moment reference is the
    one fixed once in `build_vent`, not the one `integrate_loads` would locate for
    itself: its own search runs over the mid-span section, which on a doubled mesh
    is the waterline, and it re-runs on every deflected geometry, so the yawing
    moment would drift.

    CL here is the side force of the paper's (3.3a) and CM about the span axis is
    its yawing moment (3.3c), because the span is vertical.
    """
    pan = fluid["panels"]
    p = np.array(p_gauge, dtype=float)
    p[np.asarray(vent["image_mask"], dtype=bool)] = 0.0
    res = tpw.integrate_loads(pan, p, rho, vent["u_ref"], vent["s_ref_wet"],
                              x_ref=vent["x_ref"])
    force = -p[:, None] * pan["areas"][:, None] * pan["normals"]
    return {"p_gauge": p, "force": force, "dp": p,
            "areas": pan["areas"].copy(), "normals": pan["normals"].copy(),
            "centroids": pan["centroids"].copy(),
            "corners": pan["corners"].copy(),
            "tip_flag": pan["tip_flag"].copy(),
            "quality": pan["warp"].copy(), "resultants": res}


# ------------------------------------------------------------ 5 structure

def structural_forces_half(transfer, vent, pan, force_panels):
    """Panel forces on the doubled mesh -> structural nodal forces. -> (Ns, 3)

    The image half's panel forces are already zero, so the vendored conservative
    lumping puts force only on real-half nodes and the waterline nodes receive
    only real contributions; force conservation therefore stays at round-off and
    `conservation_report` still means what it says. The real-half rows are then
    gathered for the transfer, which was built on the immersed half.
    """
    nodal_full = fsx.panel_forces_to_nodes(pan, force_panels)
    if not vent["image"]:
        return fsx.to_structure(transfer, nodal_full)
    return fsx.to_structure(transfer, nodal_full[np.asarray(vent["node_real"])])


def conservation_report_half(transfer, vent, pan, force_panels, u_struct=None):
    """`fsi_transfer.conservation_report` for the immersed half. -> dict"""
    force = np.array(force_panels, dtype=float).reshape(-1, 3)
    force[np.asarray(vent["image_mask"], dtype=bool)] = 0.0
    nodal_full = fsx.panel_forces_to_nodes(pan, force)
    nodal = nodal_full if not vent["image"] \
        else nodal_full[np.asarray(vent["node_real"])]
    f_struct = fsx.to_structure(transfer, nodal)
    x_f = pan["points"].reshape(-1, 3)
    if vent["image"]:
        x_f = x_f[np.asarray(vent["node_real"])]
    scale = max(float(np.abs(force).sum()), 1e-300)
    lever = max(float(np.linalg.norm(np.ptp(x_f, axis=0))), 1e-300)
    out = {"force_panels": force.sum(axis=0), "force_lumped": nodal.sum(axis=0),
           "force_struct": f_struct.sum(axis=0),
           "moment_panels": np.cross(pan["centroids"], force).sum(axis=0),
           "moment_lumped": np.cross(x_f, nodal).sum(axis=0),
           "offset_max": transfer["offset_max"]}
    out["force_error_lump"] = float(np.abs(out["force_lumped"]
                                           - out["force_panels"]).max() / scale)
    out["force_error_transfer"] = float(np.abs(out["force_struct"]
                                               - out["force_lumped"]).max() / scale)
    out["moment_error_lump"] = float(np.abs(out["moment_lumped"]
                                            - out["moment_panels"]).max()
                                     / (scale * lever))
    if u_struct is not None:
        u_s = np.asarray(u_struct, dtype=float).reshape(-1, 3)
        u_f = fsx.to_fluid(transfer, u_s)
        work_f = float(np.einsum("mc,mc->", nodal, u_f))
        work_s = float(np.einsum("mc,mc->", f_struct, u_s))
        out["work_fluid"], out["work_struct"] = work_f, work_s
        out["work_error"] = abs(work_f - work_s) / max(abs(work_f), 1e-300)
    return out


# -------------------------------------------------------------- 6 loading

def depth_loading(pan, vent, p_gauge, rho=1.0):
    """Sectional lift coefficient against depth below the free surface.

    The vendored `spanwise_loading` with the image half zeroed and the spanwise
    coordinate mapped to depth, so the result is directly comparable with the
    elliptic shape (4.7) and with the lifting line. Ordered from the waterline
    downward. -> (depth (n_half,), cl (n_half,))
    """
    p = np.array(p_gauge, dtype=float)
    p[np.asarray(vent["image_mask"], dtype=bool)] = 0.0
    y_strip, cl = tpw.spanwise_loading(pan, p, vent["u_ref"], rho=rho)
    keep = np.asarray(vent["real_mask"], dtype=bool).reshape(
        pan["nwrap"], pan["nspan"])[0]
    depth = vent["y_fs"] - y_strip[keep]
    order = np.argsort(depth)
    return depth[order], cl[keep][order]
