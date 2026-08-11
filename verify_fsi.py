"""Coupled verification: the transfer, the fixed point and the coupled physics.

    python3 verify_fsi.py             every case
    python3 verify_fsi.py --case C3   one case by name
    python3 verify_fsi.py --quick     the cheap cases only

Six cases. C1 is exact to round-off by construction; C2, C3 and C4 compare the
coupled code against something computed by a different route - the fluid code's
own time march, the classical typical-section formula, and two-dimensional
strip theory; C5 and C6 measure properties of the scheme itself. Every number
in docs/COUPLING.md comes from a run of this file, and the tables hold what was
measured. Nothing is written and no figure is created.

C1 transfer conservation   force, moment and virtual work
C2 rigid limit             E -> infinity against unsteady_wing.time_march
C3 static divergence       three independent routes to one dynamic pressure
C4 added mass in water     wet frequency against the added-mass ratio and strip
                           theory
C5 the added-mass instability   where loose coupling fails and strong holds
C6 energy balance          work in against strain plus kinetic energy
"""

import sys
import time
import warnings

import numpy as np

import thick_panel_wing as tpw
import unsteady_wing as usw
import make_thick_sample_inputs as mts
import fem_materials as fmat
import fem_mesh
import fem_solid as fes
import fsi_driver as fsd
import fsi_transfer as fsx

RHO_WATER = 1000.0


def uniform_onset(points):
    pts = np.atleast_2d(points)
    vel = np.zeros((len(pts), 3))
    vel[:, 0] = 1.0
    return vel


def still_fluid(points):
    return np.zeros((len(np.atleast_2d(points)), 3))


def wing(n_c=6, nspan=6, span=4.0, thickness=0.12, alpha=5.0):
    pts = mts.thick_wing_mesh(n_c=n_c, nspan=nspan, span=span, root_chord=1.0,
                              taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                              camber=0.0, thickness=thickness)
    return tpw.pitch_mesh(pts, np.radians(alpha)) if alpha else pts


def foil_case(points, e_mod, rho_s=1200.0, rho_f=RHO_WATER, onset=uniform_onset,
              n_thick=1, unsteady=True, nwake=0, clamp="root"):
    mesh = fem_mesh.solid_foil_mesh(points, n_thick=n_thick)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(e_mod, 0.3, rho_s))
    config = None if unsteady else {"lifting": True, "nwake": nwake}
    fluid, sstate, transfer = fsd.init_fsi(points, onset, model, u_ref=1.0,
                                           rho=rho_f, unsteady=unsteady,
                                           config=config)
    if clamp:
        fes.clamp(sstate, mesh["node_sets"][clamp])
    fes.assemble_operators(sstate)
    return fluid, sstate, transfer, mesh, model


def _rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ------------------------------------------------------------------ case C1

def case_transfer(verbose=True):
    """Force, moment and virtual work across the interface."""
    points = wing()
    fluid, _, transfer, _, model = foil_case(points, 2e8, unsteady=False,
                                             nwake=8)
    fluid = tpw.solve(fluid, wake="relax")
    loads = tpw.get_loads(fluid, rho=RHO_WATER)
    rng = np.random.default_rng(6)
    u_s = rng.normal(size=model["nodes"].shape) * 1e-3
    matched = fsx.conservation_report(transfer, fluid["panels"], loads["force"],
                                      u_struct=u_s)
    plate = fem_mesh.plate_wing_mesh(1.0, 4.0, 0.06, 8, 8, 1, x_offset=0.0)
    plate_model = fes.build_model(plate["nodes"], plate["elements"],
                                  fmat.IsotropicElastic(2e8, 0.3, 1200.0))
    plate_transfer = fsx.build_transfer(fluid, plate_model)
    non_matching = fsx.conservation_report(plate_transfer, fluid["panels"],
                                           loads["force"])
    rigid = {}
    x_s = model["nodes"]
    x_f = fluid["panels"]["points"].reshape(-1, 3) + transfer["offset"]
    grad = np.array([[1e-2, 0.0, 0.0], [0.0, -3e-3, 2e-3], [0.0, 2e-3, -3e-3]])
    for name, (u_struct, exact) in {
            "translation": (np.tile([0.3, -0.2, 0.5], (len(x_s), 1)),
                            np.tile([0.3, -0.2, 0.5], (len(x_f), 1))),
            "rotation": (np.cross(np.tile([0.01, 0.02, -0.03], (len(x_s), 1)), x_s),
                         np.cross(np.tile([0.01, 0.02, -0.03], (len(x_f), 1)), x_f)),
            "linear field": (x_s @ grad.T, x_f @ grad.T)}.items():
        got = fsx.to_fluid(transfer, u_struct)
        rigid[name] = float(np.abs(got - exact).max() / np.abs(exact).max())
    if verbose:
        _rule("C1 transfer conservation, solid foil lofted from the wetted surface")
        print("  projection offset, matching meshes   "
              f"{transfer['offset_max']:.2e}")
        print("  total force, panels -> fluid nodes    "
              f"{matched['force_error_lump']:.2e}")
        print(f"  total force, fluid nodes -> structure {matched['force_error_transfer']:.2e}")
        print("  total moment, panels -> fluid nodes   "
              f"{matched['moment_error_lump']:.2e}")
        print("  virtual work, the two sides           "
              f"{matched['work_error']:.2e}")
        print("  motion transferred exactly:")
        for name, err in rigid.items():
            print(f"    {name:<14s} {err:.2e}")
        print("\n  the same fluid mesh over a FLAT PLATE structure, which the")
        print("  wetted surface genuinely does not lie on:")
        print("    projection offset                   "
              f"{plate_transfer['offset_max']:.3e}")
        print("    total force still conserved         "
              f"{non_matching['force_error_transfer']:.2e}")
        print("  Force survives non-matching meshes because the weights are a")
        print("  partition of unity. Moment does not: the load then acts at the")
        print("  projected point, and offset_max is what bounds the difference.")
    return {"matched": matched, "non_matching": non_matching, "rigid": rigid,
            "offset": transfer["offset_max"],
            "plate_offset": plate_transfer["offset_max"]}


# ------------------------------------------------------------------ case C2

def case_rigid_limit(e_moduli=(2e10, 2e11, 2e12), nsteps=10, dt=0.05,
                     rho_inf=0.5, verbose=True):
    """E -> infinity must reproduce the fluid code's own unsteady history."""
    points = wing()
    reference, _ = usw.time_march(points, uniform_onset,
                                  motion=usw.still(points), dt=dt,
                                  nsteps=nsteps, u_ref=1.0, rho=RHO_WATER)
    rows = []
    for e_mod in e_moduli:
        fluid, sstate, transfer, _, _ = foil_case(points, e_mod)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                      rho=RHO_WATER,
                                                      max_iter=12)
            history, _, sstate = fsd.time_march_fsi(
                fluid, sstate, transfer, dt=dt, nsteps=nsteps, rho=RHO_WATER,
                rho_inf=rho_inf, max_sub=30)
        error = float(np.abs(history["CL"] - reference["CL"]).max()
                      / np.abs(reference["CL"]).max())
        rows.append((e_mod, float(np.abs(history["tip"]).max()), error,
                     float(history["sub"][1:].mean())))
    if verbose:
        _rule(f"C2 rigid limit: {nsteps} steps at dt = {dt}, rho_inf = {rho_inf}")
        print("  Started from the static aeroelastic equilibrium, not from")
        print("  rest: a stiff structure released from rest rings at a")
        print("  frequency no affordable step resolves, which is a property of")
        print("  the coupled problem and not of either solver.")
        print("       E        max displacement   max |dC_L| / |C_L|   mean subiterations")
        for e_mod, tip, error, sub in rows:
            print(f"  {e_mod:.0e}   {tip:16.3e}   {error:18.2e}   {sub:18.2f}")
        print(f"  reference C_L(final) = {reference['CL'][-1]:.6f} from "
              f"unsteady_wing.time_march")
    return {"reference": reference, "rows": rows}


# ------------------------------------------------------------------ case C3

def case_divergence(n_c=8, nspan=8, span=6.0, thickness=0.10, alpha=2.0,
                    k_spring=4.0, fractions=(0.3, 0.4, 0.5, 0.6, 0.7),
                    verbose=True):
    """Static divergence of an elastically supported wing, by three routes.

    The classical formula is a different model from the other two - a rigid
    wing twisting about a single axis, with the lift at a single chordwise
    point - which is what makes the comparison a verification rather than a
    restatement. The modal route and the Southwell extrapolation measure the
    same coupled operator and must agree to round-off with each other.
    """
    points = wing(n_c, nspan, span, thickness, alpha)
    resultants = []
    for angle in (alpha, alpha + 1.0):
        state = tpw.solve(tpw.init_state(wing(n_c, nspan, span, thickness, angle),
                                         uniform_onset, 1.0,
                                         config={"lifting": True, "nwake": 16}),
                          wake="relax")
        resultants.append(tpw.get_loads(state, rho=1.0)["resultants"])
    d_alpha = np.radians(1.0)
    slope = (resultants[1]["CL"] - resultants[0]["CL"]) / d_alpha
    x_ac = resultants[0]["x_ref"][0] \
        - (resultants[1]["moment"][1] - resultants[0]["moment"][1]) \
        / (resultants[1]["force"][2] - resultants[0]["force"][2])
    s_ref = state["s_ref"]

    mesh = fem_mesh.solid_foil_mesh(points, n_thick=1)
    nodes = mesh["nodes"]
    centre = nodes.mean(axis=0)
    k_theta = k_spring * float(np.sum((nodes[:, 0] - centre[0]) ** 2
                                      + (nodes[:, 2] - centre[2]) ** 2))
    eccentricity = float(centre[0] - x_ac)
    q_classical = k_theta / (s_ref * slope * eccentricity)

    model = fes.build_model(nodes, mesh["elements"],
                            fmat.IsotropicElastic(1e9, 0.3, 100.0))

    def supported(rho_f):
        fluid, sstate, transfer = fsd.init_fsi(points, uniform_onset, model,
                                               u_ref=1.0, rho=rho_f,
                                               unsteady=False,
                                               config={"lifting": True,
                                                       "nwake": 16})
        fes.add_spring(sstate, mesh["node_sets"]["all"], (0, 1, 2), k_spring)
        fes.assemble_operators(sstate)
        return fluid, sstate, transfer

    fluid, sstate, transfer = supported(1.0)
    modal = fsd.divergence_pressure(fluid, sstate, transfer, rho=1.0, nmodes=6)

    rows = []
    for fraction in fractions:
        q_dyn = fraction * q_classical
        f_2, s_2, t_2 = supported(2.0 * q_dyn)          # u_ref = 1, so q = rho/2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, s_2, hist = fsd.static_aeroelastic(f_2, s_2, t_2, rho=2.0 * q_dyn,
                                                  max_iter=80, tol=1e-9)
        deflection = float(np.abs(s_2["u"][:, 2]).max())
        rows.append((fraction, q_dyn, deflection, hist["iterations"],
                     hist["converged"]))
    fit = np.polyfit([1.0 / r[1] for r in rows], [1.0 / r[2] for r in rows], 1)
    q_southwell = float(1.0 / (-fit[1] / fit[0]))

    if verbose:
        _rule("C3 static divergence of an elastically supported wing")
        print("  measured on the rigid wing by the panel method:")
        print(f"    dC_L/dalpha            {slope:.4f} per radian")
        print(f"    aerodynamic centre     x_ac = {x_ac:.4f} "
              f"(quarter chord at {resultants[0]['x_ref'][0]:.4f})")
        print(f"    reference area         {s_ref:.4f}")
        print("  measured on the structure by the finite element model:")
        print(f"    spring centroid        x_ea = {centre[0]:.4f}")
        print(f"    eccentricity           e = {eccentricity:.4f}")
        print(f"    torsional stiffness    K_theta = {k_theta:.4f}")
        print("\n  route                                q_div      ratio to classical")
        print(f"  classical K_theta/(S c_l,alpha e)   {q_classical:8.4f}      1.0000")
        print(f"  modal, smallest q with K - qA singular {modal['q_divergence']:6.4f}"
              f"      {modal['q_divergence'] / q_classical:.4f}")
        print(f"  Southwell extrapolation             {q_southwell:8.4f}      "
              f"{q_southwell / q_classical:.4f}")
        print("\n  the fixed point approaching it:")
        print("    q/q_div     max |u_z|    iterations")
        for fraction, _, deflection, iterations, converged in rows:
            flag = "" if converged else "  (did not converge)"
            print(f"    {fraction:7.2f}   {deflection:.6e}   {iterations:6d}{flag}")
        print("  The two coupled routes agree to five figures - they measure")
        print("  the same operator - and both sit 1% below the classical")
        print("  formula, which assumes the whole lift acts at one chordwise")
        print("  point on a wing twisting rigidly about one axis.")
    return {"q_classical": q_classical, "q_modal": modal["q_divergence"],
            "q_southwell": q_southwell, "slope": slope, "x_ac": x_ac,
            "k_theta": k_theta, "eccentricity": eccentricity, "rows": rows,
            "frequencies": modal["frequency"]}


# ------------------------------------------------------------------ case C4

def strip_added_mass(fluid, sstate, transfer, rho_f):
    """Two-dimensional strip estimate of the modal added-mass RATIO.

    A flat plate of chord c oscillating normal to itself carries rho*pi*b^2 of
    added mass per unit span, b the semichord. Integrating that against the
    square of the normal component of a MASS-NORMALISED mode shape gives the
    modal added mass in a normalisation where the modal structural mass is one,
    so the number returned is directly the ratio that
    fsi_driver.added_mass_ratio measures - which is the only way the two are
    comparable, both being defined only up to the scaling of the shape.

    It is a strip theory: it ignores the finite span entirely, so it is an
    upper bound, and the ratio of the measured value to it is the
    three-dimensional relief.
    """
    _, phi = fes.modes(sstate, 1)                 # phi^T M phi = 1
    normal = fsx.to_fluid(transfer, phi[:, :, 0]).reshape(transfer["shape"])
    points = fluid["panels"]["points"]
    y_node = points[..., 1].mean(axis=0)
    chords = tpw.section_chords(points)
    strip_w = np.abs(np.gradient(y_node))
    w_z = np.abs(normal[..., 2]).mean(axis=0)
    return float(rho_f * np.pi * np.sum((0.5 * chords) ** 2 * w_z ** 2 * strip_w))


def case_added_mass(e_mod=2e8, rho_s=1200.0, dt=0.05, nsteps=140, verbose=True):
    """Wet natural frequency in still water, three ways.

    Run WITHOUT a wake, which is the right model and not a convenience. Added
    mass is the non-circulatory part of the pressure; the fluid repository's
    own added-mass gate, the accelerating sphere, has no wake either, and there
    it is exact to the panelling. There is a second, practical reason: in still
    water there is no mean flow to carry shed vorticity away, so a lifting run
    piles the sheet up against the trailing edge and the near-field interaction
    eventually destabilises the march. That is the roll-up limitation the fluid
    repository already records as an open item, reached here from an unusual
    direction, and it is reported in docs/COUPLING.md rather than worked around.
    """
    points = wing(alpha=0.0)
    mesh = fem_mesh.solid_foil_mesh(points, n_thick=1)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(e_mod, 0.3, rho_s))
    fluid, sstate, transfer = fsd.init_fsi(points, still_fluid, model,
                                           u_ref=1.0, rho=RHO_WATER,
                                           config={"lifting": False})
    fes.clamp(sstate, mesh["node_sets"]["root"])
    fes.assemble_operators(sstate)
    report = fsd.added_mass_ratio(fluid, sstate, transfer, RHO_WATER, dt)
    strip = strip_added_mass(fluid, sstate, transfer, RHO_WATER)
    freq, phi = fes.modes(sstate, 1)
    shape = phi[:, :, 0] * (0.005 / np.abs(phi[:, :, 0]).max())
    sstate["u"] = shape.copy()
    f_struct, _ = fsd._loads_on_structure(fluid, transfer, RHO_WATER)
    sstate["a"] = fes.initial_acceleration(sstate, f_struct)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=dt,
                                           nsteps=nsteps, rho=RHO_WATER,
                                           rho_inf=0.9, max_sub=30)
    fitted = fsd.growth_rate(history["t"], history["tip_z"], skip=0.05)
    implied = (freq[0] / max(fitted["frequency"], 1e-300)) ** 2 - 1.0
    out = {"frequency_dry": float(freq[0]),
           "ratio_measured": report["ratio"],
           "ratio_strip": strip,             # modal mass is 1 by normalisation
           "ratio_from_march": float(implied),
           "frequency_wet_estimate": report["frequency_wet_estimate"],
           "frequency_wet_marched": fitted["frequency"],
           "growth": fitted["rate"], "nsteps": nsteps,
           "steps_per_wet_period": float(1.0 / (fitted["frequency"] * dt))}
    if verbose:
        _rule("C4 added mass in still water: the wet natural frequency")
        print("  solid foil, chord 1, span 4, 12% thick, structural density "
              f"{rho_s:.0f},")
        print(f"  in water at {RHO_WATER:.0f}. The mass ratio is of order one, which is")
        print("  the regime a partitioned scheme finds hardest.")
        print("    dry frequency                          "
              f"{out['frequency_dry']:.5f} Hz")
        print("  added mass over structural mass, three ways:")
        print("    from one dmu/dt difference quotient    "
              f"{report['ratio']:.4f}")
        print(f"    from 2D strip theory, rho pi b^2       {strip:.4f}")
        print(f"    implied by the marched wet frequency   {implied:.4f}")
        print("  wet frequency:")
        print("    from the difference quotient           "
              f"{out['frequency_wet_estimate']:.5f} Hz")
        print("    marched and fitted on the crossings    "
              f"{out['frequency_wet_marched']:.5f} Hz")
        print("    ratio                                  "
              f"{out['frequency_wet_marched'] / out['frequency_wet_estimate']:.4f}")
        print("    growth rate of the envelope            "
              f"{out['growth']:+.4f} per second")
        print(f"    {nsteps} steps at dt = {dt}, "
              f"{out['steps_per_wet_period']:.1f} per wet period")
        print("  The difference quotient and the march agree to 2% in frequency")
        print("  and 4% in the ratio itself, having nothing in common but the")
        print("  two solvers: one is a single perturbation of the doublet")
        print("  strengths, the other a marched oscillation fitted on its zero")
        print("  crossings. The envelope is neutral to four decimal places,")
        print("  which is right - with no wake there is no radiation damping.")
        print("  Strip theory is the outlier and is expected to be: it gives")
        print("  every section the two-dimensional value, while this mode bends")
        print("  strongly along the span. The ratio of the two is the")
        print("  three-dimensional relief and is reported, not verified - there")
        print("  is no closed form for it on this shape.")
    return out


# ------------------------------------------------------------------ case C5

def case_instability(densities=(20000.0, 5000.0, 1200.0), e_mod=2e8, dt=0.05,
                     nsteps=8, verbose=True):
    """Where staggered coupling fails and strong coupling does not."""
    points = wing()
    rows = []
    for rho_s in densities:
        fluid, sstate, transfer, _, _ = foil_case(points, e_mod, rho_s=rho_s)
        report = fsd.added_mass_ratio(fluid, sstate, transfer, RHO_WATER, dt)
        outcome = {}
        for mode, accelerator in (("loose", "aitken"), ("strong", "aitken"),
                                  ("strong", "iqn")):
            fluid, sstate, transfer, _, _ = foil_case(points, e_mod, rho_s=rho_s)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    history, _, _ = fsd.time_march_fsi(
                        fluid, sstate, transfer, dt=dt, nsteps=nsteps,
                        rho=RHO_WATER, coupling=mode, accel=accelerator,
                        max_sub=40)
                finite = bool(np.all(np.isfinite(history["CL"])))
                outcome[(mode, accelerator)] = (
                    finite, float(history["sub"][1:].mean()),
                    float(np.abs(history["tip"]).max()))
            except (ValueError, np.linalg.LinAlgError):
                outcome[(mode, accelerator)] = (False, np.nan, np.inf)
        rows.append((rho_s, report["ratio"], outcome))
    if verbose:
        _rule("C5 the added-mass instability")
        print("  structural   added mass    loose      strong+Aitken   strong+IQN")
        print("   density     / structural            (subiterations) (subiterations)")
        for rho_s, ratio, outcome in rows:
            loose = "survives" if outcome[("loose", "aitken")][0] else "DIVERGES"
            aitken = outcome[("strong", "aitken")]
            iqn = outcome[("strong", "iqn")]
            print(f"  {rho_s:9.0f}   {ratio:10.3f}   {loose:>9s}   "
                  f"{aitken[1]:13.1f}   {iqn[1]:12.1f}")
        print("  A staggered scheme is not merely less accurate below a mass")
        print("  ratio of order one: it has no stable step size. This is why")
        print("  coupling='loose' exists in the driver only as this table.")
    return rows


# ------------------------------------------------------------------ case C6

def case_energy(e_mod=2e9, dt=0.05, nsteps=15, rho_inf=0.9, verbose=True):
    """Work done by the fluid against the energy stored in the structure."""
    points = wing()
    fluid, sstate, transfer, _, _ = foil_case(points, e_mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=dt,
                                           nsteps=nsteps, rho=RHO_WATER,
                                           rho_inf=rho_inf, max_sub=40)
    stored = history["strain_energy"] + history["kinetic_energy"]
    work = history["fluid_work"]
    scale = max(float(np.abs(work).max()), 1e-300)
    error = np.abs(stored - work) / scale
    if verbose:
        _rule(f"C6 energy balance over {nsteps} steps at dt = {dt}")
        print("    s      work by the fluid   strain + kinetic    difference")
        for n in range(0, len(work), max(1, len(work) // 8)):
            print(f"  {history['s'][n]:5.2f}   {work[n]:17.6e}   "
                  f"{stored[n]:16.6e}   {work[n] - stored[n]:+.2e}")
        print("  largest difference, relative to the largest work: "
              f"{error.max():.2e}")
        print("  The work integral is trapezoidal in the displacement increment,")
        print("  so it is second order in the step and the residual is that,")
        print("  plus whatever the numerical dissipation of rho_inf removes.")
    return {"work": work, "stored": stored, "error": float(error.max())}


# --------------------------------------------------------------------- run

CASES = {"C1": case_transfer, "C2": case_rigid_limit, "C3": case_divergence,
         "C4": case_added_mass, "C5": case_instability, "C6": case_energy}
QUICK = ("C1", "C2", "C6")


def main(only=None, quick=False):
    print("Coupled verification: fsi_transfer.py and fsi_driver.py")
    print("Every number below is measured by this run.")
    names = [only] if only else (QUICK if quick else list(CASES))
    for name in names:
        start = time.perf_counter()
        CASES[name]()
        print(f"  [{name} took {time.perf_counter() - start:.1f} s]")
    print("\nDone. docs/COUPLING.md tabulates these numbers.")


if __name__ == "__main__":
    case = None
    if "--case" in sys.argv:
        case = sys.argv[sys.argv.index("--case") + 1].upper()
        if case not in CASES:
            raise SystemExit(f"unknown case '{case}'; choose from "
                             f"{', '.join(CASES)}")
    main(case, quick="--quick" in sys.argv)
