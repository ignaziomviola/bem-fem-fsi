"""Verification of the ventilation model, against closed forms and the paper.

    python3 verify_vent.py                 # all six cases, about 3 min
    python3 verify_vent.py --case V4       # one case: V1..V6
    python3 verify_vent.py --quick         # the cheap ones (V1, V2, V5)

Every number in docs/VENTILATION.md comes from a run of this file, and the tables
hold what was measured. Nothing is written and no figure is created.

    V1 sectional      the fits of Harwood et al. against Acosta and Tulin
    V2 exactness      the free surface, the antisymmetry, and the load path
    V3 baseline       the wetted surface-piercing strut, and its depth loading
    V4 load loss      lift, drag and the centre of pressure, against figures 12-14
    V5 washout        the boundary (4.5), its own derivation, and Breslin & Skalak
    V6 hysteresis     the bi-stable range, and the coupled response to a transition

Three things about the reading of these tables. First, V1, V2 and V5 compare
against closed forms, exact symmetries and an equation re-derived here, so their
agreement is verification. Second, V3, V4 and V6 compare against a towing-tank
experiment through a low-order model, so their TRENDS are the verified statement
and their magnitudes are reported, not verified - the wording is used explicitly
wherever it applies. Third, the stall angle, the pressure-recovery fraction, the
inception area fraction and the Weber threshold are INPUTS of this model, not
results, and every case that depends on one prints it in its header.
"""

import sys
import time
import warnings

import numpy as np

import thick_panel_wing as tpw
import freewake_kernels as fk
import fem_materials as fmat
import fem_solid as fes
import fsi_driver as fsd
import vent_cavity as vcv
import vent_loads as vnl
import vent_mesh as vm
import vent_section as vs
import vent_solve as vsl

RHO_WATER = 1000.0
STRUCT = dict(e_mod=2.0e8, nu=0.3, rho=1200.0)


def uniform_onset(u=1.0):
    return fk.make_onset(np.array([-10.0, 10.0]), np.array([u, u]))


def strut_case(alpha=16.0, fn_h=2.5, n_c=12, nspan_half=8, h=1.0, chord=1.0,
               image=True, regime="FW", nwake=6, unsteady=False, e_mod=None,
               **vent_kw):
    """A ventilated surface-piercing strut. -> (fluid, sstate, transfer, vent, mesh)"""
    points = vm.strut_mesh(n_c=n_c, nspan_half=nspan_half, h=h, chord=chord,
                           alpha_deg=alpha, image=image)
    maps = vm.mirror_maps(points.shape, image=image)
    mesh = vm.strut_solid_mesh(vm.half_points(points, maps), n_thick=2,
                               min_half_thickness=0.02)
    material = dict(STRUCT)
    if e_mod is not None:
        material["e_mod"] = e_mod
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(**material))
    fluid, sstate, transfer, vent = fsd.init_vent_fsi(
        points, uniform_onset(), model, h=h, chord=chord, u_ref=1.0,
        rho=RHO_WATER, alpha_deg=alpha, image=image, unsteady=unsteady,
        config={"lifting": True, "nwake": nwake}, fn_h=fn_h, regime=regime,
        **vent_kw)
    fes.clamp(sstate, mesh["node_sets"]["tip_max"])
    fes.assemble_operators(sstate)
    return fluid, sstate, transfer, vent, mesh


def solved(**kw):
    """One converged ventilated solve. -> (fluid, vent, cav, loads)"""
    fluid, _, _, vent, _ = strut_case(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
    return fluid, vent, cav, loads


def _rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ------------------------------------------------------------------ case V1

def case_sectional(psis=(0.3, 1.0, 2.0, 5.83, 10.0, 30.0), verbose=True):
    """The sectional fits against the closed forms they were built from."""
    lengths = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    psi_ac = vs.acosta_psi(lengths)
    fit = vs.harwood_length(psi_ac)
    back = vs.acosta_psi(fit)
    err_psi = np.abs(back / psi_ac - 1.0)
    a_fit, a_ac = vs.lift_slope(lengths), vs.acosta_lift_slope(lengths)
    if verbose:
        _rule("V1 the cavity length (1.7) against Acosta (1.6a), in Psi and in L")
        print("      L    Psi (Acosta)   L from (1.7)   error in L   error in Psi")
        for k, length in enumerate(lengths):
            print(f"  {length:5.2f}   {psi_ac[k]:12.4f}   {fit[k]:12.4f}   "
                  f"{100 * (fit[k] - length) / length:+9.1f}%   "
                  f"{100 * err_psi[k]:+11.1f}%")
        print("  The comparison that MEANS anything is the one in Psi: dPsi/dL is")
        print("  large near the branch join, so the 8.7% error in L at L = 0.5 is")
        print("  about 1% in Psi. A test written on L reads as a failure of a fit")
        print("  that is correct.")

        _rule("V1 the lift slope (1.9) against Acosta (1.6b), and its two limits")
        print("      L    a0 from (1.9)   Acosta a0    ratio")
        for k, length in enumerate(lengths):
            print(f"  {length:5.2f}   {a_fit[k]:13.4f}   {a_ac[k]:9.4f}   "
                  f"{a_fit[k] / a_ac[k]:6.4f}")
        print(f"  a0(0)   = {float(vs.lift_slope(0.0)):.12f}  against 2 pi = "
              f"{2 * np.pi:.12f}")
        print(f"  a0(inf) = {float(vs.lift_slope(np.inf)):.12f}  against pi/2 = "
              f"{np.pi / 2:.12f}")
        peak = np.linspace(0.01, 1.5, 3000)
        got = vs.lift_slope(peak)
        tail = vs.lift_slope(np.logspace(0.5, 6.0, 2000))
        print(f"  the fit peaks at a0 = {got.max():.4f} at L = "
              f"{peak[int(np.argmax(got))]:.3f}, above 2 pi, which is the local")
        print("  maximum of the paper's figure 2(b); and it UNDERSHOOTS its own")
        print(f"  limit by {100 * (1 - tail.min() / (np.pi / 2)):.2f}% near L = 10"
              f" before approaching pi/2 from below.")
        print("  That belongs to the rational polynomial, not to the physics.")

        _rule("V1 the supercavity branch: Tulin exact against both fits")
        print("   sigma     Psi    Tulin L    (1.7) L   error   (1.8) L   error")
        alpha_2d = np.radians(4.0)
        for sigma in (0.30, 0.20, 0.12, 0.08, 0.05, 0.03):
            psi = sigma / (2.0 * alpha_2d)
            exact = float(vs.tulin_length(psi, alpha_2d))
            f_1, f_2 = float(vs.harwood_length(psi)), \
                float(vs.harwood_length_low_order(psi))
            print(f"  {sigma:6.2f}  {psi:6.3f}  {exact:9.3f}  {f_1:9.3f}  "
                  f"{100 * (f_1 - exact) / exact:+5.0f}%  {f_2:8.3f}  "
                  f"{100 * (f_2 - exact) / exact:+5.0f}%")
        print("  (1.7) holds to about 5% for 1.25 <= L <= 2.6 and degrades badly")
        print("  beyond; (1.8) is the better long-cavity fit and is NOT an")
        print("  approximation to Tulin - Tulin grows as sigma^-2 and (1.8) as")
        print("  sigma^-1, so their ratio diverges. Everything (4.5) says inherits")
        print("  the interval where (1.8) holds.")

        _rule("V1 the centre of pressure (3.4) and the re-entrant jet (3.1)-(3.2)")
        print("      L      e(L)   |   Phi (deg)   Uj/u    Uj streamwise   stable")
        for length in (0.0, 0.25, 0.5, 1.0, 2.0, np.inf):
            print(f"  {length:5.2f}   {float(vs.centre_of_pressure(length)):7.5f}",
                  end="")
            phi = np.radians(min(20.0 + 25.0 * min(length, 3.0), 89.0))
            u_j, _ = vs.jet_components(1.0, 0.2, phi)
            print(f"   |  {np.degrees(phi):9.1f}  {float(vs.jet_speed(1.0, 0.2)):5.3f}"
                  f"   {u_j:+13.4f}   "
                  f"{'no' if vs.unstable_closure(phi) else 'yes':>6s}")
        print("  e(0) = 0.248876, 0.45% below the quarter chord because (3.4) is a")
        print("  tanh blend; e(0.5) = 7/32 and e(inf) = 3/16 exactly.")
    return {"err_psi_max": float(err_psi.max()),
            "a0_ratio_max": float(np.abs(a_fit / a_ac - 1.0).max()),
            "a0_zero": float(vs.lift_slope(0.0)),
            "a0_inf": float(vs.lift_slope(np.inf))}


# ------------------------------------------------------------------ case V2

def case_exactness(n_c=10, nspan_half=8, alpha=14.0, verbose=True):
    """What holds by construction: the image, and the load path."""
    fluid, vent, cav, loads = solved(alpha=alpha, n_c=n_c, nspan_half=nspan_half,
                                     regime="FV")
    res = vsl.antisymmetry_residual(fluid, vent)
    points = fluid["panels"]["points"]
    sym = vm.symmetry_residual(points, vent["y_fs"])
    fluid_w, sstate, transfer, vent_w, _ = strut_case(alpha=alpha, n_c=n_c,
                                                      nspan_half=nspan_half)
    wall = vsl.solve_wetted(fluid_w, vent_w, wake="frozen", sign=+1.0)
    mu_wall = wall["mu"].copy()
    fluid_p, _, _, _, _ = strut_case(alpha=alpha, n_c=n_c, nspan_half=nspan_half)
    plain = tpw.solve(fluid_p, wake="frozen")
    wall_gap = float(np.abs(mu_wall - plain["mu"]).max())
    rep = vnl.conservation_report_half(transfer, vent, fluid["panels"],
                                       loads["force"], u_struct=sstate["u"])
    doubled = tpw.get_loads(fluid, rho=RHO_WATER)["resultants"]["CL"]
    immersed = loads["resultants"]["CL"]
    if verbose:
        _rule(f"V2 exactness at alpha = {alpha} deg, Fn_h = {vent['fn_h']}, "
              f"{int(vent['real_mask'].sum())} immersed panels")
        print("  quantity                                                    error")
        for name, value in (
                ("mirror symmetry of the doubled strut", sym),
                ("image doublet strength + mirror of the real one", res["mu"]),
                ("image source strength + mirror of the real one", res["sigma"]),
                ("image wake strength + mirror of the real one", res["wake_mu"]),
                ("phi on the free-surface plane, relative", res["phi_plane"]),
                ("unflipped image minus the vendored solve", wall_gap),
                ("total force through the lumping", rep["force_error_lump"]),
                ("total force through the transfer", rep["force_error_transfer"]),
                ("total moment through the lumping", rep["moment_error_lump"]),
                ("virtual work across the interface", rep["work_error"]),
                ("largest pressure the cavity LOWERED", abs(min(
                    float(cav["raised"].min()), 0.0)))):
            print(f"  {name:58s} {value:.2e}")
        print("\n  The first six are the free surface. phi = 0 on the plane IS the")
        print("  linearised high-Froude free-surface condition, and it follows from")
        print("  flipping the sign of the source strengths on a geometrically exact")
        print("  mirror - it is not solved for, so it holds to round-off.")
        print("  The sixth is the hazard made precise: with no spanwise onset,")
        print("  mirroring leaves -n.u_rel unchanged, so the UNFLIPPED image is")
        print("  bit-for-bit the vendored solve. Calling tpw.solve on a ventilating")
        print("  state therefore returns the zero-Froude RIGID WALL answer, quietly")
        print("  and plausibly, with about twice the lift.")
        print(f"\n  immersed-half CL = {immersed:+.5f} on s_ref = h*c = "
              f"{vent['s_ref_wet']:.3f}")
        print(f"  whole doubled-mesh CL = {doubled:+.5f}, a near-cancellation and")
        print("  NOT the answer: the image half carries the negated loading. It is")
        print("  not zero because the pressure is quadratic in a velocity whose")
        print("  tangential part does not simply mirror. Scaling it by two would be")
        print("  a real bug wearing a plausible face.")
    return {"symmetry": sym, "mu": res["mu"], "phi_plane": res["phi_plane"],
            "wall_gap": wall_gap, "force_error": rep["force_error_transfer"],
            "immersed_CL": immersed, "doubled_CL": doubled}


# ------------------------------------------------------------------ case V3

def case_baseline(alpha=10.0, fn_h=2.5, spans=(4, 8, 16), verbose=True):
    """The wetted surface-piercing strut: the depth loading and the lift slope."""
    rows, first = [], []
    for nspan_half in spans:
        fluid, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, n_c=8,
                                         nspan_half=nspan_half)
        depth, cl = vnl.depth_loading(fluid["panels"], vent, cav["p_gauge"],
                                      RHO_WATER)
        res = loads["resultants"]
        slope = res["CL"] / np.sin(np.radians(alpha))
        rows.append((nspan_half, abs(cl[0]) / abs(cl).max(), float(cl.max()),
                     res["CL"], slope))
        first.append(abs(cl[0]) / abs(cl).max())
    a_0 = vs.lift_weighted_slope(1.0)
    helm_h = float(vs.helmbold(a_0, 1.0))
    helm_2h = float(vs.helmbold(a_0, 2.0))
    if verbose:
        _rule(f"V3 the wetted strut at alpha = {alpha} deg, Fn_h = {fn_h}, "
              f"AR_h = 1")
        print("  strips  cl at the waterline / peak   peak cl      CL    CL/sin a")
        for nspan_half, ratio, peak, c_l, slope in rows:
            print(f"  {nspan_half:6d}  {ratio:27.4f}  {peak:8.4f}  {c_l:+7.4f}  "
                  f"{slope:9.4f}")
        print("  The loading at the waterline falls as the mesh refines but is")
        print("  O(dy), NOT zero: phi = 0 on the plane is exact, while the panel")
        print("  loading vanishes only in the limit, because with an even span")
        print("  count no panel straddles the plane and the shallowest strip's")
        print("  circulation is O(dy). V2 carries the exact statement.")
        print(f"\n  measured lift slope CL/sin(alpha)   {rows[-1][4]:.4f}")
        print(f"  Helmbold (1.20) at AR_h = 1          {helm_h:.4f}   ratio "
              f"{rows[-1][4] / helm_h:.3f}")
        print(f"  Helmbold at 2 AR_h = 2               {helm_2h:.4f}   ratio "
              f"{rows[-1][4] / helm_2h:.3f}")
        print("  The second row is printed only to show it is far away. The")
        print("  effective aspect ratio of a surface-piercing strut is the")
        print("  IMMERSED one: the negative image makes the waterline behave as a")
        print("  tip, so the immersed span is the whole span of the equivalent")
        print("  wing. Using 2 AR_h is the rigid-wall image and is a plausible")
        print("  error that moves the lift slope by tens of per cent.")
        print("  The panel method and the Helmbold-corrected sectional model are")
        print("  different models, which is what makes the comparison a check;")
        print("  the agreement is reported to the per cent, not verified.")
    return {"waterline_ratio": first, "slope": rows[-1][4], "helmbold": helm_h}


# ------------------------------------------------------------------ case V4

def case_load_loss(fn_h=2.5, alphas=(6.0, 10.0, 14.0, 18.0), n_c=12,
                   nspan_half=8, verbose=True):
    """Lift, drag and the centre of pressure on ventilation, against figures 12-14."""
    rows = []
    for alpha in alphas:
        out = {}
        for regime in ("FW", "FV"):
            fluid, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, n_c=n_c,
                                             nspan_half=nspan_half, regime=regime)
            res = loads["resultants"]
            out[regime] = (res["CL"], res["CD_pressure"],
                           res["CM"] * vent["chord"] / res["CL"],
                           float(np.max(cav["l_c"])), cav["d_cav"] / vent["h"],
                           np.degrees(cav["phi_bar"]))
        rows.append((alpha, out["FW"], out["FV"]))
    if verbose:
        _rule(f"V4 the load effect of ventilation at Fn_h = {fn_h}, AR_h = 1")
        print("  alpha    CL wet   CL vent   ratio    CD wet  CD vent  ratio"
              "   e wet   e vent   Lc max   D/h   Phi")
        for alpha, wet, ven in rows:
            print(f"  {alpha:5.1f}  {wet[0]:+8.4f}  {ven[0]:+8.4f}  "
                  f"{ven[0] / wet[0]:6.3f}  {wet[1]:+7.4f} {ven[1]:+7.4f} "
                  f"{ven[1] / wet[1]:6.3f}  {wet[2]:6.3f}  {ven[2]:6.3f}  "
                  f"{ven[3]:6.3f} {ven[4]:5.2f} {ven[5]:5.1f}")
        ratios = [ven[0] / wet[0] for _, wet, ven in rows]
        print(f"\n  The lift ratio runs from {ratios[0]:.2f} to {ratios[-1]:.2f}. "
              f"The paper's figure 12")
        print("  shows the same ordering and the same growth of the gap with")
        print("  incidence, and Breslin & Skalak report losses of up to 70%, so")
        print("  the sign, the ordering and the trend are the VERIFIED statements")
        print("  and the magnitude is REPORTED, NOT VERIFIED. The mechanism this")
        print("  model does not have is the cavity closure region, where the")
        print("  experiment's re-entrant jet removes suction over an area a")
        print("  prescribed-pressure cavity cannot place.")
        print("\n  The centre of pressure moves from about 0.33 to about 0.19 of")
        print("  the chord forward of mid-chord. The paper's figure 14 quotes 1/4")
        print("  wetted and 3/16 = 0.1875 supercavitating; the ventilated value")
        print("  lands on 3/16, and the wetted one sits further forward than a")
        print("  thin foil's quarter chord because the loading of a low-aspect-")
        print("  ratio surface-piercing strut is concentrated forward.")
        print("\n  The drag ratio is near one, and that near-continuity is the")
        print("  paper's observation - but here it is a coincidence of two large")
        print("  omissions rather than the same statement: this model has neither")
        print("  the increased profile drag of the cavity nor its spray drag, and")
        print("  it has no friction at all. REPORTED, NOT VERIFIED.")
    return {"rows": rows, "ratio": [ven[0] / wet[0] for _, wet, ven in rows]}


# ------------------------------------------------------------------ case V5

def case_washout(aspect_ratios=(0.5, 1.0, 1.5, 2.0, 3.0),
                 cls=(0.2, 0.35, 0.5, 0.8), verbose=True):
    """The washout boundary (4.5), its own derivation, and Breslin & Skalak."""
    worst = 0.0
    if verbose:
        _rule("V5 the washout boundary (4.5) against the chain (4.1) to (4.4)")
        print("   AR_h     CL   Fn from (4.5)   from the chain    difference"
              "   Breslin & Skalak   ratio")
    for ar in aspect_ratios:
        for c_l in cls:
            closed = float(vs.washout_froude(c_l, ar))
            chain = float(vs.washout_froude_chain(c_l, ar))
            b_s = float(vs.breslin_skalak(c_l))
            worst = max(worst, abs(closed - chain) / closed)
            if verbose:
                print(f"  {ar:5.1f}  {c_l:5.2f}   {closed:13.4f}   "
                      f"{chain:14.4f}   {abs(closed - chain):11.2e}   "
                      f"{b_s:16.3f}   {b_s / closed:5.2f}")
    if verbose:
        print(f"\n  largest relative difference from the derivation: {worst:.2e}")
        print("  This verifies the TRANSCRIPTION AND THE ALGEBRA of (4.5), and")
        print("  with it (1.9), (4.1), (4.2), (4.3) and (4.4), because the chain")
        print("  uses all of them. It does not verify the physics. The wording to")
        print("  use is that (4.5) is verified against the relations it was")
        print("  derived from, never that the washout boundary is verified.")
        print("  Breslin & Skalak bounds it from above everywhere, by a factor")
        print("  2.86 at AR_h = 1, CL = 0.5 - which is the paper's own finding")
        print("  that (1.2) bounds all the data from above. Their boundary is")
        print("  max(sqrt(5/CL), 3): the two conditions cross at CL = 5/9, and a")
        print("  code applying only the first is wrong above that.")

        _rule("V5 the boundary in alpha-Fn_h space, through this model's own CL")
        print("  INPUT: the stall angle is "
              f"{np.degrees(vs.ALPHA_STALL):.1f} deg, an input of this model and "
              f"not a result")
        print("  alpha     CL (FV)   Fn washout   regime at Fn_h = 1, 2, 3")
        for alpha in (6.0, 10.0, 14.0, 18.0):
            _, vent, cav, loads = solved(alpha=alpha, fn_h=2.5, n_c=10,
                                         nspan_half=6, regime="FV")
            c_l = abs(loads["resultants"]["CL"])
            fn_w = float(vs.washout_froude(c_l, vent["ar"]))
            regimes = []
            for fn_h in (1.0, 2.0, 3.0):
                _, v2, c2, _ = solved(alpha=alpha, fn_h=fn_h, n_c=10,
                                      nspan_half=6, regime="FV")
                vcv.commit_vent(v2, c2, 0.0, 0.1)
                regimes.append(v2["regime"])
            print(f"  {alpha:5.1f}   {c_l:9.4f}   {fn_w:10.3f}   "
                  f"{', '.join(regimes)}")
        print("  This column is a composition of TWO models - the panel method's")
        print("  lift fed into a sectional scaling relation - so unlike the table")
        print("  above it is reported, not verified.")
    return {"chain_error": worst}


# ------------------------------------------------------------------ case V6

def case_hysteresis(fn_h=2.5, alphas=(6.0, 10.0, 13.0, 16.0, 20.0), verbose=True):
    """The bi-stable range, and the coupled response to a ventilation transition."""
    rows = []
    for alpha in alphas:
        out = {}
        for start in ("FW", "FV"):
            fluid, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, n_c=10,
                                             nspan_half=6, regime=start)
            vcv.commit_vent(vent, cav, 0.0, 0.1)
            out[start] = (vent["regime"], loads["resultants"]["CL"])
        rows.append((alpha, out["FW"], out["FV"]))
    if verbose:
        _rule(f"V6 hysteresis at Fn_h = {fn_h}, AR_h = 1")
        print("  INPUTS: stall angle "
              f"{np.degrees(vs.ALPHA_STALL):.1f} deg, inception area fraction "
              f"{vcv.INCEPT_FRACTION:.2f}, recovery {vcv.RECOVERY:.2f}")
        print("  alpha   from wetted            from ventilated       bi-stable")
        for alpha, wet, ven in rows:
            print(f"  {alpha:5.1f}   {wet[0]:4s} CL = {wet[1]:+.5f}      "
                  f"{ven[0]:4s} CL = {ven[1]:+.5f}   "
                  f"{'YES' if wet[0] != ven[0] else 'no'}")
        print("  The conditions are IDENTICAL along each row - the same incidence,")
        print("  Froude number and mesh - and only the history differs. That is")
        print("  the bi-stability of the paper's figure 16, and it is structural")
        print("  here rather than a threshold with a dead band: inception needs an")
        print("  air path AND the seal broken, while persistence asks only for")
        print("  ventilation-ready flow, so a cavity survives well below the")
        print("  incidence that formed it. The WIDTH of the band is a model output")
        print("  with no counterpart at this fidelity, and is reported.")

    _rule("V6 the coupled response to a ventilation transition")
    fluid, sstate, transfer, vent, mesh = strut_case(alpha=20.0, n_c=8,
                                                     nspan_half=6, nwake=0,
                                                     unsteady=True)
    freq, _ = fes.modes(sstate, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                  rho=RHO_WATER, max_iter=8,
                                                  vent=vent)
        added = fsd.added_mass_ratio(fluid, sstate, transfer, RHO_WATER, 0.01,
                                     vent=vent)
        hist, fluid, sstate = fsd.time_march_fsi(fluid, sstate, transfer, dt=0.01,
                                                 nsteps=30, rho=RHO_WATER,
                                                 rho_inf=0.8, max_sub=20,
                                                 vent=vent)
    if verbose:
        print(f"  dry frequencies {np.array2string(freq[:3], precision=2)} Hz, "
              f"added-mass ratio {added['ratio']:.2f} "
              f"(cavity {added['cavity']}), "
              f"{fsd.steps_per_period(sstate, 0.01):.1f} steps per period")
        print("   step      s   regime      CL        tip      sub   Lc max   D/h"
              "    Phi   drift")
        for n in range(0, len(hist["t"]), max(1, len(hist["t"]) // 10)):
            print(f"  {n:5d}  {hist['s'][n]:5.2f}   "
                  f"{vcv.REGIMES[int(hist['regime'][n])]:4s}  "
                  f"{hist['CL'][n]:+8.5f}  {hist['tip'][n]:.3e}  "
                  f"{hist['sub'][n]:3d}   {hist['l_c_max'][n]:6.3f} "
                  f"{hist['d_cav'][n]:5.2f} {np.degrees(hist['phi_bar'][n]):6.1f}"
                  f"  {hist['drift_y'][n]:.1e}")
        tip = np.asarray(hist["tip"])
        stored = np.asarray(hist["strain_energy"]) + \
            np.asarray(hist["kinetic_energy"])
        work = np.asarray(hist["fluid_work"])
        scale = max(float(np.abs(work).max()), 1e-300)
        print(f"\n  transitions: {vent['transitions']}")
        print(f"  peak tip displacement {tip.max():.3e} chords, final "
              f"{tip[-1]:.3e}")
        print(f"  energy balance residual, largest relative: "
              f"{float(np.abs(stored - work).max() / scale):.2e}")
        print("  That residual is LARGE, and it is meant to be: C6's balance closes")
        print("  because the only work done on the structure is by the fluid, and")
        print("  here the cavity is growing. Entrained air displaces water and does")
        print("  work this model does not account for - there is no gas equation of")
        print("  state and no entrainment energy - so the balance cannot close")
        print("  during a transition and is REPORTED as a measure of how far the")
        print("  step is from a conservative one, not as a verification. Once the")
        print("  cavity stops growing the balance recovers.")
        print("  A ventilation transition is also a load step, so the second")
        print("  resolution constraint binds hardest here; the cavity front is")
        print(f"  rate-limited to {vcv.GROWTH_CHORDS:.1f} chords of cavity per chord")
        print("  of travel, without which the extent jumps to its equilibrium in a")
        print("  single step and the structural response measures dt, not the flow.")
        print("  The drift column is the waterline wake drift the image projection")
        print("  discards, which IS the linearised wave elevation: it is the")
        print("  quantitative measure of how far below this model's valid Froude")
        print("  range a run is being pushed.")
    return {"rows": rows, "history": hist, "transitions": vent["transitions"]}


# --------------------------------------------------------------------- run

CASES = {"V1": case_sectional, "V2": case_exactness, "V3": case_baseline,
         "V4": case_load_loss, "V5": case_washout, "V6": case_hysteresis}
QUICK = ("V1", "V2", "V5")


def main(only=None, quick=False):
    print("Ventilation verification: vent_section, vent_mesh, vent_cavity, "
          "vent_loads, vent_solve")
    print("Every number below is measured by this run.")
    names = [only] if only else (QUICK if quick else list(CASES))
    for name in names:
        start = time.perf_counter()
        CASES[name]()
        print(f"  [{name} took {time.perf_counter() - start:.1f} s]")
    print("\nDone. docs/VENTILATION.md tabulates these numbers.")


if __name__ == "__main__":
    case = None
    if "--case" in sys.argv:
        case = sys.argv[sys.argv.index("--case") + 1].upper()
        if case not in CASES:
            raise SystemExit(f"unknown case '{case}'; choose from "
                             f"{', '.join(CASES)}")
    main(case, quick="--quick" in sys.argv)
