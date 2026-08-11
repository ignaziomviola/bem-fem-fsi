"""Verification of the time-dependent driver against exact unsteady solutions.

Three cases, each isolating one part of the formulation:

A  added mass    a sphere accelerated in still fluid, no wake and no
                 circulation, against m_a = (2/3) pi rho a^3. This tests the
                 dmu/dt pressure term alone: a wrong sign or a wrong factor
                 cannot survive it, and neither can a wrong body-motion
                 transfer, since the whole force comes from the motion.
B  Wagner        the indicial lift of a wing started impulsively, against
                 Wagner's function in the two-term approximation of R. T.
                 Jones. This tests the shedding rule: how much circulation
                 leaves the trailing edge at each step, and where it goes.
C  Theodorsen    the harmonic lift in plunge, against Theodorsen's C(k) with
                 the added-mass term. This tests both together, and the phase
                 is the part a plausible but wrong scheme gets wrong.

Cases B and C are two-dimensional theories. They are run on a wing of aspect
ratio 20 and read at the midspan section, and every comparison is normalised by
the code's OWN steady section lift slope, measured on the same mesh. That
removes the finite-span deficit, the thickness correction and the discretisation
error from the comparison, leaving the unsteady content, which is what is under
test. The residual three-dimensionality is quantified in each case.

Run:
    python3 verify_unsteady.py            # added mass, steady limit, B and C
    python3 verify_unsteady.py --refine   # adds the convergence and sensitivity
                                          #   studies that diagnose the residual

The default programme takes about an hour and --refine about two, because the
cost of a step grows with the number of wake rows and the harmonic cases march
four cycles. Import the module and call one case when that is what is wanted:

    import verify_unsteady as vu
    vu.case_added_mass()                       # seconds
    vu.case_theodorsen(k=0.2, cycles=2)        # minutes

Results are tabulated in docs/UNSTEADY.md, which states the configuration of
every number so that any single case can be reproduced on its own.
"""

import sys

import numpy as np

import thick_panel_wing as tpw
import unsteady_wing as usw
import make_thick_sample_inputs as mts
from freewake_kernels import pitch_mesh

RHO = 1.0


def uniform_onset(points):
    pts = np.atleast_2d(points)
    v = np.zeros((len(pts), 3))
    v[:, 0] = 1.0
    return v


def still_fluid(points):
    return np.zeros((len(np.atleast_2d(points)), 3))


# ------------------------------------------------------- analytical references

def wagner(s):
    """Wagner's indicial lift function in the approximation of R. T. Jones.

    s is the reduced time 2 U t / c. The circulatory lift of a flat plate given
    a step change of incidence at s = 0 is C_L(s) = C_L(inf) * wagner(s), with
    wagner(0) = 0.5 and wagner(inf) = 1.
    """
    s = np.asarray(s, dtype=float)
    return 1.0 - 0.165 * np.exp(-0.0455 * s) - 0.335 * np.exp(-0.3 * s)


def theodorsen(k):
    """Theodorsen's function C(k), the Fourier transform of Wagner's.

    Obtained from the same Jones coefficients, C(k) = i k times the transform
    of wagner(s), so the two references are consistent by construction:
    C(k) = 1 - 0.165 i k/(0.0455 + i k) - 0.335 i k/(0.3 + i k),
    with C(0) = 1 and C(inf) = 1/2. Using Hankel functions instead would need
    scipy, which this repository deliberately does not depend on; the Jones
    form differs from the exact C(k) by under 1% in modulus and under half a
    degree in phase over 0 < k < 1.
    """
    ik = 1j * np.asarray(k, dtype=float)
    return 1.0 - 0.165 * ik / (0.0455 + ik) - 0.335 * ik / (0.3 + ik)


def plunge_lift_coefficient(k, h_over_c, omega, t, slope):
    """C_L(t) of a plunging plate, Theodorsen, on the section chord.

    h(t) = h0 sin(omega t), positive upwards, so the effective incidence is
    -h_dot/U and the circulatory lift follows it through C(k). With b = c/2,
    L = -pi rho b^2 h_ddot - slope rho U b C(k) h_dot, where slope replaces the
    thin-aerofoil 2 pi by the lift slope of the section actually being modelled;
    the added-mass term is geometric and is not rescaled.
    """
    ck = theodorsen(k)
    h0 = h_over_c
    # C_L = 2 pi k^2 h0 sin(omega t) - 2 slope k h0 |C| cos(omega t + arg C)
    added = 2.0 * np.pi * k ** 2 * h0 * np.sin(omega * t)
    circ = -2.0 * slope * k * h0 * np.abs(ck) * np.cos(omega * t
                                                       + np.angle(ck))
    return added + circ


# ------------------------------------------------------------------- helpers

def sphere_wrap_mesh(nwrap=20, nspan=20, radius=1.0):
    """Wrap-format sphere: sections at constant y, poles pinched to points."""
    pts = np.zeros((nwrap + 1, nspan + 1, 3))
    for j in range(nspan + 1):
        r = radius * np.sin(np.pi * j / nspan)
        ang = 2.0 * np.pi * np.arange(nwrap + 1) / nwrap
        pts[:, j, 0] = r * np.cos(ang)
        pts[:, j, 2] = -r * np.sin(ang)
        pts[:, j, 1] = -radius * np.cos(np.pi * j / nspan)
    pts[-1] = pts[0]
    for jt in (0, nspan):
        for i in range(nwrap // 2 + 1):
            pts[nwrap - i, jt] = pts[i, jt]
    return pts


def quasi2d_wing(n_c=16, nspan=12, span=20.0, thickness=0.08):
    """Rectangular wing of unit chord and large aspect ratio, cosine spanwise."""
    return mts.thick_wing_mesh(n_c=n_c, nspan=nspan, span=span, root_chord=1.0,
                               taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                               camber=0.0, thickness=thickness,
                               cosine_span=True)


def midspan_section_slope(points, onset, alpha_deg=5.0, u_ref=1.0):
    """Steady section lift slope at midspan, per radian, on this mesh.

    Measured with the free wake, so it carries the same downwash the unsteady
    march will have; every two-dimensional comparison is normalised by it.
    """
    pitched = pitch_mesh(points, np.radians(alpha_deg))
    state = tpw.init_state(pitched, onset, u_ref)
    state = tpw.solve(state)
    loads = tpw.get_loads(state, rho=RHO)
    y, cl = tpw.spanwise_loading(state["panels"], loads["p_gauge"], u_ref, RHO)
    jm = len(cl) // 2
    cl_mid = 0.5 * (cl[jm] + cl[-1 - jm])
    return cl_mid / np.radians(alpha_deg), cl_mid, state


# ------------------------------------------------------------------ case A

def case_added_mass(nwrap=20, nspan=20, radius=1.0, accel=1.0, dt=0.05,
                    nsteps=6, verbose=True):
    """Sphere accelerated from rest in still fluid: force against -m_a a.

    The flow is acyclic and wakeless, so the entire force is -rho dmu/dt
    integrated over the surface. mu is linear in the body speed and the speed
    is linear in time, so dmu/dt is constant and the backward difference is
    exact: any error is the panelling of the sphere, not the time scheme.
    """
    points = sphere_wrap_mesh(nwrap, nspan, radius)
    motion = usw.accelerate(points, (accel, 0.0, 0.0))
    history, state = usw.time_march(points, still_fluid, motion, dt=dt,
                                    nsteps=nsteps, u_ref=1.0, rho=RHO,
                                    lifting=False)
    fx = history["force"][-1, 0]
    exact = -(2.0 / 3.0) * np.pi * RHO * radius ** 3 * accel
    err = abs(fx - exact) / abs(exact)
    if verbose:
        print(f"A  added mass, sphere {nwrap}x{nspan}")
        print(f"   force  {fx:+.5f}   exact -m_a a = {exact:+.5f}   "
              f"error {100 * err:.2f}%")
        drift = np.abs(history["force"][2:, 0] - fx).max()
        print(f"   steady in time to {drift:.2e} over the last "
              f"{nsteps - 1} steps (constant acceleration)")
    return fx, exact, err


# ------------------------------------------------------------------ case B

def case_wagner(n_c=16, nspan=12, span=20.0, alpha_deg=5.0, dt=0.05,
                nsteps=120, thickness=0.08, core=0.02, verbose=True):
    """Impulsive start: indicial lift against Wagner's function.

    The wing is held at incidence in a uniform onset with no wake at t = 0,
    which is the impulsive start seen from the fluid. The lift is normalised by
    the steady value of the same mesh, so what is compared is the SHAPE of the
    growth. The reference is two-dimensional; the residual three-dimensionality
    of an aspect-ratio-20 wing is reported as the deficit of the steady value
    against the two-dimensional slope.
    """
    points = pitch_mesh(quasi2d_wing(n_c, nspan, span, thickness),
                        np.radians(alpha_deg))
    steady = tpw.init_state(points, uniform_onset, 1.0)
    steady = tpw.solve(steady)
    cl_inf = tpw.get_loads(steady, rho=RHO)["resultants"]["CL"]
    pan = steady["panels"]
    mu_w_inf = float((steady["mu"][pan["k_up"]]
                      - steady["mu"][pan["k_low"]])[pan["nspan"] // 2])
    start = usw.init_unsteady_state(points, uniform_onset, 1.0, core=core)
    history, state = usw.time_march(points, uniform_onset, dt=dt,
                                    nsteps=nsteps, u_ref=1.0, rho=RHO,
                                    state=start)
    s = history["s"]
    ratio = history["CL"] / cl_inf
    ref = wagner(s)
    # the first step carries the impulsive added-mass spike, which Wagner's
    # circulatory function excludes by construction; compare from s = 1
    sel = s >= 1.0
    rms = float(np.sqrt(np.mean((ratio[sel] - ref[sel]) ** 2)))
    # the bound circulation is the cleaner comparison: it is what the shedding
    # rule sets, free of the pressure model and of the added-mass term
    circ = history["mu_w_mid"] / mu_w_inf
    rms_circ = float(np.sqrt(np.mean((circ[sel] - ref[sel]) ** 2)))
    i2 = int(np.argmin(np.abs(s - 2.0)))
    out = {"s": s, "ratio": ratio, "circulation": circ, "wagner": ref,
           "rms": rms, "rms_circulation": rms_circ,
           "cl_inf": cl_inf, "mu_w_inf": mu_w_inf,
           "cl_ratio_at_s2": float(ratio[i2]),
           "wagner_at_s2": float(ref[i2]),
           "final_ratio": float(ratio[-1]), "history": history,
           "state": state}
    if verbose:
        print(f"B  Wagner, {2 * n_c}x{nspan} panels, aspect ratio "
              f"{span:.0f}, alpha {alpha_deg} deg, dt U/c = {dt:.3f}")
        print(f"   steady C_L {cl_inf:.4f}; marched to s = {s[-1]:.1f}, "
              f"reaching {100 * ratio[-1]:.1f}% of it "
              f"(Wagner {100 * ref[-1]:.1f}%)")
        print(f"   root mean square difference over 1 <= s <= {s[-1]:.0f}: "
              f"lift {rms:.4f}, bound circulation {rms_circ:.4f}")
        print("     s     lift   circulation   Wagner")
        for target in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0):
            if target > s[-1]:
                break
            i = int(np.argmin(np.abs(s - target)))
            print(f"   {s[i]:5.2f}   {ratio[i]:6.3f}   {circ[i]:11.3f}   "
                  f"{ref[i]:6.3f}")
    return out


# ------------------------------------------------------------------ case C

def case_theodorsen(k=0.2, n_c=16, nspan=12, span=20.0, h_over_c=0.02,
                    cycles=4, steps_per_cycle=32, thickness=0.08, core=0.02,
                    slope=None, verbose=True):
    """Plunge at reduced frequency k: amplitude and phase against Theodorsen.

    The section lift slope of the same mesh replaces the thin-aerofoil 2 pi in
    the circulatory term, so the comparison isolates the unsteady transfer
    function. The added-mass term is geometric and is not rescaled. The first
    harmonic is fitted over the last whole cycles, which removes the starting
    transient.
    """
    points = quasi2d_wing(n_c, nspan, span, thickness)
    chord = float(tpw.section_chords(points).mean())
    u_ref = 1.0
    omega = 2.0 * k * u_ref / chord
    if slope is None:
        slope, _, _ = midspan_section_slope(points, uniform_onset, 5.0, u_ref)
    motion = usw.heave(points, h_over_c * chord, omega)
    dt = 2.0 * np.pi / omega / steps_per_cycle
    nsteps = int(cycles * steps_per_cycle)
    start = usw.init_unsteady_state(points, uniform_onset, u_ref, core=core,
                                    point_velocities=motion(0.0)[1])
    history, state = usw.time_march(points, uniform_onset, motion, dt=dt,
                                    nsteps=nsteps, u_ref=u_ref, rho=RHO,
                                    state=start)
    t = history["t"]
    _, amp, phase = usw.first_harmonic(t, history["CL"], omega, cycles=2)
    ref = plunge_lift_coefficient(k, h_over_c, omega, t, slope)
    _, amp_ref, phase_ref = usw.first_harmonic(t, ref, omega, cycles=2)
    d_phase = np.degrees(np.angle(np.exp(1j * (phase - phase_ref))))
    out = {"k": k, "slope": slope, "core": core, "amplitude": amp, "amplitude_ref": amp_ref,
           "amplitude_ratio": amp / amp_ref, "phase_deg": np.degrees(phase),
           "phase_ref_deg": np.degrees(phase_ref), "phase_error_deg": d_phase,
           "history": history, "reference": ref, "state": state}
    if verbose:
        print(f"C  Theodorsen plunge, k = {k:.2f}, h0/c = {h_over_c}, "
              f"{cycles} cycles at {steps_per_cycle} steps each")
        print(f"   section lift slope of this mesh {slope:.3f} /rad "
              f"({slope / (2 * np.pi):.3f} of 2 pi)")
        print(f"   |C_L| code {amp:.5f}   theory {amp_ref:.5f}   "
              f"ratio {amp / amp_ref:.4f}")
        print(f"   phase code {np.degrees(phase):+7.2f} deg   theory "
              f"{np.degrees(phase_ref):+7.2f} deg   difference "
              f"{d_phase:+.2f} deg")
    return out


# -------------------------------------------------------- time-step refinement

def case_dt_refinement(n_c=12, nspan=12, span=20.0, alpha_deg=5.0, k=0.2,
                       verbose=True):
    """The two unsteady cases under time refinement.

    Halving dt halves the convection per step and doubles the number of wake
    rows, so this measures the whole scheme, not the difference formula alone.
    Both discrepancies fall monotonically, at a fractional observed order: the
    leading error of the scheme is that the vorticity shed during a step is
    carried at the strength of the END of the step and therefore acts from the
    downstream edge of the new strip rather than from within it, which places
    it too far from the wing and so overstates the lift and understates the
    phase lag - the sign of both discrepancies.
    """
    rows = []
    for dt in (0.2, 0.1, 0.05):
        res = case_wagner(n_c=n_c, nspan=nspan, span=span, alpha_deg=alpha_deg,
                          dt=dt, nsteps=int(round(8.0 / dt)), verbose=False)
        i = int(np.argmin(np.abs(res["s"] - 8.0)))
        rows.append((dt, res["ratio"][i], res["rms"]))
    harmonic = []
    points = quasi2d_wing(n_c, nspan, span)
    slope, _, _ = midspan_section_slope(points, uniform_onset)
    for spc in (16, 32, 64):
        r = case_theodorsen(k=k, n_c=n_c, nspan=nspan, span=span, cycles=3,
                            steps_per_cycle=spc, slope=slope, verbose=False)
        harmonic.append((spc, r["amplitude_ratio"], r["phase_error_deg"]))
    if verbose:
        print("D  time refinement")
        print("   Wagner case, read at s = 8")
        print("   dt U/c   C_L/C_L(inf)   rms vs Wagner")
        for dt, r, rms in rows:
            print(f"   {dt:6.3f}   {r:12.4f}   {rms:13.4f}")
        print(f"   plunge case, k = {k}")
        print("   steps/cycle   amplitude ratio   phase error")
        for spc, amp, ph in harmonic:
            for_ = f"{spc:11d}   {amp:15.4f}   {ph:+8.2f} deg"
            print(f"   {for_}")
        d1 = abs(harmonic[1][1] - harmonic[0][1])
        d2 = abs(harmonic[2][1] - harmonic[1][1])
        if d2 > 0:
            print(f"   observed order of the amplitude discrepancy "
                  f"{np.log2(max(d1, 1e-15) / d2):.2f}")
    return rows, harmonic


def case_steady_limit(n_c=12, nspan=12, span=8.0, alpha_deg=5.0, dt=0.25,
                      nsteps=80, thickness=0.12, verbose=True):
    """A stationary wing marched until the wake settles, against the steady
    driver on the same mesh.

    This is the closure test of the whole loop rather than a comparison with a
    theory: shedding, convection, the Kutta fold and the pressure must together
    reproduce the answer the relaxation driver reaches by a different route.
    The two wakes differ - one is convected, the other relaxed - so the
    comparison is against both the free and the frozen steady wake.
    """
    points = pitch_mesh(mts.thick_wing_mesh(
        n_c=n_c, nspan=nspan, span=span, root_chord=1.0, taper=1.0,
        sweep_deg=0.0, twist_deg=0.0, camber=0.0, thickness=thickness),
        np.radians(alpha_deg))
    history, state = usw.time_march(points, uniform_onset, dt=dt,
                                    nsteps=nsteps, u_ref=1.0, rho=RHO)
    free = tpw.solve(tpw.init_state(points, uniform_onset, 1.0))
    cl_free = tpw.get_loads(free, rho=RHO)["resultants"]["CL"]
    frozen = tpw.init_state(points, uniform_onset, 1.0,
                            config={"nwake": nsteps,
                                    "wake_length_spans": nsteps * dt / span})
    frozen = tpw.solve(frozen, wake="frozen")
    cl_frozen = tpw.get_loads(frozen, rho=RHO)["resultants"]["CL"]
    cl = history["CL"][-1]
    if verbose:
        print(f"S  steady limit, aspect ratio {span:.0f}, {2 * n_c}x{nspan} "
              f"panels, alpha {alpha_deg} deg, dt U/c = {dt}, {nsteps} steps")
        print(f"   marched to s = {history['s'][-1]:.0f}: C_L {cl:.5f}")
        print(f"   steady relaxed wake  {cl_free:.5f}   ratio "
              f"{cl / cl_free:.4f}")
        print(f"   steady frozen wake, same length  {cl_frozen:.5f}   ratio "
              f"{cl / cl_frozen:.4f}")
        print(f"   shed strip strength monotone: "
              f"{bool(np.all(np.diff(history['mu_w_mid'][1:]) > -1e-9))}")
    return {"cl": cl, "cl_free": cl_free, "cl_frozen": cl_frozen,
            "history": history, "state": state}


def case_aspect_ratio(k=0.2, verbose=True):
    """The plunge discrepancy against aspect ratio: is it three-dimensional?"""
    rows = []
    for span, nspan in ((20.0, 12), (40.0, 16), (80.0, 20)):
        r = case_theodorsen(k=k, n_c=12, nspan=nspan, span=span, cycles=3,
                            steps_per_cycle=24, verbose=False)
        rows.append((span, r["slope"], r["amplitude_ratio"],
                     r["phase_error_deg"]))
    if verbose:
        print(f"E  aspect-ratio dependence of the plunge discrepancy, "
              f"k = {k}, 24 steps per cycle")
        print("   AR   slope   amplitude ratio   phase error")
        for span, slope, amp, ph in rows:
            print(f"   {span:3.0f}   {slope:5.3f}   {amp:15.4f}   "
                  f"{ph:+8.2f} deg")
    return rows


def case_core(k=0.2, verbose=True):
    """The plunge response against the wake core radius: it does not depend on
    it, because the wake reaches the body through the core-free potential."""
    rows = []
    for core in (0.3333, 0.05, 0.01):
        r = case_theodorsen(k=k, n_c=12, nspan=12, span=20.0, cycles=3,
                            steps_per_cycle=24, core=core, verbose=False)
        rows.append((core, r["amplitude_ratio"], r["phase_error_deg"]))
    if verbose:
        print(f"F  wake core radius, k = {k}, 24 steps per cycle")
        print("   core/c   amplitude ratio   phase error")
        for core, amp, ph in rows:
            print(f"   {core:6.4f}   {amp:15.4f}   {ph:+8.2f} deg")
    return rows


# ---------------------------------------------------------------------- run

def main(refine=False):
    print("Verification of the time-dependent driver "
          "(rho = 1, U_ref = 1, chord = 1)\n")
    case_added_mass()
    print()
    case_steady_limit()
    print()
    case_wagner(n_c=12, nsteps=160, dt=0.05)
    print()
    points = quasi2d_wing(12, 12, 20.0, 0.08)
    slope, _, _ = midspan_section_slope(points, uniform_onset)
    for k in (0.1, 0.2, 0.4):
        case_theodorsen(k=k, n_c=12, nspan=12, span=20.0, cycles=4,
                        steps_per_cycle=48, slope=slope)
        print()
    if refine:
        case_dt_refinement()
        print()
        case_aspect_ratio()
        print()
        case_core()


if __name__ == "__main__":
    main(refine="--refine" in sys.argv)
