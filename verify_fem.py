"""Structural verification against closed-form solutions.

    python3 verify_fem.py            every case, about a minute
    python3 verify_fem.py --case F4  one case by name

Nine cases, each with an exact or classical reference. The tables printed here
are the ones tabulated in docs/FEM.md, and those tables hold what this script
measured, not what it was hoped to measure. Nothing is written and no figure is
created.

F1 patch test               constant strain on a distorted mesh
F2 rigid-body modes         K u = 0 and six zero eigenvalues
F3 mass properties          mass, centre of mass and inertia of a box
F4 cantilever deflection    Euler-Bernoulli with the Timoshenko shear term
F5 natural frequencies      the first three flapwise modes
F6 anisotropy               orthotropic reduction and tensor rotation
F7 viscoelastic decay       Kelvin-Voigt against the analytical envelope
F8 time integration         generalised-alpha period elongation and dissipation
F9 locking                  incompatible modes on and off
"""

import sys

import numpy as np

import fem_materials as fmat
import fem_mesh
import fem_solid as fes

ALU = dict(e_mod=70.0e9, nu=0.3, rho=2700.0)
STEEL = dict(e_mod=210.0e9, nu=0.3, rho=7800.0)
BEAM = dict(length=1.0, width=0.1, thick=0.02)


def _rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def _cantilever(n_length, n_thick=2, material=None, incompatible=True):
    mesh = fem_mesh.box_mesh(BEAM["length"], BEAM["width"], BEAM["thick"],
                             n_length, 1, n_thick)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            material or fmat.IsotropicElastic(**ALU),
                            incompatible=incompatible)
    sstate = fes.init_state(model)
    fes.clamp(sstate, mesh["node_sets"]["x_min"])
    fes.assemble_operators(sstate)
    return mesh, model, sstate


def beam_constants(material=None):
    mat = material or fmat.IsotropicElastic(**ALU)
    inertia = BEAM["width"] * BEAM["thick"] ** 3 / 12.0
    area = BEAM["width"] * BEAM["thick"]
    kappa = 5.0 * (1.0 + mat.nu) / (6.0 + 5.0 * mat.nu)
    return mat, inertia, area, kappa


# ------------------------------------------------------------------ case F1

def case_patch_test(verbose=True):
    """Constant strain reproduced exactly on a randomly distorted mesh."""
    grad = np.array([[1.0e-4, 2.0e-5, -1.0e-5], [3.0e-5, -3.0e-5, 2.0e-5],
                     [-2.0e-5, 1.0e-5, -4.0e-5]])
    mat = fmat.IsotropicElastic(**STEEL)
    exact_stress = mat.tangent() @ fmat.tensor_to_voigt(0.5 * (grad + grad.T))
    out = {}
    for incompatible in (True, False):
        mesh = fem_mesh.distort_interior(
            fem_mesh.box_mesh(1.0, 1.0, 1.0, 3, 3, 3), 0.35, 1)
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat,
                                incompatible=incompatible)
        sstate = fes.init_state(model)
        boundary = np.unique(np.concatenate(
            [mesh["node_sets"][k] for k in ("x_min", "x_max", "y_min", "y_max",
                                            "z_min", "z_max")]))
        fes.clamp(sstate, boundary)
        fes.assemble_operators(sstate)
        free = fes.free_mask(sstate)
        u_exact = mesh["nodes"] @ grad.T
        u = np.zeros(len(free))
        u[~free] = u_exact.reshape(-1)[~free]
        u[free] = np.linalg.solve(sstate["K"][np.ix_(free, free)],
                                  -sstate["K"][np.ix_(free, ~free)] @ u[~free])
        _, _, stress = fes.internal_force(model, u.reshape(-1, 3))
        out[incompatible] = (
            float(np.abs(u - u_exact.reshape(-1)).max() / np.abs(u_exact).max()),
            float(np.abs(stress - exact_stress).max()
                  / np.abs(exact_stress).max()))
    if verbose:
        _rule("F1 patch test: constant strain on a distorted 3x3x3 mesh")
        print("  incompatible modes   displacement error   stress error")
        for key, (d_err, s_err) in out.items():
            print(f"  {'on' if key else 'off':<19s}  {d_err:18.2e}   {s_err:12.2e}")
        print("  Both are at round-off. The Taylor-Beresford-Wilson correction is")
        print("  what keeps the incompatible element passing; without it the")
        print("  bubbles survive a linear field and the errors are O(1e-2).")
    return out


# ------------------------------------------------------------------ case F2

def case_rigid_body(verbose=True):
    """Six rigid-body modes carry no strain energy, and K is symmetric."""
    mesh = fem_mesh.distort_interior(fem_mesh.box_mesh(1.0, 1.0, 1.0, 3, 3, 3),
                                     0.35, 1)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(**STEEL))
    k_mat = fes.assemble_stiffness(model)
    scale = np.abs(k_mat).max()
    pts = mesh["nodes"]
    residuals = {}
    for name, mode in (("translation x", np.tile([1.0, 0, 0], (len(pts), 1))),
                       ("translation y", np.tile([0, 1.0, 0], (len(pts), 1))),
                       ("translation z", np.tile([0, 0, 1.0], (len(pts), 1))),
                       ("rotation x", np.cross(np.tile([1.0, 0, 0], (len(pts), 1)), pts)),
                       ("rotation y", np.cross(np.tile([0, 1.0, 0], (len(pts), 1)), pts)),
                       ("rotation z", np.cross(np.tile([0, 0, 1.0], (len(pts), 1)), pts))):
        residuals[name] = float(np.abs(k_mat @ mode.reshape(-1)).max()
                                / (scale * np.abs(mode).max()))
    eigs = np.linalg.eigvalsh(k_mat)
    out = {"residuals": residuals,
           "symmetry": float(np.abs(k_mat - k_mat.T).max() / scale),
           "zero_modes": int(np.sum(eigs < 1e-9 * eigs.max())),
           "eig_ratio": float(np.abs(eigs[:8]).max() / eigs.max())}
    if verbose:
        _rule("F2 rigid-body modes and the symmetry of K")
        for name, res in residuals.items():
            print(f"  |K u| / |K| |u|, {name:<14s} {res:.2e}")
        print(f"  symmetry |K - K^T| / |K|          {out['symmetry']:.2e}")
        print(f"  eigenvalues below 1e-9 of the largest: {out['zero_modes']} "
              f"(six expected)")
    return out


# ------------------------------------------------------------------ case F3

def case_mass_properties(verbose=True):
    """Mass, centre of mass and inertia tensor of a box against the closed form."""
    l_x, l_y, l_z = 2.0, 3.0, 4.0
    mesh = fem_mesh.box_mesh(l_x, l_y, l_z, 2, 3, 2)
    mat = fmat.IsotropicElastic(**STEEL)
    model = fes.build_model(mesh["nodes"], mesh["elements"], mat)
    prop = fes.mass_properties(model)
    mass = mat.rho * l_x * l_y * l_z
    inertia = mass / 12.0 * np.array([l_y ** 2 + l_z ** 2, l_x ** 2 + l_z ** 2,
                                      l_x ** 2 + l_y ** 2])
    out = {"mass_error": abs(prop["mass"] / mass - 1.0),
           "centre_error": float(np.abs(prop["centre"]
                                        - 0.5 * np.array([l_x, l_y, l_z])).max()),
           "inertia_error": float(np.abs(np.diag(prop["inertia"]) - inertia).max()
                                  / inertia.max()),
           "off_diagonal": float(np.abs(prop["inertia"]
                                        - np.diag(np.diag(prop["inertia"]))).max()
                                 / inertia.max())}
    if verbose:
        _rule("F3 mass properties of a 2 x 3 x 4 box")
        print(f"  mass                {prop['mass']:.6f} against "
              f"{mass:.6f}, error {out['mass_error']:.2e}")
        print(f"  centre of mass      error {out['centre_error']:.2e}")
        print(f"  principal inertia   error {out['inertia_error']:.2e}")
        print(f"  off-diagonal terms  {out['off_diagonal']:.2e}")
        print("  Exact, because the consistent mass of a parallelepiped mesh")
        print("  integrates a quadratic exactly on the 2x2x2 rule.")
    return out


# ------------------------------------------------------------------ case F4

def case_cantilever(refine=(4, 8, 16, 32), verbose=True):
    """Tip deflection against Euler-Bernoulli with the Timoshenko shear term."""
    load = -100.0
    mat, inertia, area, kappa = beam_constants()
    euler = load * BEAM["length"] ** 3 / (3.0 * mat.e_mod * inertia)
    timoshenko = euler + load * BEAM["length"] / (kappa * mat.shear * area)
    rows = []
    for n in refine:
        mesh, _, sstate = _cantilever(n)
        f_ext = np.zeros_like(sstate["u"])
        tip = mesh["node_sets"]["x_max"]
        f_ext[tip, 2] = load / len(tip)
        level = fes.solve_static(sstate, f_ext)
        deflection = float(level["u"][tip, 2].mean())
        rows.append((n, deflection, deflection / timoshenko))
    if verbose:
        _rule("F4 cantilever tip deflection, 1.0 x 0.1 x 0.02 m, aluminium, 100 N")
        print(f"  Euler-Bernoulli PL^3/3EI          {euler:+.6e} m")
        print(f"  with the Timoshenko shear term    {timoshenko:+.6e} m")
        print("  elements along the length   deflection        ratio")
        for n, deflection, ratio in rows:
            print(f"  {n:>25d}   {deflection:+.6e}   {ratio:.4f}")
        print("  Monotone from below and converging. The remaining deficit is")
        print("  the clamped end face, which restrains the Poisson contraction")
        print("  and the warping that the beam solution allows.")
    return {"euler": euler, "timoshenko": timoshenko, "rows": rows}


# ------------------------------------------------------------------ case F5

def case_frequencies(n_length=24, verbose=True):
    """The first three flapwise frequencies of the same cantilever."""
    mesh, _, sstate = _cantilever(n_length)
    mat, inertia, area, _ = beam_constants()
    freq, shapes = fes.modes(sstate, 12)
    tip = mesh["node_sets"]["x_max"]
    flap = [j for j in range(len(freq))
            if np.abs(shapes[tip, 2, j]).mean()
            > 3.0 * np.abs(shapes[tip, 1, j]).mean()]
    scale = np.sqrt(mat.e_mod * inertia
                    / (mat.rho * area * BEAM["length"] ** 4)) / (2.0 * np.pi)
    rows = []
    for j, beta in zip(flap[:3], (1.875104, 4.694091, 7.854757)):
        exact = beta ** 2 * scale
        rows.append((freq[j], exact, freq[j] / exact))
    if verbose:
        _rule(f"F5 flapwise natural frequencies, {n_length} elements")
        print("  mode   finite element      Euler-Bernoulli      ratio")
        for n, (measured, exact, ratio) in enumerate(rows, start=1):
            print(f"  {n:>4d}   {measured:14.4f} Hz   {exact:14.4f} Hz   "
                  f"{ratio:.4f}")
        print("  Above the beam reference, and by more in the higher modes: the")
        print("  clamped face is stiffer than a beam's built-in end, and the")
        print("  beam reference itself omits rotary inertia and shear.")
    return {"rows": rows, "all": freq}


# ------------------------------------------------------------------ case F6

def case_anisotropy(verbose=True):
    """The orthotropic material reduces to isotropy and rotates as a tensor."""
    e_mod, nu = ALU["e_mod"], ALU["nu"]
    shear = 0.5 * e_mod / (1.0 + nu)
    iso = fmat.IsotropicElastic(**ALU)
    equal = fmat.OrthotropicElastic(e_mod, e_mod, e_mod, shear, shear, shear,
                                    nu, nu, nu, ALU["rho"])
    reduction = float(np.abs(equal.tangent() - iso.tangent()).max()
                      / np.abs(iso.tangent()).max())
    carbon = fmat.OrthotropicElastic(150e9, 10e9, 10e9, 5e9, 3.5e9, 5e9,
                                     0.3, 0.3, 0.4, 1600.0)
    rng = np.random.default_rng(5)
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = basis * np.sign(np.linalg.det(basis))
    strain = rng.normal(size=6) * 1e-4
    t_mat = fmat.strain_rotation(rot)
    direct = carbon.tangent(rot) @ strain
    through = t_mat.T @ (carbon.tangent() @ (t_mat @ strain))
    covariance = float(np.abs(direct - through).max() / np.abs(direct).max())
    energy_global = 0.5 * float(strain @ (carbon.tangent(rot) @ strain))
    energy_material = 0.5 * float((t_mat @ strain)
                                  @ (carbon.tangent() @ (t_mat @ strain)))
    invariance = abs(energy_global / energy_material - 1.0)
    isotropic_rot = float(np.abs(iso.tangent(rot) - iso.tangent()).max()
                          / np.abs(iso.tangent()).max())
    if verbose:
        _rule("F6 anisotropy: the seam the material extension runs through")
        print("  orthotropic with equal constants reduces to isotropic  "
              f"{reduction:.2e}")
        print("  isotropic modulus is invariant under a rotation        "
              f"{isotropic_rot:.2e}")
        print("  carbon-like modulus rotates as D -> T^T D T            "
              f"{covariance:.2e}")
        print("  strain energy is invariant under the rotation          "
              f"{invariance:.2e}")
        print("  The rotation operator is built by pushing the six Voigt basis")
        print("  strains through the tensor transformation, so the engineering-")
        print("  shear factors take care of themselves.")
    return {"reduction": reduction, "covariance": covariance,
            "invariance": invariance, "isotropic_rotation": isotropic_rot}


# ------------------------------------------------------------------ case F7

def case_viscoelastic(taus=(1e-4, 3e-4), steps_per_period=128, cycles=3,
                      verbose=True):
    """Kelvin-Voigt decay against exp(-zeta omega t), zeta = tau omega / 2."""
    base = fmat.IsotropicElastic(**ALU)
    rows = []
    for tau in taus:
        material = fmat.KelvinVoigt(base, tau)
        mesh, _, sstate = _cantilever(12, material=material)
        freq, phi = fes.modes(sstate, 1)
        omega = 2.0 * np.pi * freq[0]
        zeta = 0.5 * tau * omega
        sstate["u"] = phi[:, :, 0] * 1e-3 / np.abs(phi[:, :, 0]).max()
        zero = np.zeros_like(sstate["u"])
        sstate["a"] = fes.initial_acceleration(sstate, zero)
        tip = mesh["node_sets"]["x_max"]
        start = sstate["u"][tip, 2].mean()
        dt = 1.0 / (freq[0] * steps_per_period)
        for _ in range(cycles * steps_per_period):
            fes.commit(sstate, fes.step_dynamic(sstate, zero, dt, None), zero)
        measured = abs(sstate["u"][tip, 2].mean() / start)
        exact = float(np.exp(-zeta * omega * cycles / freq[0]))
        rows.append((tau, zeta, measured, exact, measured / exact))
    if verbose:
        _rule(f"F7 Kelvin-Voigt decay over {cycles} cycles at "
              f"{steps_per_period} steps per period")
        print("      tau      damping ratio    measured      exact     ratio")
        for tau, zeta, measured, exact, ratio in rows:
            print(f"  {tau:.1e}   {zeta:12.5f}   {measured:9.6f}  "
                  f"{exact:9.6f}   {ratio:.5f}")
        print("  Exact because the rate tangent tau*D assembles to C = tau*K,")
        print("  so the modal damping is exactly stiffness-proportional; the")
        print("  residual is the period elongation of the integrator.")
    return rows


# ------------------------------------------------------------------ case F8

def case_integrator(steps=(16, 32, 64, 128), spectral=(1.0, 0.9, 0.8, 0.5),
                    cycles=5, verbose=True):
    """Period elongation and numerical dissipation of generalised-alpha."""
    mesh, _, sstate = _cantilever(12)
    freq, phi = fes.modes(sstate, 1)
    period, omega = 1.0 / freq[0], 2.0 * np.pi * freq[0]
    tip = mesh["node_sets"]["x_max"]
    shape = phi[:, :, 0] * 1e-3 / np.abs(phi[:, :, 0]).max()

    def march(rho_inf, n_steps, start=None):
        state = fes.init_state(sstate["model"])
        fes.clamp(state, mesh["node_sets"]["x_min"])
        fes.assemble_operators(state)
        state["u"] = (shape if start is None else start).copy()
        zero = np.zeros_like(state["u"])
        state["a"] = fes.initial_acceleration(state, zero)
        par = fes.integrator_parameters(rho_inf)
        dt = period / n_steps
        times, tips = [0.0], [state["u"][tip, 2].mean()]
        for n in range(1, cycles * n_steps + 1):
            fes.commit(state, fes.step_dynamic(state, zero, dt, par), zero)
            times.append(n * dt)
            tips.append(state["u"][tip, 2].mean())
        return np.array(times), np.array(tips), dt

    elongation, dissipation = [], []
    for n_steps in steps:
        times, tips, dt = march(1.0, n_steps)
        cross = np.where(np.diff(np.sign(tips)) != 0)[0]
        zeros = times[cross] - tips[cross] * (times[cross + 1] - times[cross]) \
            / (tips[cross + 1] - tips[cross])
        measured = 2.0 * np.mean(np.diff(zeros))
        elongation.append((n_steps, measured / period - 1.0,
                           (omega * dt) ** 2 / 12.0, abs(tips[-1] / tips[0])))
    # an UNRESOLVED mode is what the dissipation is for: mode 8 at a step sized
    # for mode 1 sits far beyond the Nyquist limit of its own period
    freq_hi, phi_hi = fes.modes(sstate, 9)
    high = phi_hi[:, :, 8] * 1e-3 / np.abs(phi_hi[:, :, 8]).max()
    ratio_hi = freq_hi[8] / freq[0]
    for rho_inf in spectral:
        _, tips, _ = march(rho_inf, 32)
        _, tips_hi, _ = march(rho_inf, 32, start=high)
        dissipation.append((rho_inf, abs(tips[-1] / tips[0]),
                            abs(tips_hi[-1] / tips_hi[0])))
    if verbose:
        _rule("F8 generalised-alpha on a single resolved mode")
        print("  steps per period   T/T_exact - 1     (omega dt)^2/12    "
              f"amplitude after {cycles} cycles")
        for n_steps, measured, theory, amplitude in elongation:
            print(f"  {n_steps:>16d}   {measured:+.6e}     {theory:.6e}      "
                  f"{amplitude:.6f}")
        print("\n  numerical dissipation, dt sized for mode 1 at 32 steps per")
        print(f"  period, {cycles} cycles; mode 9 is {ratio_hi:.0f} times faster and")
        print("  therefore far beyond its own Nyquist limit")
        print("  rho_inf   resolved mode 1   unresolved mode 9")
        for rho_inf, low, hi in dissipation:
            print(f"  {rho_inf:>7.1f}   {low:15.6f}   {hi:17.6f}")
        print("  The elongation matches the leading term to four figures and")
        print("  falls at second order. The point of rho_inf is the right-hand")
        print("  column: it leaves a resolved mode alone and annihilates the")
        print("  unresolved one, which is what a coupled march needs when the")
        print("  structural spectrum reaches far above the step.")
    return {"elongation": elongation, "dissipation": dissipation}


# ------------------------------------------------------------------ case F9

def case_locking(refine=(4, 8, 16, 32), verbose=True):
    """The bending deficit of the trilinear hexahedron, with and without bubbles."""
    load = -100.0
    _, _, _, _ = beam_constants()
    reference = case_cantilever(refine=(), verbose=False)["timoshenko"]
    rows = []
    for n in refine:
        ratios = []
        for incompatible in (True, False):
            mesh, _, sstate = _cantilever(n, incompatible=incompatible)
            f_ext = np.zeros_like(sstate["u"])
            tip = mesh["node_sets"]["x_max"]
            f_ext[tip, 2] = load / len(tip)
            level = fes.solve_static(sstate, f_ext)
            ratios.append(level["u"][tip, 2].mean() / reference)
        rows.append((n, ratios[0], ratios[1]))
    if verbose:
        _rule("F9 shear locking: incompatible modes on and off")
        print("  elements   with bubbles   without   ratio between them")
        for n, with_b, without in rows:
            print(f"  {n:>8d}   {with_b:12.4f}   {without:7.4f}   "
                  f"{with_b / without:8.1f}")
        print("  Without the bubbles the element is an order of magnitude too")
        print("  stiff at the resolutions a coupled run can afford, and it")
        print("  converges from below only very slowly. This is the entire")
        print("  reason the incompatible modes are there.")
    return rows


# --------------------------------------------------------------------- run

CASES = {"F1": case_patch_test, "F2": case_rigid_body, "F3": case_mass_properties,
         "F4": case_cantilever, "F5": case_frequencies, "F6": case_anisotropy,
         "F7": case_viscoelastic, "F8": case_integrator, "F9": case_locking}


def main(only=None):
    print("Structural verification: fem_solid.py against closed-form solutions")
    print("Every number below is measured by this run.")
    for name, case in CASES.items():
        if only and name != only:
            continue
        case()
    print("\nDone. docs/FEM.md tabulates these numbers.")


if __name__ == "__main__":
    case = None
    if "--case" in sys.argv:
        case = sys.argv[sys.argv.index("--case") + 1].upper()
        if case not in CASES:
            raise SystemExit(f"unknown case '{case}'; choose from "
                             f"{', '.join(CASES)}")
    main(case)
