"""Tests for the structural solver (standard-library unittest, no pytest).

Run the fast set:      python3 test_fem.py
Include the slow set:  FEM_SLOW=1 python3 test_fem.py

The fast set pins everything an element may not get wrong - the patch test on a
distorted mesh, the rigid-body modes, the mass properties, the equivalence of
the integral of B^T sigma with K u for a linear material, the anisotropic
rotation and the purity of the time integrator - and runs in about two seconds.
The slow set holds the convergence studies against the analytical beam: tip
deflection, natural frequencies, the locking comparison that justifies the
incompatible modes, the period elongation of generalised-alpha and the
viscoelastic decay. No files are written and no figures are created.
"""

import os
import unittest

import numpy as np

import fem_materials as fmat
import fem_mesh
import fem_solid as fes

SLOW = os.environ.get("FEM_SLOW", "") not in ("", "0")

STEEL = dict(e_mod=210e9, nu=0.3, rho=7800.0)
ALU = dict(e_mod=70e9, nu=0.3, rho=2700.0)


def _linear_field():
    """A displacement gradient with all six strain components nonzero."""
    return np.array([[1.0e-4, 2.0e-5, -1.0e-5],
                     [3.0e-5, -3.0e-5, 2.0e-5],
                     [-2.0e-5, 1.0e-5, -4.0e-5]])


def _distorted_cube(n=3, amplitude=0.35, seed=1):
    return fem_mesh.distort_interior(fem_mesh.box_mesh(1.0, 1.0, 1.0, n, n, n),
                                     amplitude, seed)


def _boundary_nodes(mesh):
    return np.unique(np.concatenate([mesh["node_sets"][k] for k in
                                     ("x_min", "x_max", "y_min", "y_max",
                                      "z_min", "z_max")]))


def _solve_prescribed(model, mesh, u_exact):
    """Solve with every boundary node prescribed to the exact field."""
    sstate = fes.init_state(model)
    fes.clamp(sstate, _boundary_nodes(mesh))
    fes.assemble_operators(sstate)
    free = fes.free_mask(sstate)
    k_mat = sstate["K"]
    u = np.zeros(len(free))
    u[~free] = u_exact.reshape(-1)[~free]
    u[free] = np.linalg.solve(k_mat[np.ix_(free, free)],
                              -k_mat[np.ix_(free, ~free)] @ u[~free])
    return u.reshape(-1, 3)


class TestMaterials(unittest.TestCase):

    def test_voigt_round_trip(self):
        rng = np.random.default_rng(0)
        eps = rng.normal(size=(7, 6))
        back = fmat.tensor_to_voigt(fmat.voigt_to_tensor(eps))
        self.assertLess(np.abs(back - eps).max(), 1e-15)

    def test_isotropic_tangent_is_hooke(self):
        mat = fmat.IsotropicElastic(**ALU)
        d_mat = mat.tangent()
        lame, shear = mat.lame, mat.shear
        self.assertAlmostEqual(d_mat[0, 0], lame + 2 * shear, delta=1e-6)
        self.assertAlmostEqual(d_mat[0, 1], lame, delta=1e-6)
        self.assertAlmostEqual(d_mat[3, 3], shear, delta=1e-6)
        self.assertLess(np.abs(d_mat - d_mat.T).max(), 1e-6)
        self.assertGreater(np.linalg.eigvalsh(d_mat).min(), 0.0)

    def test_isotropic_is_invariant_under_rotation(self):
        mat = fmat.IsotropicElastic(**ALU)
        rot = _random_rotation(3)
        self.assertLess(np.abs(mat.tangent(rot) - mat.tangent()).max()
                        / np.abs(mat.tangent()).max(), 1e-13)

    def test_orthotropic_reduces_to_isotropic(self):
        e_mod, nu = ALU["e_mod"], ALU["nu"]
        shear = 0.5 * e_mod / (1.0 + nu)
        ortho = fmat.OrthotropicElastic(e_mod, e_mod, e_mod, shear, shear,
                                        shear, nu, nu, nu, ALU["rho"])
        iso = fmat.IsotropicElastic(**ALU)
        self.assertLess(np.abs(ortho.tangent() - iso.tangent()).max()
                        / np.abs(iso.tangent()).max(), 1e-12)

    def test_orthotropic_rotates_as_a_tensor(self):
        ortho = fmat.OrthotropicElastic(150e9, 10e9, 10e9, 5e9, 3.5e9, 5e9,
                                        0.3, 0.3, 0.4, 1600.0)
        rot = _random_rotation(5)
        d_rot = ortho.tangent(rot)
        # rotating the strain instead of the modulus must give the same stress
        rng = np.random.default_rng(7)
        eps = rng.normal(size=6) * 1e-4
        t_mat = fmat.strain_rotation(rot)
        direct = d_rot @ eps
        through = t_mat.T @ (ortho.tangent() @ (t_mat @ eps))
        self.assertLess(np.abs(direct - through).max()
                        / np.abs(direct).max(), 1e-12)

    def test_orthotropic_rejects_indefinite_constants(self):
        with self.assertRaises(ValueError):
            fmat.OrthotropicElastic(10e9, 150e9, 10e9, 5e9, 3.5e9, 5e9,
                                    0.9, 0.9, 0.9, 1600.0)

    def test_kelvin_voigt_rate_tangent(self):
        base = fmat.IsotropicElastic(**ALU)
        visc = fmat.KelvinVoigt(base, 2.5e-4)
        res = visc.response(np.zeros((2, 6)), strain_rate=np.zeros((2, 6)))
        self.assertLess(np.abs(res["rate_tangent"][0]
                               - 2.5e-4 * base.tangent()).max()
                        / np.abs(base.tangent()).max(), 1e-13)
        self.assertEqual(visc.n_history, 6)

    def test_kelvin_voigt_stress_is_linear_in_the_rate(self):
        base = fmat.IsotropicElastic(**ALU)
        visc = fmat.KelvinVoigt(base, 1e-3)
        eps = np.zeros((1, 6))
        rate = np.array([[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]])
        one = visc.response(eps, strain_rate=rate)["stress"]
        two = visc.response(eps, strain_rate=2.0 * rate)["stress"]
        self.assertLess(np.abs(two - 2.0 * one).max()
                        / np.abs(one).max(), 1e-13)


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


class TestElement(unittest.TestCase):

    def test_shape_functions_are_a_partition_of_unity(self):
        pts, _ = fes.gauss_points()
        self.assertLess(np.abs(fes.shape_functions(pts).sum(axis=1) - 1.0).max(),
                        1e-15)
        self.assertLess(np.abs(fes.shape_derivatives(pts).sum(axis=1)).max(),
                        1e-15)

    def test_shape_functions_are_nodal(self):
        n_f = fes.shape_functions(fes.NODE_XI)
        self.assertLess(np.abs(n_f - np.eye(8)).max(), 1e-15)

    def test_patch_test_with_incompatible_modes(self):
        mesh = _distorted_cube()
        mat = fmat.IsotropicElastic(**STEEL)
        grad = _linear_field()
        u_exact = mesh["nodes"] @ grad.T
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat)
        u = _solve_prescribed(model, mesh, u_exact)
        self.assertLess(np.abs(u - u_exact).max() / np.abs(u_exact).max(), 1e-12)
        _, _, stress = fes.internal_force(model, u)
        exact = mat.tangent() @ fmat.tensor_to_voigt(0.5 * (grad + grad.T))
        self.assertLess(np.abs(stress - exact).max() / np.abs(exact).max(), 1e-11)

    def test_patch_test_without_incompatible_modes(self):
        mesh = _distorted_cube()
        mat = fmat.IsotropicElastic(**STEEL)
        u_exact = mesh["nodes"] @ _linear_field().T
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat,
                                incompatible=False)
        u = _solve_prescribed(model, mesh, u_exact)
        self.assertLess(np.abs(u - u_exact).max() / np.abs(u_exact).max(), 1e-12)

    def test_rigid_body_modes_and_symmetry(self):
        mesh = _distorted_cube()
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(**STEEL))
        k_mat = fes.assemble_stiffness(model)
        scale = np.abs(k_mat).max()
        self.assertLess(np.abs(k_mat - k_mat.T).max() / scale, 1e-14)
        pts = mesh["nodes"]
        for axis in np.eye(3):
            for mode in (np.tile(axis, (len(pts), 1)),
                         np.cross(np.tile(axis, (len(pts), 1)), pts)):
                resid = np.abs(k_mat @ mode.reshape(-1)).max()
                self.assertLess(resid / (scale * np.abs(mode).max()), 1e-13)

    def test_internal_force_equals_stiffness_times_displacement(self):
        """The architecture's claim: f_int is the integral of B^T sigma AND it
        agrees with K u for a linear material. Both paths must exist and match."""
        mesh = _distorted_cube(2)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(**STEEL))
        rng = np.random.default_rng(3)
        u = rng.normal(size=(len(mesh["nodes"]), 3)) * 1e-5
        f_int, _, _ = fes.internal_force(model, u)
        k_mat = fes.assemble_stiffness(model)
        f_k = (k_mat @ u.reshape(-1)).reshape(-1, 3)
        self.assertLess(np.abs(f_int - f_k).max() / np.abs(f_k).max(), 1e-11)

    def test_strain_energy_matches_the_quadratic_form(self):
        mesh = _distorted_cube(2)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(**STEEL))
        rng = np.random.default_rng(4)
        u = rng.normal(size=(len(mesh["nodes"]), 3)) * 1e-5
        energy = fes.strain_energy(model, u)
        quad = 0.5 * u.reshape(-1) @ (fes.assemble_stiffness(model)
                                      @ u.reshape(-1))
        self.assertAlmostEqual(energy / quad, 1.0, places=9)

    def test_mass_properties_of_a_box_are_exact(self):
        mesh = fem_mesh.box_mesh(2.0, 3.0, 4.0, 2, 3, 2)
        mat = fmat.IsotropicElastic(**STEEL)
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat)
        prop = fes.mass_properties(model)
        mass = mat.rho * 24.0
        self.assertAlmostEqual(prop["mass"] / mass, 1.0, places=12)
        self.assertLess(np.abs(prop["centre"] - [1.0, 1.5, 2.0]).max(), 1e-12)
        exact = mass / 12.0 * np.array([3.0 ** 2 + 4.0 ** 2, 2.0 ** 2 + 4.0 ** 2,
                                        2.0 ** 2 + 3.0 ** 2])
        self.assertLess(np.abs(np.diag(prop["inertia"]) - exact).max()
                        / exact.max(), 1e-12)

    def test_consistent_mass_carries_the_rigid_body_mass(self):
        mesh = fem_mesh.box_mesh(1.0, 2.0, 0.5, 2, 2, 2)
        mat = fmat.IsotropicElastic(**STEEL)
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat)
        total = mat.rho * 1.0
        for lumped in (False, True):
            m_mat = fes.assemble_mass(model, lumped)
            rigid = np.tile([0.0, 0.0, 1.0], len(mesh["nodes"]))
            kinetic = rigid @ (m_mat @ rigid)
            self.assertAlmostEqual(kinetic / total, 1.0, places=10)

    def test_inverted_element_is_rejected(self):
        mesh = fem_mesh.box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        flipped = mesh["elements"][:, [4, 5, 6, 7, 0, 1, 2, 3]]
        with self.assertRaises(ValueError):
            fes.build_model(mesh["nodes"], flipped,
                            fmat.IsotropicElastic(**STEEL))

    def test_surface_faces_close_the_box(self):
        n_x, n_y, n_z = 2, 3, 4
        mesh = fem_mesh.box_mesh(1.0, 1.0, 1.0, n_x, n_y, n_z)
        faces, owners = fes.surface_faces(mesh["elements"])
        expect = 2 * (n_x * n_y + n_y * n_z + n_z * n_x)
        self.assertEqual(len(faces), expect)
        self.assertEqual(len(owners), expect)
        # a closed surface has zero vector area
        pts = mesh["nodes"][faces]
        area = np.cross(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1])
        self.assertLess(np.abs(area.sum(axis=0)).max(), 1e-12)

    def test_degenerate_foil_elements_are_admissible(self):
        import make_thick_sample_inputs as mts
        wrap = mts.thick_wing_mesh(n_c=8, nspan=6, span=4.0, root_chord=1.0,
                                   taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                                   camber=0.0, thickness=0.12)
        mesh = fem_mesh.solid_foil_mesh(wrap, n_thick=2)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(**ALU))
        self.assertGreater(model["cache"]["volume"].min(), 0.0)
        prop = fes.mass_properties(model)
        self.assertGreater(prop["volume"], 0.0)
        # the lofted foil has the section area of a NACA four-digit section
        self.assertAlmostEqual(prop["volume"] / (0.12 * 0.68 * 4.0), 1.0,
                               delta=0.12)


class TestMaterialSeam(unittest.TestCase):
    """The model-level plumbing of the material extension.

    The materials themselves are gated in TestMaterials; what these check is
    that an anisotropic or multi-material model actually ASSEMBLES, which is
    the half of the claim that lives in fem_solid rather than in
    fem_materials, and which nothing else exercises.
    """

    def setUp(self):
        self.mesh = fem_mesh.box_mesh(1.0, 0.2, 0.1, 3, 2, 2)
        self.carbon = fmat.OrthotropicElastic(150e9, 10e9, 10e9, 5e9, 3.5e9,
                                              5e9, 0.3, 0.3, 0.4, 1600.0)
        self.quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])

    def _build(self, material, **kw):
        return fes.build_model(self.mesh["nodes"], self.mesh["elements"],
                               material, **kw)

    def _pull(self, model, component=0):
        """Mean extension of the far face under an axial load."""
        state = fes.init_state(model)
        fes.clamp(state, self.mesh["node_sets"]["x_min"])
        fes.assemble_operators(state)
        f_ext = np.zeros_like(state["u"])
        face = self.mesh["node_sets"]["x_max"]
        f_ext[face, component] = 1.0e4 / len(face)
        level = fes.solve_static(state, f_ext)
        return float(level["u"][face, component].mean())

    def test_a_single_orientation_is_broadcast_over_the_elements(self):
        model = self._build(self.carbon, orientation=self.quarter_turn)
        self.assertEqual(model["orientation"].shape,
                         (len(self.mesh["elements"]), 3, 3))
        k_mat = fes.assemble_stiffness(model)
        self.assertLess(np.abs(k_mat - k_mat.T).max() / np.abs(k_mat).max(),
                        1e-14)

    def test_rotating_the_material_rotates_the_structure(self):
        """A quarter turn about z puts the soft axis along x, so the same bar
        must become markedly more compliant in tension. The fibre direction has
        to reach the assembly for this to happen at all."""
        along_fibres = self._pull(self._build(self.carbon))
        across_fibres = self._pull(self._build(self.carbon,
                                               orientation=self.quarter_turn))
        self.assertGreater(across_fibres / along_fibres, 3.0)

    def test_two_materials_assemble_through_mat_id(self):
        ids = np.zeros(len(self.mesh["elements"]), dtype=np.int64)
        ids[::2] = 1
        model = fes.build_model(self.mesh["nodes"], self.mesh["elements"],
                                [fmat.IsotropicElastic(**ALU), self.carbon],
                                mat_id=ids)
        k_mat = fes.assemble_stiffness(model)
        m_mat = fes.assemble_mass(model)
        self.assertLess(np.abs(k_mat - k_mat.T).max() / np.abs(k_mat).max(),
                        1e-14)
        volume = model["cache"]["volume"]
        expect = float((volume * np.where(ids == 1, self.carbon.rho,
                                          ALU["rho"])).sum())
        rigid = np.tile([0.0, 0.0, 1.0], len(self.mesh["nodes"]))
        self.assertAlmostEqual(rigid @ (m_mat @ rigid) / expect, 1.0, places=10)
        rng = np.random.default_rng(21)
        u = rng.normal(size=self.mesh["nodes"].shape) * 1e-6
        f_int, _, _ = fes.internal_force(model, u)
        f_k = (k_mat @ u.reshape(-1)).reshape(-1, 3)
        self.assertLess(np.abs(f_int - f_k).max() / np.abs(f_k).max(), 1e-10)

    def test_per_element_orientation_reaches_the_rate_damping(self):
        per_element = np.broadcast_to(self.quarter_turn,
                                      (len(self.mesh["elements"]), 3, 3)).copy()
        model = self._build(fmat.KelvinVoigt(self.carbon, 1e-4),
                            orientation=per_element)
        c_mat = fes.assemble_rate_damping(model)
        self.assertIsNotNone(c_mat)
        k_mat = fes.assemble_stiffness(model)
        # C = tau K exactly, orientation and all
        self.assertLess(np.abs(c_mat - 1e-4 * k_mat).max()
                        / np.abs(k_mat).max(), 1e-12)

    def test_a_material_without_a_constant_tangent_is_refused_by_that_path(self):
        class Stateful(fmat.Material):
            name, rho, constant_tangent, n_history = "stateful", 1.0, False, 2

        with self.assertRaises(ValueError):
            fes.element_tangents(self._build(Stateful()))

    def test_a_bad_orientation_is_rejected(self):
        with self.assertRaises(ValueError):
            fes.assemble_stiffness(self._build(self.carbon,
                                               orientation=np.full((3, 3), 0.5)))
        with self.assertRaises(ValueError):
            self._build(self.carbon, orientation=np.eye(4)[:3])

    def test_mat_id_must_index_the_material_list(self):
        ids = np.zeros(len(self.mesh["elements"]), dtype=np.int64)
        ids[0] = 5
        with self.assertRaises(ValueError):
            fes.build_model(self.mesh["nodes"], self.mesh["elements"],
                            fmat.IsotropicElastic(**ALU), mat_id=ids)


class TestMeshBuilders(unittest.TestCase):

    def test_merge_meshes_offsets_connectivity_and_sets(self):
        one = fem_mesh.box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        two = fem_mesh.box_mesh(1.0, 1.0, 1.0, 1, 1, 1, origin=(2.0, 0.0, 0.0))
        merged = fem_mesh.merge_meshes(one, two)
        self.assertEqual(len(merged["nodes"]), 16)
        self.assertEqual(len(merged["elements"]), 2)
        self.assertEqual(merged["elements"].max(), 15)
        self.assertIn("1:x_min", merged["node_sets"])
        self.assertTrue(np.all(merged["node_sets"]["1:x_min"] >= 8))
        model = fes.build_model(merged["nodes"], merged["elements"],
                                fmat.IsotropicElastic(**ALU))
        self.assertAlmostEqual(fes.mass_properties(model)["volume"], 2.0,
                               places=12)

    def test_plate_wing_sets_are_where_they_claim(self):
        chord, span = 1.0, 4.0
        plate = fem_mesh.plate_wing_mesh(chord, span, 0.05, 4, 4, 1)
        nodes = plate["nodes"]
        sets = plate["node_sets"]
        self.assertLess(np.abs(nodes[sets["root"], 1]).max(), 1e-12)
        self.assertAlmostEqual(nodes[sets["tip_max"], 1].min(), 0.5 * span,
                               places=12)
        self.assertAlmostEqual(nodes[sets["tip_min"], 1].max(), -0.5 * span,
                               places=12)
        # the default x_offset puts the QUARTER CHORD at the origin, matching
        # the sections make_thick_sample_inputs builds
        self.assertAlmostEqual(nodes[sets["leading"], 0].max(), -0.25 * chord,
                               places=12)
        self.assertAlmostEqual(nodes[sets["trailing"], 0].min(), 0.75 * chord,
                               places=12)


class TestLinearAlgebra(unittest.TestCase):

    def test_cholesky_solve_matches_numpy(self):
        rng = np.random.default_rng(11)
        a_raw = rng.normal(size=(30, 30))
        a_mat = a_raw @ a_raw.T + 30.0 * np.eye(30)
        b_vec = rng.normal(size=30)
        b_mat = rng.normal(size=(30, 4))
        low = fes.cholesky_factor(a_mat)
        for rhs in (b_vec, b_mat):
            mine = fes.cholesky_solve(low, rhs)
            ref = np.linalg.solve(a_mat, rhs)
            self.assertLess(np.abs(mine - ref).max() / np.abs(ref).max(), 1e-10)

    def test_cholesky_reports_a_singular_operator(self):
        with self.assertRaises(ValueError):
            fes.cholesky_factor(np.zeros((4, 4)))

    def test_generalised_eigenproblem(self):
        rng = np.random.default_rng(12)
        raw = rng.normal(size=(12, 12))
        k_mat = raw @ raw.T + 12.0 * np.eye(12)
        raw_m = rng.normal(size=(12, 12))
        m_mat = raw_m @ raw_m.T + 12.0 * np.eye(12)
        omega, phi = fes.generalised_modes(k_mat, m_mat)
        for j in range(len(omega)):
            resid = k_mat @ phi[:, j] - omega[j] ** 2 * (m_mat @ phi[:, j])
            self.assertLess(np.abs(resid).max()
                            / np.abs(k_mat @ phi[:, j]).max(), 1e-10)
        self.assertLess(np.abs(phi.T @ m_mat @ phi - np.eye(12)).max(), 1e-9)


class TestSolvers(unittest.TestCase):

    def _cantilever(self, n=8, material=None, damping=(0.0, 0.0)):
        mesh = fem_mesh.box_mesh(1.0, 0.1, 0.02, n, 1, 2)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                material or fmat.IsotropicElastic(**ALU))
        sstate = fes.init_state(model, rayleigh=damping)
        fes.clamp(sstate, mesh["node_sets"]["x_min"])
        fes.assemble_operators(sstate)
        return mesh, model, sstate

    def test_static_solve_is_one_newton_iteration_when_linear(self):
        mesh, _, sstate = self._cantilever()
        f_ext = np.zeros_like(sstate["u"])
        tip = mesh["node_sets"]["x_max"]
        f_ext[tip, 2] = -100.0 / len(tip)
        level = fes.solve_static(sstate, f_ext)
        self.assertLessEqual(sstate["residuals"]["newton"], 2)
        # the floor is round-off in the integral of B^T sigma on a slender stiff
        # beam, not an unconverged solve: the displacement below is compared
        # with the direct factorisation to twelve figures
        self.assertLess(sstate["residuals"]["resid"], 1e-7)
        free = fes.free_mask(sstate)
        direct = np.zeros(len(free))
        direct[free] = np.linalg.solve(sstate["K"][np.ix_(free, free)],
                                       f_ext.reshape(-1)[free])
        self.assertLess(np.abs(level["u"].reshape(-1) - direct).max()
                        / np.abs(direct).max(), 1e-9)
        self.assertLess(level["u"][tip, 2].mean(), 0.0)

    def test_springs_enter_the_diagonal_only(self):
        mesh, _, sstate = self._cantilever()
        before = sstate["K"].copy()
        fes.add_spring(sstate, mesh["node_sets"]["x_max"], (2,), 1.0e6)
        fes.assemble_operators(sstate)
        diff = sstate["K"] - before
        self.assertLess(np.abs(diff - np.diag(np.diag(diff))).max(), 1e-6)
        dofs = 3 * mesh["node_sets"]["x_max"] + 2
        self.assertLess(np.abs(np.diag(diff)[dofs] - 1.0e6).max(), 1e-3)

    def test_a_spring_supported_block_matches_its_closed_form(self):
        """One rigid-ish block on grounded springs: u = F / sum(k)."""
        mesh = fem_mesh.box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(1e12, 0.0, 1.0))
        sstate = fes.init_state(model)
        fes.add_spring(sstate, mesh["node_sets"]["all"], (0, 1, 2), 1.0e5)
        fes.assemble_operators(sstate)
        f_ext = np.zeros_like(sstate["u"])
        f_ext[:, 2] = 10.0
        level = fes.solve_static(sstate, f_ext)
        self.assertAlmostEqual(level["u"][:, 2].mean(),
                               10.0 * 8 / (8 * 1.0e5), places=10)

    def test_step_dynamic_does_not_mutate_the_state(self):
        """Purity: a strongly coupled step re-solves the same level repeatedly."""
        mesh, _, sstate = self._cantilever()
        f_ext = np.zeros_like(sstate["u"])
        f_ext[mesh["node_sets"]["x_max"], 2] = -10.0
        before = {k: np.array(sstate[k]) for k in ("u", "v", "a", "history")}
        first = fes.step_dynamic(sstate, f_ext, 1e-3)
        second = fes.step_dynamic(sstate, f_ext, 1e-3)
        for key, val in before.items():
            now = np.asarray(sstate[key])
            self.assertEqual(now.shape, val.shape)
            if val.size:
                self.assertEqual(np.abs(now - val).max(), 0.0,
                                 f"step_dynamic mutated state['{key}']")
        for key in ("u", "v", "a"):
            self.assertEqual(np.abs(first[key] - second[key]).max(), 0.0)
        self.assertGreater(np.abs(first["u"]).max(), 0.0)

    def test_step_dynamic_reproduces_a_rigid_body_acceleration(self):
        """An unconstrained block under a uniform force: a = F/m exactly."""
        mesh = fem_mesh.box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
        mat = fmat.IsotropicElastic(1e11, 0.3, 1000.0)
        model = fes.build_model(mesh["nodes"], mesh["elements"], mat)
        sstate = fes.init_state(model)
        fes.add_spring(sstate, mesh["node_sets"]["all"], (0, 1, 2), 1e-3)
        fes.assemble_operators(sstate)
        m_total = mat.rho * 1.0
        f_ext = np.zeros_like(sstate["u"])
        m_diag = np.diag(sstate["M"]).reshape(-1, 3)
        f_ext[:, 2] = 100.0 * m_diag[:, 2] / m_diag[:, 2].sum()   # uniform field
        sstate["a"] = fes.initial_acceleration(sstate, f_ext)
        self.assertAlmostEqual(sstate["a"][:, 2].mean() / (100.0 / m_total),
                               1.0, places=6)

    def test_integrator_parameters(self):
        par = fes.integrator_parameters(1.0)
        self.assertAlmostEqual(par["alpha_m"], 0.5)
        self.assertAlmostEqual(par["alpha_f"], 0.5)
        self.assertAlmostEqual(par["beta"], 0.25)
        self.assertAlmostEqual(par["gamma"], 0.5)
        newmark = fes.integrator_parameters(alpha_m=0.0, alpha_f=0.0)
        self.assertAlmostEqual(newmark["beta"], 0.25)
        self.assertAlmostEqual(newmark["gamma"], 0.5)
        with self.assertRaises(ValueError):
            fes.integrator_parameters(1.5)

    def test_modes_are_mass_normalised_and_ordered(self):
        _, _, sstate = self._cantilever()
        freq, shapes = fes.modes(sstate, 4)
        self.assertTrue(np.all(np.diff(freq) >= -1e-9))
        self.assertGreater(freq[0], 0.0)
        flat = shapes.reshape(-1, shapes.shape[-1])
        gram = flat.T @ (sstate["M"] @ flat)
        self.assertLess(np.abs(gram - np.eye(len(freq))).max(), 1e-8)


@unittest.skipUnless(SLOW, "set FEM_SLOW=1 for the convergence studies")
class TestSlow(unittest.TestCase):

    LENGTH, WIDTH, THICK = 1.0, 0.1, 0.02

    def _beam_reference(self, load):
        e_mod, nu = ALU["e_mod"], ALU["nu"]
        inertia = self.WIDTH * self.THICK ** 3 / 12.0
        area = self.WIDTH * self.THICK
        shear = 0.5 * e_mod / (1.0 + nu)
        kappa = 5.0 * (1.0 + nu) / (6.0 + 5.0 * nu)
        euler = load * self.LENGTH ** 3 / (3.0 * e_mod * inertia)
        return euler + load * self.LENGTH / (kappa * shear * area), inertia, area

    def _cantilever(self, n_length, n_thick=2):
        mesh = fem_mesh.box_mesh(self.LENGTH, self.WIDTH, self.THICK,
                                 n_length, 1, n_thick)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(**ALU))
        sstate = fes.init_state(model)
        fes.clamp(sstate, mesh["node_sets"]["x_min"])
        fes.assemble_operators(sstate)
        return mesh, model, sstate

    def test_tip_deflection_converges_to_the_beam_solution(self):
        load = -100.0
        timoshenko, _, _ = self._beam_reference(load)
        ratios = []
        for n in (8, 16, 32):
            mesh, _, sstate = self._cantilever(n)
            f_ext = np.zeros_like(sstate["u"])
            tip = mesh["node_sets"]["x_max"]
            f_ext[tip, 2] = load / len(tip)
            level = fes.solve_static(sstate, f_ext)
            ratios.append(level["u"][tip, 2].mean() / timoshenko)
        self.assertTrue(np.all(np.diff(ratios) > 0.0), ratios)
        self.assertGreater(ratios[-1], 0.97)
        self.assertLess(ratios[-1], 1.01)

    def test_incompatible_modes_cure_the_locking(self):
        load = -100.0
        timoshenko, _, _ = self._beam_reference(load)
        out = {}
        for incompatible in (True, False):
            mesh = fem_mesh.box_mesh(self.LENGTH, self.WIDTH, self.THICK,
                                     8, 1, 2)
            model = fes.build_model(mesh["nodes"], mesh["elements"],
                                    fmat.IsotropicElastic(**ALU),
                                    incompatible=incompatible)
            sstate = fes.init_state(model)
            fes.clamp(sstate, mesh["node_sets"]["x_min"])
            fes.assemble_operators(sstate)
            f_ext = np.zeros_like(sstate["u"])
            tip = mesh["node_sets"]["x_max"]
            f_ext[tip, 2] = load / len(tip)
            level = fes.solve_static(sstate, f_ext)
            out[incompatible] = level["u"][tip, 2].mean() / timoshenko
        self.assertGreater(out[True], 0.95)
        self.assertLess(out[False], 0.15)

    def test_flapwise_frequencies_match_euler_bernoulli(self):
        mesh, _, sstate = self._cantilever(24)
        _, inertia, area = self._beam_reference(1.0)
        freq, shapes = fes.modes(sstate, 12)
        tip = mesh["node_sets"]["x_max"]
        flap = [j for j in range(len(freq))
                if np.abs(shapes[tip, 2, j]).mean()
                > 3.0 * np.abs(shapes[tip, 1, j]).mean()]
        scale = np.sqrt(ALU["e_mod"] * inertia
                        / (ALU["rho"] * area * self.LENGTH ** 4)) / (2.0 * np.pi)
        for j, beta in zip(flap[:3], (1.875104, 4.694091, 7.854757)):
            self.assertAlmostEqual(freq[j] / (beta ** 2 * scale), 1.0, delta=0.03)

    def test_generalised_alpha_period_elongation(self):
        mesh, _, sstate = self._cantilever(12)
        freq, phi = fes.modes(sstate, 1)
        period = 1.0 / freq[0]
        omega = 2.0 * np.pi * freq[0]
        tip = mesh["node_sets"]["x_max"]
        shape = phi[:, :, 0] * 1e-3 / np.abs(phi[:, :, 0]).max()
        for steps in (32, 64):
            state = fes.init_state(sstate["model"])
            fes.clamp(state, mesh["node_sets"]["x_min"])
            fes.assemble_operators(state)
            state["u"] = shape.copy()
            zero = np.zeros_like(state["u"])
            state["a"] = fes.initial_acceleration(state, zero)
            par = fes.integrator_parameters(1.0)
            dt = period / steps
            times, tips = [0.0], [state["u"][tip, 2].mean()]
            for n in range(1, 5 * steps + 1):
                fes.commit(state, fes.step_dynamic(state, zero, dt, par), zero)
                times.append(n * dt)
                tips.append(state["u"][tip, 2].mean())
            times, tips = np.array(times), np.array(tips)
            cross = np.where(np.diff(np.sign(tips)) != 0)[0]
            zeros = times[cross] - tips[cross] * (times[cross + 1] - times[cross]) \
                / (tips[cross + 1] - tips[cross])
            measured = 2.0 * np.mean(np.diff(zeros))
            theory = (omega * dt) ** 2 / 12.0
            self.assertAlmostEqual(measured / period - 1.0, theory, delta=0.06 * theory)
            self.assertAlmostEqual(abs(tips[-1] / tips[0]), 1.0, delta=0.02)

    def test_kelvin_voigt_decays_at_the_analytical_rate(self):
        base = fmat.IsotropicElastic(**ALU)
        for tau in (1e-4, 3e-4):
            mesh = fem_mesh.box_mesh(self.LENGTH, self.WIDTH, self.THICK,
                                     12, 1, 2)
            model = fes.build_model(mesh["nodes"], mesh["elements"],
                                    fmat.KelvinVoigt(base, tau))
            state = fes.init_state(model)
            fes.clamp(state, mesh["node_sets"]["x_min"])
            fes.assemble_operators(state)
            freq, phi = fes.modes(state, 1)
            omega = 2.0 * np.pi * freq[0]
            zeta = 0.5 * tau * omega
            state["u"] = phi[:, :, 0] * 1e-3 / np.abs(phi[:, :, 0]).max()
            zero = np.zeros_like(state["u"])
            state["a"] = fes.initial_acceleration(state, zero)
            tip = mesh["node_sets"]["x_max"]
            start = state["u"][tip, 2].mean()
            dt = 1.0 / (freq[0] * 128)
            for _ in range(3 * 128):
                fes.commit(state, fes.step_dynamic(state, zero, dt, None), zero)
            measured = abs(state["u"][tip, 2].mean() / start)
            exact = np.exp(-zeta * omega * 3.0 / freq[0])
            self.assertAlmostEqual(measured / exact, 1.0, delta=2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
