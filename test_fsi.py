"""Tests for the transfer operators and the coupled drivers.

Run the fast set:      python3 test_fsi.py
Include the slow set:  FSI_SLOW=1 python3 test_fsi.py

The fast set pins the properties the transfer has by construction rather than
by tolerance - partition of unity, exactness for rigid-body motion, and
conservation of force, moment and virtual work - together with the bookkeeping
of the coupled time step, which is where a partitioned code goes wrong quietly:
that the wake is shed once per step and not once per subiteration, that the
fluid residual is a deterministic function of the displacement, and that the
doublet history advances only when the step is committed.

The slow set holds the coupled physics: the rigid limit against the fluid
code's own time march, static divergence against the classical typical-section
formula, and the energy balance of the coupled system.
"""

import os
import unittest
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

SLOW = os.environ.get("FSI_SLOW", "") not in ("", "0")


def uniform_onset(points):
    pts = np.atleast_2d(points)
    vel = np.zeros((len(pts), 3))
    vel[:, 0] = 1.0
    return vel


def still_fluid(points):
    return np.zeros((len(np.atleast_2d(points)), 3))


def small_wing(n_c=6, nspan=6, span=4.0, thickness=0.12, alpha=5.0):
    pts = mts.thick_wing_mesh(n_c=n_c, nspan=nspan, span=span, root_chord=1.0,
                              taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                              camber=0.0, thickness=thickness)
    return tpw.pitch_mesh(pts, np.radians(alpha)) if alpha else pts


def build_case(points, e_mod=2.0e8, rho_s=1200.0, rho_f=1000.0, n_thick=1,
               unsteady=True, nwake=0, clamp="root"):
    mesh = fem_mesh.solid_foil_mesh(points, n_thick=n_thick)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(e_mod, 0.3, rho_s))
    config = None if unsteady else {"lifting": True, "nwake": nwake}
    fluid, sstate, transfer = fsd.init_fsi(points, uniform_onset, model,
                                           u_ref=1.0, rho=rho_f,
                                           unsteady=unsteady, config=config)
    if clamp:
        fes.clamp(sstate, mesh["node_sets"][clamp])
    fes.assemble_operators(sstate)
    return fluid, sstate, transfer, mesh, model


class TestTransfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.points = small_wing()
        cls.fluid, cls.sstate, cls.transfer, cls.mesh, cls.model = \
            build_case(cls.points, unsteady=False, nwake=8)
        cls.fluid = tpw.solve(cls.fluid, wake="frozen")
        cls.loads = tpw.get_loads(cls.fluid, rho=1000.0)

    def test_weights_are_a_partition_of_unity(self):
        self.assertLess(np.abs(self.transfer["weights"].sum(axis=1) - 1.0).max(),
                        1e-14)

    def test_the_lofted_structure_carries_the_wetted_surface(self):
        """Every fluid node projects onto the structural surface exactly."""
        self.assertLess(self.transfer["offset_rel"], 1e-12)

    def test_rigid_body_motion_transfers_exactly(self):
        x_s = self.model["nodes"]
        x_f = self.fluid["panels"]["points"].reshape(-1, 3) \
            + self.transfer["offset"]
        cases = {
            "translation": (np.tile([0.3, -0.2, 0.5], (len(x_s), 1)),
                            np.tile([0.3, -0.2, 0.5], (len(x_f), 1))),
            "rotation": (np.cross(np.tile([0.01, 0.02, -0.03], (len(x_s), 1)), x_s),
                         np.cross(np.tile([0.01, 0.02, -0.03], (len(x_f), 1)), x_f)),
        }
        grad = np.array([[1e-2, 0.0, 0.0], [0.0, -3e-3, 2e-3], [0.0, 2e-3, -3e-3]])
        cases["linear"] = (x_s @ grad.T, x_f @ grad.T)
        for name, (u_s, exact) in cases.items():
            got = fsx.to_fluid(self.transfer, u_s)
            self.assertLess(np.abs(got - exact).max() / np.abs(exact).max(),
                            1e-12, name)

    def test_welded_fluid_nodes_receive_identical_displacements(self):
        rng = np.random.default_rng(2)
        u_s = rng.normal(size=self.model["nodes"].shape) * 1e-3
        u_f = fsx.to_fluid(self.transfer, u_s)
        weld = self.transfer["weld_map"]
        self.assertLess(np.abs(u_f - u_f[weld]).max(), 0.0 + 1e-300)

    def test_panel_lumping_conserves_force_and_moment(self):
        pan = self.fluid["panels"]
        force = self.loads["force"]
        nodal = fsx.panel_forces_to_nodes(pan, force)
        scale = np.abs(force).sum()
        self.assertLess(np.abs(nodal.sum(axis=0) - force.sum(axis=0)).max()
                        / scale, 1e-13)
        x_f = pan["points"].reshape(-1, 3)
        lever = np.linalg.norm(np.ptp(x_f, axis=0))
        m_panel = np.cross(pan["centroids"], force).sum(axis=0)
        m_nodal = np.cross(x_f, nodal).sum(axis=0)
        self.assertLess(np.abs(m_nodal - m_panel).max() / (scale * lever), 1e-13)

    def test_the_lumping_is_the_adjoint_of_the_fluid_velocity_sampling(self):
        """<f_panels, S v_nodes> = <S^T f_panels, v_nodes> exactly."""
        pan = self.fluid["panels"]
        rng = np.random.default_rng(5)
        force = rng.normal(size=(len(pan["areas"]), 3))
        v_nodes = rng.normal(size=pan["points"].shape)
        left = float(np.einsum("kc,kc->", force,
                               tpw.sample_at_centroids(pan, v_nodes)))
        right = float(np.einsum("mc,mc->", fsx.panel_forces_to_nodes(pan, force),
                                v_nodes.reshape(-1, 3)))
        self.assertLess(abs(left - right) / max(abs(left), 1e-300), 1e-13)

    def test_the_transfer_conserves_virtual_work(self):
        rng = np.random.default_rng(6)
        u_s = rng.normal(size=self.model["nodes"].shape) * 1e-3
        report = fsx.conservation_report(self.transfer, self.fluid["panels"],
                                         self.loads["force"], u_struct=u_s)
        self.assertLess(report["force_error_lump"], 1e-13)
        self.assertLess(report["force_error_transfer"], 1e-13)
        self.assertLess(report["moment_error_lump"], 1e-13)
        self.assertLess(report["work_error"], 1e-12)

    def test_a_non_matching_structure_still_conserves_force(self):
        """A flat plate under a curved wetted surface: force exact, moment offset
        bounded by the projection distance and reported, not hidden."""
        plate = fem_mesh.plate_wing_mesh(1.0, 4.0, 0.06, 8, 8, 1, x_offset=0.0)
        model = fes.build_model(plate["nodes"], plate["elements"],
                                fmat.IsotropicElastic(2e8, 0.3, 1200.0))
        transfer = fsx.build_transfer(self.fluid, model)
        self.assertGreater(transfer["offset_max"], 1e-3)     # genuinely apart
        report = fsx.conservation_report(transfer, self.fluid["panels"],
                                         self.loads["force"])
        self.assertLess(report["force_error_transfer"], 1e-13)
        self.assertLess(np.abs(transfer["weights"].sum(axis=1) - 1.0).max(), 1e-13)


class TestAccelerators(unittest.TestCase):
    """Exercised on a linear fixed point, with no fluid in the way."""

    N_DOF = 12

    def _matrix(self, gain, dominant):
        """A fixed-point map with either one dominant mode or a full spread.

        The distinction is the whole difference between the two accelerators.
        Aitken carries one scalar, so it can annihilate a single dominant mode
        and little else - and the added-mass mode of a partitioned coupling is
        exactly such a mode, which is why a scalar relaxation works there at
        all. IQN-ILS builds a subspace and does not care either way.
        """
        n = self.N_DOF
        basis, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(n, n)))
        vals = np.concatenate([[gain], np.linspace(0.3, 0.05, n - 1)]) \
            if dominant else np.linspace(gain, -gain, n)
        return basis @ np.diag(vals) @ basis.T

    def _fixed_point(self, method, gain=0.98, iterations=60, reuse=30,
                     dominant=True):
        """u -> M u + b, driven through the accelerator.

        The state is carried as an (n, 3) array because that is the shape the
        accelerator sees in the driver; only the first column is active.
        """
        n = self.N_DOF
        mat = self._matrix(gain, dominant)
        rhs = np.random.default_rng(2).normal(size=n)
        acc = fsd.FixedPointAccelerator(method, omega=0.5, reuse=reuse)
        u = np.zeros((n, 3))
        resid = np.inf
        for it in range(iterations):
            tilde = np.zeros_like(u)
            tilde[:, 0] = mat @ u[:, 0] + rhs
            resid = np.linalg.norm(tilde - u)
            if resid < 1e-10:
                return it, resid
            u = acc.step(u, tilde)
        return iterations, resid

    def test_iqn_terminates_in_at_most_n_plus_one_iterations(self):
        """On a LINEAR fixed point, IQN-ILS with full history is a Krylov method
        and must terminate in at most the dimension of the problem - whether or
        not one mode dominates."""
        for dominant in (True, False):
            count, resid = self._fixed_point("iqn", dominant=dominant)
            self.assertLessEqual(count, self.N_DOF + 1, f"dominant={dominant}")
            self.assertLess(resid, 1e-10)

    def test_aitken_converges_where_a_constant_relaxation_does_not(self):
        """Measured on the residual, not on an iteration count: Aitken is a
        scalar method, so on a problem with a spectral radius of 0.98 it
        converges steadily but not at the terminating rate of IQN-ILS. Both
        statements are true and the tolerance says which is which."""
        for dominant in (True, False):
            _, aitken = self._fixed_point("aitken", dominant=dominant)
            _, constant = self._fixed_point("constant", dominant=dominant)
            self.assertLess(aitken, 1e-6, f"dominant={dominant}")
            self.assertGreater(constant, 1e-2, f"dominant={dominant}")

    def test_unknown_accelerator_is_rejected(self):
        with self.assertRaises(ValueError):
            fsd.FixedPointAccelerator("newton")


class TestCoupledStep(unittest.TestCase):

    def test_the_wake_grows_by_one_row_per_step_not_per_subiteration(self):
        """shed mutates the wake, so it must be outside the subiteration loop."""
        points = small_wing(n_c=4, nspan=4)
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=5e7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history, fluid, _ = fsd.time_march_fsi(fluid, sstate, transfer,
                                                   dt=0.1, nsteps=4, rho=1000.0,
                                                   max_sub=12)
        self.assertGreater(history["sub"][1:].max(), 1)      # it did subiterate
        self.assertEqual(list(history["nrows"]), [0, 1, 2, 3, 4])
        self.assertEqual(fluid["wake"]["mu"].shape[0], 4)

    def test_the_fluid_solve_is_deterministic_at_fixed_geometry(self):
        """What quasi-Newton acceleration needs: the residual is a function of
        the displacement, not of the path taken to it."""
        points = small_wing(n_c=4, nspan=4)
        fluid, _, transfer, _, model = build_case(points, unsteady=False, nwake=6)
        rng = np.random.default_rng(9)
        u_s = rng.normal(size=model["nodes"].shape) * 1e-4
        pts = fsx.displaced_points(transfer, points, u_s)
        first = None
        for _ in range(3):
            fluid = tpw.update_points(fluid, pts)
            fluid = tpw.solve(fluid, wake="frozen")
            if first is None:
                first = fluid["mu"].copy()
            else:
                self.assertEqual(np.abs(fluid["mu"] - first).max(), 0.0)

    def test_added_mass_ratio_is_positive_and_restores_the_geometry(self):
        points = small_wing(n_c=4, nspan=4)
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=2e8)
        before = fluid["panels"]["points"].copy()
        report = fsd.added_mass_ratio(fluid, sstate, transfer, 1000.0, 0.05)
        self.assertGreater(report["ratio"], 0.0)
        self.assertLess(report["frequency_wet_estimate"], report["frequency_dry"])
        self.assertLess(np.abs(fluid["panels"]["points"] - before).max(), 1e-12)

    def test_steps_per_period_is_the_structural_resolution(self):
        points = small_wing(n_c=4, nspan=4)
        _, sstate, _, _, _ = build_case(points, e_mod=2e8)
        freq, _ = fes.modes(sstate, 1)
        self.assertAlmostEqual(fsd.steps_per_period(sstate, 0.05),
                               1.0 / (freq[0] * 0.05), places=8)

    def test_static_coupling_reproduces_the_rigid_solution(self):
        """E -> infinity: the coupled fixed point must return the rigid answer."""
        points = small_wing(n_c=6, nspan=6)
        rigid = tpw.solve(tpw.init_state(points, uniform_onset, 1.0,
                                         config={"lifting": True, "nwake": 10}),
                          wake="relax")
        cl_rigid = tpw.get_loads(rigid, rho=1000.0)["resultants"]["CL"]
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=1e12,
                                                   unsteady=False, nwake=10)
        fluid, sstate, hist = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                    rho=1000.0, max_iter=6)
        cl_coupled = tpw.get_loads(fluid, rho=1000.0)["resultants"]["CL"]
        self.assertTrue(hist["converged"])
        self.assertLess(abs(cl_coupled - cl_rigid) / abs(cl_rigid), 1e-5)
        # a real deflection, falling as 1/E: 3e-8 chords at E = 1e14. The
        # statement worth making is that it is negligible against the mesh
        self.assertLess(np.abs(sstate["u"]).max(), 1e-5)


@unittest.skipUnless(SLOW, "set FSI_SLOW=1 for the coupled physics gates")
class TestSlow(unittest.TestCase):

    def test_the_time_march_reproduces_the_rigid_history(self):
        """The whole coupled loop against the fluid code's own driver.

        Started from the static aeroelastic equilibrium rather than from rest,
        because a stiff structure released from rest rings at a frequency the
        step cannot resolve; that is a resolution requirement of the coupled
        problem and is documented as one, not a defect of either solver.
        """
        points = small_wing(n_c=6, nspan=6)
        reference, _ = usw.time_march(points, uniform_onset,
                                      motion=usw.still(points), dt=0.05,
                                      nsteps=8, u_ref=1.0, rho=1000.0)
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=2e12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                      rho=1000.0, max_iter=10)
            history, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=0.05,
                                               nsteps=8, rho=1000.0,
                                               rho_inf=0.5, max_sub=20)
        error = np.abs(history["CL"] - reference["CL"]).max() \
            / np.abs(reference["CL"]).max()
        self.assertLess(error, 0.02)

    def test_static_divergence_matches_the_typical_section_formula(self):
        """Three independent routes to one dynamic pressure.

        The classical formula uses the fluid's own lift slope and aerodynamic
        centre and the structure's own torsional stiffness; the modal route
        finds where K - q A loses rank; the Southwell extrapolation watches the
        fixed point stop contracting. The first is a different model from the
        other two, which is what makes the comparison a verification.
        """
        points = small_wing(n_c=8, nspan=8, span=6.0, thickness=0.10, alpha=2.0)
        loads = []
        for alpha in (2.0, 3.0):
            wing = small_wing(n_c=8, nspan=8, span=6.0, thickness=0.10,
                              alpha=alpha)
            state = tpw.solve(tpw.init_state(wing, uniform_onset, 1.0,
                                             config={"lifting": True,
                                                     "nwake": 16}),
                              wake="relax")
            loads.append(tpw.get_loads(state, rho=1.0)["resultants"])
        d_alpha = np.radians(1.0)
        slope = (loads[1]["CL"] - loads[0]["CL"]) / d_alpha
        x_ac = loads[0]["x_ref"][0] - (loads[1]["moment"][1]
                                       - loads[0]["moment"][1]) \
            / (loads[1]["force"][2] - loads[0]["force"][2])
        s_ref = state["s_ref"]

        mesh = fem_mesh.solid_foil_mesh(points, n_thick=1)
        nodes, k_spring = mesh["nodes"], 4.0
        centre = nodes.mean(axis=0)
        k_theta = k_spring * np.sum((nodes[:, 0] - centre[0]) ** 2
                                    + (nodes[:, 2] - centre[2]) ** 2)
        q_classical = k_theta / (s_ref * slope * (centre[0] - x_ac))

        model = fes.build_model(nodes, mesh["elements"],
                                fmat.IsotropicElastic(1e9, 0.3, 100.0))
        fluid, sstate, transfer = fsd.init_fsi(points, uniform_onset, model,
                                               u_ref=1.0, rho=1.0,
                                               unsteady=False,
                                               config={"lifting": True,
                                                       "nwake": 16})
        fes.add_spring(sstate, mesh["node_sets"]["all"], (0, 1, 2), k_spring)
        fes.assemble_operators(sstate)
        modal = fsd.divergence_pressure(fluid, sstate, transfer, rho=1.0,
                                        nmodes=6)
        self.assertAlmostEqual(modal["q_divergence"] / q_classical, 1.0,
                               delta=0.05)

        inverse_q, inverse_a = [], []
        for fraction in (0.3, 0.4, 0.5, 0.6, 0.7):
            q_dyn = fraction * q_classical
            f_2, s_2, t_2 = fsd.init_fsi(points, uniform_onset, model,
                                         u_ref=1.0, rho=2.0 * q_dyn,
                                         unsteady=False,
                                         config={"lifting": True, "nwake": 16})
            fes.add_spring(s_2, mesh["node_sets"]["all"], (0, 1, 2), k_spring)
            fes.assemble_operators(s_2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, s_2, _ = fsd.static_aeroelastic(f_2, s_2, t_2,
                                                   rho=2.0 * q_dyn,
                                                   max_iter=60, tol=1e-9)
            inverse_q.append(1.0 / q_dyn)
            inverse_a.append(1.0 / float(np.abs(s_2["u"][:, 2]).max()))
        fit = np.polyfit(inverse_q, inverse_a, 1)
        q_southwell = 1.0 / (-fit[1] / fit[0])
        self.assertAlmostEqual(q_southwell / modal["q_divergence"], 1.0,
                               delta=1e-3)
        self.assertAlmostEqual(q_southwell / q_classical, 1.0, delta=0.05)

    def test_loose_coupling_fails_where_strong_coupling_holds(self):
        """The added-mass instability, demonstrated rather than asserted."""
        points = small_wing(n_c=6, nspan=6)
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=2e8)
        ratio = fsd.added_mass_ratio(fluid, sstate, transfer, 1000.0, 0.05)
        self.assertGreater(ratio["ratio"], 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strong, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=0.05,
                                              nsteps=8, rho=1000.0, max_sub=30)
            self.assertTrue(np.all(np.isfinite(strong["CL"])))
            fluid, sstate, transfer, _, _ = build_case(points, e_mod=2e8)
            diverged = False
            try:
                loose, _, _ = fsd.time_march_fsi(fluid, sstate, transfer,
                                                 dt=0.05, nsteps=8, rho=1000.0,
                                                 coupling="loose")
                diverged = not np.all(np.isfinite(loose["CL"])) or \
                    np.abs(loose["tip"]).max() > 100.0 * np.abs(strong["tip"]).max()
            except (ValueError, np.linalg.LinAlgError):
                diverged = True                     # the mesh inverted: divergence
        self.assertTrue(diverged, "loose coupling survived at an added-mass "
                                  "ratio above one, which it should not")

    def test_energy_balance_of_the_coupled_system(self):
        """Work done by the fluid = strain energy + kinetic energy, in vacuo-free
        form, to the accuracy of the trapezoidal work integral."""
        points = small_wing(n_c=6, nspan=6)
        fluid, sstate, transfer, _, _ = build_case(points, e_mod=2e9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=0.05,
                                               nsteps=12, rho=1000.0,
                                               rho_inf=0.9, max_sub=30)
        stored = history["strain_energy"] + history["kinetic_energy"]
        work = history["fluid_work"]
        scale = max(np.abs(work).max(), 1e-30)
        self.assertLess(np.abs(stored - work).max() / scale, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
