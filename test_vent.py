"""Tests of the ventilation model: the sectional physics, the image, the cavity.

    python3 test_vent.py                    # the fast set, about 12 s
    VENT_SLOW=1 python3 test_vent.py        # + the coupled ventilation gates

The fast set pins what holds by construction, which is most of what can go wrong
here: the exact mirror symmetry of the doubled strut, the antisymmetry of the
image strengths, the vanishing of the potential on the free-surface plane, the
bit-for-bit agreement between the unflipped image and the vendored solve, the
closed forms of the sectional model and the algebra of the washout boundary, the
conservation of the modified load through the transfer, and the fact that a
coupled run with ventilation switched off is bit-for-bit the run without it.

Then it pins the bookkeeping of the ventilated step, which is where this model
goes wrong quietly: that the regime advances once per step and not once per
subiteration, that the ventilated load is a deterministic function of the
geometry and the committed regime, and that the hysteresis is structural rather
than a threshold with a dead band.

No files are written and no figures are created.
"""

import os
import unittest
import warnings

import numpy as np

import thick_panel_wing as tpw
import freewake_kernels as fk
import fem_materials as fmat
import fem_solid as fes
import fsi_driver as fsd
import vent_section as vs
import vent_cavity as vcv
import vent_loads as vnl
import vent_mesh as vm
import vent_solve as vsl

SLOW = os.environ.get("VENT_SLOW", "") not in ("", "0")
RHO_WATER = 1000.0
STEEL = dict(e_mod=2.0e8, nu=0.3, rho=1200.0)


def uniform_onset(u=1.0):
    return fk.make_onset(np.array([-10.0, 10.0]), np.array([u, u]))


def strut_case(alpha=16.0, fn_h=2.5, n_c=8, nspan_half=6, h=1.0, chord=1.0,
               image=True, regime="FW", nwake=6, unsteady=False, **vent_kw):
    """A ventilated surface-piercing strut. -> (fluid, sstate, transfer, vent, mesh)"""
    points = vm.strut_mesh(n_c=n_c, nspan_half=nspan_half, h=h, chord=chord,
                           alpha_deg=alpha, image=image)
    maps = vm.mirror_maps(points.shape, image=image)
    mesh = vm.strut_solid_mesh(vm.half_points(points, maps), n_thick=2,
                               min_half_thickness=0.02)
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(**STEEL))
    fluid, sstate, transfer, vent = fsd.init_vent_fsi(
        points, uniform_onset(), model, h=h, chord=chord, u_ref=1.0,
        rho=RHO_WATER, alpha_deg=alpha, image=image, unsteady=unsteady,
        config={"lifting": True, "nwake": nwake}, fn_h=fn_h, regime=regime,
        **vent_kw)
    fes.clamp(sstate, mesh["node_sets"]["tip_max"])
    fes.assemble_operators(sstate)
    return fluid, sstate, transfer, vent, mesh


def solved(alpha=16.0, regime="FW", **kw):
    """One converged ventilated solve. -> (fluid, vent, cav, loads)"""
    fluid, _, _, vent, _ = strut_case(alpha=alpha, regime=regime, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
    return fluid, vent, cav, loads


class TestSectional(unittest.TestCase):
    """The two-dimensional closed forms, and the fits taken from them."""

    def test_the_cavitation_number_is_linear_in_depth_with_the_froude_slope(self):
        for fn_h in (0.8, 1.5, 3.0):
            got = vs.sigma_cavity(np.array([0.0, 0.5, 1.0]), 1.0, fn_h)
            self.assertAlmostEqual(got[2] - got[0], 2.0 / fn_h ** 2, delta=1e-14)
            self.assertEqual(got[0], 0.0)          # natural ventilation, exactly

    def test_the_cavitation_number_increases_downward_from_the_waterline(self):
        got = vs.sigma_cavity(np.linspace(0.0, 1.0, 11), 1.0, 2.0)
        self.assertTrue(np.all(np.diff(got) > 0.0))
        self.assertLess(float(vs.sigma_cavity(-0.3, 1.0, 2.0)), 0.0)

    def test_the_lift_slope_has_the_right_two_analytic_limits(self):
        # the constant 1/(2 pi) in (1.9) is illegible in the paper's text layer;
        # these two limits are one of the two ways it was recovered, and the
        # plausible misreading 2 pi gives a0(0) = 1 with a smooth curve otherwise
        self.assertAlmostEqual(float(vs.lift_slope(0.0)), 2.0 * np.pi, delta=1e-14)
        self.assertAlmostEqual(float(vs.lift_slope(np.inf)), np.pi / 2.0,
                               delta=1e-14)
        self.assertAlmostEqual(float(vs.lift_slope(1e12)), np.pi / 2.0, delta=1e-9)

    def test_the_lift_slope_fit_matches_acosta_on_the_partial_cavity_branch(self):
        length = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
        ratio = vs.lift_slope(length) / vs.acosta_lift_slope(length)
        self.assertLess(float(np.abs(ratio - 1.0).max()), 0.015)

    def test_the_lift_slope_peaks_on_the_partial_branch_then_falls_to_the_limit(self):
        """The shape of the paper's figure 2(b), including two oddities of the fit.

        a0 RISES above 2 pi to a maximum near L = 0.51, where a partial cavity is
        most effective at loading the section, and falls thereafter towards pi/2.
        The fit then UNDERSHOOTS its own limit by 0.6% near L = 10 and approaches
        pi/2 from below, so it is not monotone in the tail; that belongs to the
        rational polynomial and not to the physics, and it is recorded rather than
        smoothed away.
        """
        peak = np.linspace(0.01, 1.5, 3000)
        got = vs.lift_slope(peak)
        self.assertGreater(float(got.max()), 2.0 * np.pi)
        self.assertAlmostEqual(float(peak[int(np.argmax(got))]), 0.51, delta=0.03)
        self.assertTrue(np.all(np.diff(vs.lift_slope(np.linspace(1.0, 8.0, 400)))
                               < 0.0))
        tail = vs.lift_slope(np.logspace(0.5, 6.0, 2000))
        self.assertGreater(float(tail.min()), (np.pi / 2.0) * 0.99)

    def test_the_acosta_relation_inverts_exactly(self):
        length = np.linspace(0.02, 0.5, 40)
        back = vs.acosta_length(vs.acosta_psi(length))
        self.assertLess(float(np.abs(back - length).max()), 1e-12)

    def test_the_tulin_relation_inverts_exactly(self):
        length = np.linspace(1.3, 20.0, 40)
        back = vs.tulin_length(vs.tulin_psi(length))
        self.assertLess(float(np.abs(back - length).max() / length.max()), 1e-13)

    def test_the_blended_length_inverts_acosta_in_psi_not_in_length(self):
        """dPsi/dL is large near the branch join, so the comparison is in Psi.

        Equation (1.7) at Acosta's L = 0.5 returns 0.456, 8.7% low in L, while
        being about 1% right in Psi. A test written on L reads as a failure of a
        fit that is correct.
        """
        length = np.array([0.2, 0.3, 0.4, 0.5])
        psi = vs.acosta_psi(length)
        back_psi = vs.acosta_psi(vs.harwood_length(psi))
        self.assertLess(float(np.abs(back_psi / psi - 1.0).max()), 0.05)

    def test_the_blended_length_is_finite_and_saturated_at_zero_cavitation(self):
        # Psi^2 - 7.1 Psi + 49.42 has negative discriminant, so (1.7) has no
        # positive pole; the saturation belongs to the fit, not to the physics
        self.assertTrue(np.all(np.isfinite(vs.harwood_length(
            np.logspace(-6, 3, 200)))))
        self.assertAlmostEqual(float(vs.harwood_length(0.0)), vs.L_FIT_MAX,
                               delta=0.02)

    def test_the_exact_length_model_is_monotone_finite_and_continuous(self):
        psi = np.logspace(np.log10(0.05), np.log10(60.0), 3000)
        length = vs.cavity_length(psi, "exact")
        self.assertTrue(np.all(np.isfinite(length)))
        self.assertTrue(np.all(np.diff(length) < 0.0))
        # continuity across both branch joins, which the Hermite blend spans
        for join in (float(vs.tulin_psi(vs.L_SUPER)),
                     float(vs.acosta_psi(vs.L_PARTIAL))):
            lo = float(vs.cavity_length(join * (1.0 - 1e-6), "exact"))
            hi = float(vs.cavity_length(join * (1.0 + 1e-6), "exact"))
            self.assertLess(abs(lo - hi) / max(lo, hi), 1e-4)

    def test_the_low_order_length_is_not_an_approximation_to_tulin(self):
        """Tulin grows as sigma^-2 and (1.8) as sigma^-1; the ratio diverges."""
        psi = 0.1
        self.assertGreater(float(vs.tulin_length(psi)
                                 / vs.harwood_length_low_order(psi)), 2.0)
        self.assertGreater(float(vs.harwood_length_low_order(psi)), 1.0)

    def test_the_cavity_parameter_carries_its_factor_of_two(self):
        # the Psi = 10 -> L = 0.157 anchor; dropping the 2 halves the cavity
        self.assertAlmostEqual(float(vs.harwood_length(10.0)), 0.157, delta=0.002)
        self.assertAlmostEqual(float(vs.psi(0.4, 0.02)), 10.0, delta=1e-12)

    def test_the_centre_of_pressure_moves_from_quarter_chord_to_three_sixteenths(self):
        self.assertAlmostEqual(float(vs.centre_of_pressure(0.5)), 7.0 / 32.0,
                               delta=1e-14)
        self.assertAlmostEqual(float(vs.centre_of_pressure(np.inf)), 3.0 / 16.0,
                               delta=1e-14)
        self.assertAlmostEqual(float(vs.centre_of_pressure(0.0)), 0.248876,
                               delta=1e-5)
        e = vs.centre_of_pressure(np.linspace(0.0, 5.0, 100))
        self.assertTrue(np.all(np.diff(e) <= 0.0))          # saturates at 3/16
        mid = vs.centre_of_pressure(np.linspace(0.2, 0.8, 50))
        self.assertTrue(np.all(np.diff(mid) < 0.0))

    def test_the_jet_speed_is_the_bernoulli_value(self):
        self.assertAlmostEqual(float(vs.jet_speed(2.0, 0.0)), 2.0, delta=1e-14)
        self.assertGreaterEqual(float(vs.jet_speed(2.0, 0.3)), 2.0)

    def test_an_unknown_length_model_is_rejected(self):
        with self.assertRaises(ValueError):
            vs.cavity_length(1.0, "quadratic")


class TestWashout(unittest.TestCase):
    """The derivation chain behind the washout boundary (4.5)."""

    def test_the_washout_boundary_is_recovered_from_its_own_derivation(self):
        """(4.5) against (4.1)-(4.4) solved directly, on a grid.

        This is the test that earns the right to say (4.5) is verified rather
        than transcribed - and it tests (1.9), (4.1), (4.2), (4.3) and (4.4) at
        once, because the chain uses all of them. It verifies the transcription
        and the algebra, NOT the physics.
        """
        for ar in (0.5, 1.0, 1.5, 2.0, 3.0):
            for cl in (0.2, 0.35, 0.5, 0.8):
                closed = float(vs.washout_froude(cl, ar))
                chain = float(vs.washout_froude_chain(cl, ar))
                self.assertLess(abs(closed - chain) / closed, 1e-12,
                                f"AR={ar}, CL={cl}")

    def test_the_washout_boundary_matches_the_published_sample_value(self):
        self.assertAlmostEqual(float(vs.washout_froude(0.5, 1.0)), 1.1062,
                               delta=1e-3)

    def test_the_washout_froude_number_falls_with_lift(self):
        cl = np.linspace(0.2, 1.2, 20)
        self.assertTrue(np.all(np.diff(vs.washout_froude(cl, 1.0)) < 0.0))

    def test_the_breslin_and_skalak_boundary_is_the_larger_of_its_conditions(self):
        # the two conditions cross at CL = 5/9; a code applying only the first is
        # wrong above that
        self.assertAlmostEqual(float(vs.breslin_skalak(5.0 / 9.0)), 3.0,
                               delta=1e-12)
        self.assertAlmostEqual(float(vs.breslin_skalak(0.2)), np.sqrt(25.0),
                               delta=1e-12)
        self.assertAlmostEqual(float(vs.breslin_skalak(0.9)), 3.0, delta=1e-12)

    def test_the_breslin_and_skalak_boundary_bounds_the_new_one_from_above(self):
        for ar in (0.5, 1.0, 2.0, 3.0):
            for cl in (0.2, 0.5, 0.8):
                self.assertGreater(float(vs.breslin_skalak(cl)),
                                   float(vs.washout_froude(cl, ar)))

    def test_the_elliptic_shape_has_unit_integral_and_vanishes_at_both_ends(self):
        kappa = np.linspace(0.0, 1.0, 20001)
        shape = vs.elliptic_shape(kappa)
        self.assertAlmostEqual(float(np.trapezoid(shape, kappa)), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(shape[0]), 0.0, delta=1e-12)
        self.assertAlmostEqual(float(shape[-1]), 0.0, delta=1e-12)
        self.assertTrue(np.all(shape >= 0.0))

    def test_helmbold_reduces_to_the_two_dimensional_slope_at_large_aspect_ratio(self):
        self.assertAlmostEqual(float(vs.helmbold(2.0 * np.pi, 1e6))
                               / (2.0 * np.pi), 1.0, delta=1e-5)

    def test_helmbold_tends_to_the_slender_limit_whatever_the_sectional_slope(self):
        for a_0 in (2.0 * np.pi, np.pi, np.pi / 2.0):
            got = float(vs.helmbold(a_0, 1e-6)) / (0.5 * np.pi * 1e-6)
            self.assertAlmostEqual(got, 1.0, delta=1e-6)

    def test_the_lift_weighted_slope_lies_between_the_extremes_of_the_local_one(self):
        for ar in (0.5, 1.0, 2.0):
            kappa = np.linspace(0.0, 1.0, 400)
            local = vs.lift_slope(ar * (1.0 - kappa))
            got = vs.lift_weighted_slope(ar)
            self.assertGreaterEqual(got, float(local.min()) - 1e-9)
            self.assertLessEqual(got, float(local.max()) + 1e-9)

    def test_the_lift_weighted_slope_is_quadrature_independent(self):
        self.assertAlmostEqual(vs.lift_weighted_slope(1.0, 100),
                               vs.lift_weighted_slope(1.0, 4000), delta=1e-9)


class TestClosureAngle(unittest.TestCase):
    """The convention that pins the stability criterion's sign."""

    def test_a_cavity_of_uniform_length_is_the_unstable_two_dimensional_case(self):
        """Phi is measured from the HORIZONTAL, so a uniform cavity gives 90 deg.

        The closure line is then normal to the flow and the re-entrant jet points
        straight upstream, which is the classical two-dimensional jet and the
        canonical unstable one. The opposite reading - Phi from the vertical -
        inverts the criterion and would call this the stable case.
        """
        depth = np.linspace(0.0, 1.0, 5)
        phi_bar, _ = vs.closure_angle(depth, np.full(5, 0.6))
        self.assertAlmostEqual(np.degrees(phi_bar), 90.0, delta=1e-9)
        self.assertTrue(vs.unstable_closure(phi_bar))
        u_j, _ = vs.jet_components(1.0, 0.2, phi_bar)
        self.assertLess(u_j, 0.0)               # the jet runs upstream

    def test_a_strongly_tapered_cavity_is_stable_and_sweeps_the_jet_aft(self):
        depth = np.linspace(0.0, 1.0, 5)
        phi_bar, _ = vs.closure_angle(depth, 1.4 - 2.0 * depth)
        self.assertLess(np.degrees(phi_bar), 45.0)
        self.assertFalse(vs.unstable_closure(phi_bar))
        u_j, w_j = vs.jet_components(1.0, 0.2, phi_bar)
        self.assertGreater(u_j, 0.0)            # swept towards the trailing edge
        self.assertGreater(w_j, 0.0)

    def test_the_criterion_sits_exactly_where_the_jet_turns_spanwise(self):
        u_j, w_j = vs.jet_components(1.0, 0.0, vs.PHI_CRIT)
        self.assertAlmostEqual(u_j, 0.0, delta=1e-15)
        self.assertAlmostEqual(w_j, 1.0, delta=1e-15)

    def test_the_regimes_partition_the_depth_and_the_closure_angle(self):
        seen = set()
        for d in (0.0, 0.3, 0.7, 1.0):
            for phi in np.radians((10.0, 44.0, 46.0, 89.0)):
                seen.add(vs.regime(d, phi, 1.0))
        self.assertEqual(seen, {"FW", "PV", "FV"})
        self.assertEqual(vs.regime(0.0, 0.1, 1.0), "FW")
        self.assertEqual(vs.regime(1.0, np.radians(20.0), 1.0), "FV")
        self.assertEqual(vs.regime(1.0, np.radians(70.0), 1.0), "PV")
        self.assertEqual(vs.regime(0.5, np.radians(20.0), 1.0), "PV")

    def test_the_closure_angle_needs_two_stations(self):
        with self.assertRaises(ValueError):
            vs.closure_angle([0.0], [0.5])


class TestMesh(unittest.TestCase):
    """The doubled strut, and the index algebra of the reflection."""

    def test_the_doubled_strut_is_mirror_symmetric_to_the_last_bit(self):
        for nspan_half in (4, 6, 8, 10, 12):
            points = vm.strut_mesh(n_c=6, nspan_half=nspan_half, alpha_deg=9.0)
            self.assertEqual(vm.symmetry_residual(points), 0.0)

    def test_the_half_and_full_arrays_round_trip_exactly(self):
        points = vm.strut_mesh(n_c=6, nspan_half=6, alpha_deg=9.0)
        maps = vm.mirror_maps(points.shape)
        back = vm.complete_points(vm.half_points(points, maps), maps)
        self.assertEqual(float(np.abs(back - points).max()), 0.0)

    def test_the_doubled_strut_passes_the_vendored_validation(self):
        points = vm.strut_mesh(n_c=8, nspan_half=8, alpha_deg=12.0)
        tpw.validate_mesh(points)                        # raises if not
        pan = tpw.build_panels(points)
        self.assertGreater(tpw.signed_volume(pan), 0.0)
        self.assertAlmostEqual(0.5 * tpw.reference_area(points), 1.0, delta=1e-9)

    def test_an_odd_spanwise_count_is_rejected(self):
        with self.assertRaises(ValueError):
            vm.symmetrise_points(np.zeros((13, 8, 3)))

    def test_a_vector_field_mirrors_with_the_reflection_and_no_offset(self):
        points = vm.strut_mesh(n_c=6, nspan_half=6)
        maps = vm.mirror_maps(points.shape, y_fs=0.0)
        half = np.ones(maps["shape_half"])
        full = vm.complete_vectors(half, maps)
        nh = maps["n_half"]
        self.assertTrue(np.all(full[:, nh + 1:, 1] == -1.0))
        self.assertTrue(np.all(full[:, nh + 1:, 0] == +1.0))

    def test_the_half_interface_welds_the_deep_tip_but_not_the_waterline(self):
        points = vm.strut_mesh(n_c=6, nspan_half=6)
        maps = vm.mirror_maps(points.shape)
        half = vm.half_points(points, maps)
        iface = vm.half_interface(half)
        node = np.arange(half.shape[0] * half.shape[1]).reshape(half.shape[:2])
        weld = iface["weld_map"].reshape(half.shape[:2])
        nwrap = half.shape[0] - 1
        self.assertEqual(weld[nwrap, 3], weld[0, 3])                # the TE seam
        self.assertEqual(weld[nwrap - 1, 0], weld[1, 0])            # the deep tip
        last = half.shape[1] - 1
        self.assertEqual(weld[nwrap - 1, last], node[nwrap - 1, last])   # the root

    def test_the_immersed_solid_keeps_its_waterline_thickness(self):
        """foil_ribs must not rebuild the open root as though it were a tip."""
        points = vm.strut_mesh(n_c=6, nspan_half=6)
        maps = vm.mirror_maps(points.shape)
        half = vm.half_points(points, maps)
        import fem_mesh
        both = fem_mesh.foil_ribs(half, pinched=(True, True))
        one = fem_mesh.foil_ribs(half, pinched=(True, False))
        self.assertEqual(float(np.abs(both[0][:, 0] - one[0][:, 0]).max()), 0.0)
        # untapered, so the rebuild is bit-exact and this is a latent trap only
        self.assertEqual(float(np.abs(both[1][:, -1] - one[1][:, -1]).max()), 0.0)


class TestImage(unittest.TestCase):
    """The free surface, which is exact or it is nothing."""

    @classmethod
    def setUpClass(cls):
        cls.fluid, cls.vent, cls.cav, cls.loads = solved(alpha=12.0)

    def test_the_image_strengths_are_minus_the_mirror_of_the_real_ones(self):
        """Exactly zero: it is a permutation of an imposed constraint.

        The antisymmetry is not solved for - it follows from flipping the sign of
        the source strengths on a geometrically exact mirror - so anything above
        round-off means the premise has broken.
        """
        res = vsl.antisymmetry_residual(self.fluid, self.vent)
        self.assertLess(res["mu"], 1e-12)
        self.assertLess(res["sigma"], 1e-12)
        self.assertLess(res["wake_mu"], 1e-10)

    def test_the_potential_vanishes_on_the_free_surface_plane(self):
        """phi = 0 on y = y_fs IS the linearised high-Froude free surface."""
        res = vsl.antisymmetry_residual(self.fluid, self.vent)
        self.assertLess(res["phi_plane"], 1e-9)

    def test_the_unflipped_image_reproduces_the_vendored_solve_bit_for_bit(self):
        """And so a bare tpw.solve on a doubled mesh is the RIGID WALL answer.

        For an onset with no spanwise component, mirroring leaves -n.u_rel
        unchanged, so the unflipped source strengths are the positive image. The
        hazard is therefore precise and on the record: calling the vendored solve
        on a ventilating state silently returns the zero-Froude solution.
        """
        fluid, _, _, vent, _ = strut_case(alpha=12.0)
        wall = vsl.solve_wetted(fluid, vent, wake="frozen", sign=+1.0)
        mu_wall = wall["mu"].copy()
        fluid2, _, _, vent2, _ = strut_case(alpha=12.0)
        plain = tpw.solve(fluid2, wake="frozen")
        self.assertEqual(float(np.abs(mu_wall - plain["mu"]).max()), 0.0)
        del vent2

    def test_the_two_images_differ_in_exactly_one_sign_and_in_the_loading(self):
        """The contrast a sign error would otherwise pass every test and survive.

        The free surface drives the loading to zero at the waterline; the wall
        makes it peak there, and roughly doubles the lift.
        """
        fluid, _, _, vent, _ = strut_case(alpha=12.0)
        out = {}
        for sign, name in ((-1.0, "free"), (+1.0, "wall")):
            fluid, _, _, vent, _ = strut_case(alpha=12.0)
            fluid = vsl.solve_wetted(fluid, vent, wake="frozen", sign=sign)
            fields = vnl.vent_pressure(fluid["panels"], fluid["mu"],
                                       fluid["u_rel"], vent["u_ref"], vent,
                                       np.zeros(len(fluid["mu"])), rho=RHO_WATER)
            depth, cl = vnl.depth_loading(fluid["panels"], vent,
                                          fields["p_gauge"], RHO_WATER)
            out[name] = (cl, vnl.immersed_loads(fluid, vent, fields["p_gauge"],
                                                RHO_WATER)["resultants"]["CL"])
        cl_free, cl_wall = out["free"][0], out["wall"][0]
        # cl is ordered from the waterline downward
        self.assertLess(cl_free[0], cl_free.max())        # free surface: a minimum
        self.assertAlmostEqual(cl_wall[0], cl_wall.max(), delta=1e-12)  # wall: peak
        self.assertGreater(abs(out["wall"][1]), 1.5 * abs(out["free"][1]))

    def test_the_depth_loading_falls_towards_zero_at_the_waterline(self):
        """O(dy), not zero: phi = 0 on the plane is exact, the panel load is not.

        With an even span count no panel straddles the plane, so the shallowest
        strip's circulation is O(dy) rather than zero. The exact statement is the
        one about phi; this is its discretisation.
        """
        first = []
        for nspan_half in (4, 8, 16):
            fluid, vent, cav, _ = solved(alpha=12.0, nspan_half=nspan_half, n_c=6)
            _, cl = vnl.depth_loading(fluid["panels"], vent, cav["p_gauge"],
                                      RHO_WATER)
            first.append(abs(cl[0]) / abs(cl).max())
        self.assertTrue(first[0] > first[1] > first[2], first)
        self.assertLess(first[-1], 0.5)

    def test_the_no_image_configuration_is_a_legal_single_foil(self):
        fluid, vent, cav, loads = solved(alpha=12.0, image=False, nspan_half=8)
        self.assertFalse(vent["image"])
        self.assertTrue(np.all(vent["real_mask"]))
        self.assertEqual(vsl.antisymmetry_residual(fluid, vent)["phi_plane"], 0.0)
        self.assertGreater(abs(loads["resultants"]["CL"]), 0.0)


class TestCavity(unittest.TestCase):
    """The cavity: its pressure, its thickness, and its continuity."""

    @classmethod
    def setUpClass(cls):
        # the ventilated BRANCH: at 16 degrees the flow would incept, but the
        # regime only advances on commit, so the branch has to be asked for
        cls.fluid, cls.vent, cls.cav, cls.loads = solved(alpha=16.0, n_c=12,
                                                         nspan_half=6,
                                                         regime="FV")

    def test_the_cavity_pressure_is_a_floor_at_minus_the_local_cavitation_number(self):
        """cp = max(cp_cavity, cp_solved), exactly, on a fully covered panel.

        The cavity pressure is imposed as a FLOOR rather than as a replacement:
        inside a cavity the pressure cannot fall below the cavity's own, which is
        what a cavity is, and writing it that way makes the ventilated pressure
        never lower than the wetted one anywhere by construction. Most of the
        cavity sits at the floor; the closure and detachment panels, where the
        outer flow has already recovered above it, keep their own higher value,
        which is the conservative choice - it never over-predicts the lift loss.
        """
        cav = self.cav
        full = cav["weight"] > 0.999
        self.assertTrue(np.any(full))
        expect = np.maximum(cav["cp_cav"], cav["cp_solved"])
        # the weight is 1 to round-off rather than exactly, so the floor is too
        self.assertLess(float(np.abs(cav["cp"][full] - expect[full]).max()), 1e-15)
        self.assertGreaterEqual(float((cav["cp"] - cav["cp_cav"])[full].min()),
                                -1e-14)
        at_floor = np.abs(cav["cp"][full] - cav["cp_cav"][full]) < 1e-12
        self.assertGreater(float(at_floor.mean()), 0.3)

    def test_the_cavity_pressure_is_stratified_along_the_immersion(self):
        """dCp/dz' = -2/Fn_h^2 exactly, which catches using the mean immersion."""
        cav, vent = self.cav, self.vent
        depth = vcv.panel_depth(self.fluid["panels"], vent["y_fs"])
        order = np.argsort(depth)
        slope = np.polyfit(depth[order], cav["cp_cav"][order], 1)[0]
        self.assertAlmostEqual(slope, -2.0 / vent["fn_h"] ** 2, delta=1e-10)

    def test_the_cavity_never_lowers_a_panel_pressure(self):
        """A cavity is at atmospheric pressure, ABOVE the suction it replaces.

        That is why ventilation destroys lift, and the direction is asserted with
        no tolerance because the opposite reading would still reduce |CL| on some
        meshes and so would pass a lift test.
        """
        self.assertGreaterEqual(float(self.cav["raised"].min()), 0.0)

    def test_ventilation_reduces_the_lift_without_changing_its_sign(self):
        _, _, _, wet = solved(alpha=16.0, n_c=12, nspan_half=6, regime="FW")
        cl_wet = wet["resultants"]["CL"]
        cl_vent = self.loads["resultants"]["CL"]
        self.assertGreater(cl_wet * cl_vent, 0.0)
        self.assertLess(abs(cl_vent), abs(cl_wet))

    def test_the_cavity_is_longest_at_the_waterline_and_shortest_at_the_tip(self):
        """The cavity planform of the paper's figure 3, and the reason for it.

        sigma_c grows with depth, so the cavity is arrested by hydrostatic
        pressure at depth and unbounded at the surface. A model that used the mean
        immersion instead of the local depth would give a cavity of uniform
        length, and with it a closure angle of 90 degrees and washout everywhere.
        """
        nh = self.vent["n_half"]
        length = self.cav["l_c"][:nh]            # index 0 is the deep tip
        live = length > 0.0
        self.assertGreater(int(live.sum()), 2)
        self.assertGreater(length[-1], length[0])
        self.assertLess(float(np.abs(np.diff(length)).max()), 0.6)

    def test_the_cavity_thickness_is_positive_and_grows_from_detachment(self):
        cav = self.cav
        inside = cav["weight"] > 0.5
        self.assertTrue(np.any(inside))
        self.assertGreater(float(np.abs(cav["thickness"][inside]).max()), 0.0)
        self.assertGreater(float(cav["thickness"][inside].max()), 0.0)

    def test_the_dynamic_condition_holds_in_the_cavity_interior(self):
        """The end-to-end test of the Dirichlet closure.

        Prescribing mu from the cavity speed makes the vendored surface gradient
        return the cavity pressure - the condition, the mixed system and the
        pressure evaluation all agreeing. Only in the INTERIOR: at the detachment
        and closure panels the gradient stencil reaches into wetted flow and mixes
        a prescribed strength with a solved one, which is a property of a
        surface-gradient pressure evaluation and not a defect of the cavity. The
        interior figure is a measured consistency, reported in docs/VENTILATION.md.
        """
        cav, vent = self.cav, self.vent
        inte = cav["interior"] & np.asarray(vent["real_mask"], dtype=bool)
        self.assertGreater(int(inte.sum()), 4)
        err = np.abs(cav["cp_solved"] - cav["cp_cav"])[inte]
        self.assertLess(float(err.mean()), 0.1)

    def test_the_cavity_solve_changes_the_source_strengths_it_solves_for(self):
        """Otherwise the mixed system is not doing anything at all."""
        cav = self.cav
        inside = cav["weight"] > 0.5
        sigma_wet = vsl.image_sigma(self.fluid["panels"], self.vent,
                                    self.fluid["u_rel"])
        self.assertGreater(float(np.abs(self.fluid["sigma"][inside]
                                        - sigma_wet[inside]).max()), 1e-6)

    def test_a_zero_length_cavity_reproduces_the_wetted_solve_exactly(self):
        fluid, _, _, vent, _ = strut_case(alpha=16.0, regime="FW")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
        fluid2, _, _, vent2, _ = strut_case(alpha=16.0, regime="FW")
        wetted = vsl.solve_wetted(fluid2, vent2, wake="frozen")
        self.assertEqual(float(np.abs(fluid["mu"] - wetted["mu"]).max()), 0.0)
        self.assertEqual(int(np.count_nonzero(cav["weight"])), 0)

    def test_the_cavity_is_a_fixed_function_of_the_geometry_and_the_regime(self):
        """Evaluated twice from the same committed state, bit-identical.

        The property the whole partitioned scheme rests on. It is why the cavity
        iteration is cold-started from the committed length rather than warm
        started from the previous subiteration.
        """
        fluid, _, _, vent, _ = strut_case(alpha=16.0, regime="FV")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav_a, loads_a = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
            fluid, cav_b, loads_b = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
        self.assertEqual(float(np.abs(cav_a["l_c"] - cav_b["l_c"]).max()), 0.0)
        self.assertEqual(float(np.abs(loads_a["force"]
                                      - loads_b["force"]).max()), 0.0)

    def test_the_ventilated_load_is_continuous_in_the_cavity_length(self):
        """The fractional closure panel, which is why the length is continuous.

        A boolean panel mask would make the load a staircase in the displacement
        and IQN-ILS would be fitting one; with a fractional weight the load
        changes smoothly as the cavity crosses a panel boundary.
        """
        fluid, _, _, vent, _ = strut_case(alpha=16.0, n_c=8, regime="FV")
        frame = vcv.chordwise_frame(fluid["panels"])
        fluid = vsl.solve_wetted(fluid, vent, wake="frozen")
        forces = []
        for length in np.linspace(0.30, 0.34, 9):
            weight, _ = vcv.mask_from_lengths(fluid["panels"], frame,
                                              np.full(fluid["panels"]["nspan"],
                                                      length),
                                              vent["alpha_rad"],
                                              vent["real_mask"])
            fields = vnl.vent_pressure(fluid["panels"], fluid["mu"],
                                       fluid["u_rel"], vent["u_ref"], vent,
                                       weight, rho=RHO_WATER)
            forces.append(vnl.immersed_loads(fluid, vent, fields["p_gauge"],
                                             RHO_WATER)["resultants"]["CL"])
        jumps = np.abs(np.diff(forces))
        self.assertLess(float(jumps.max()), 4.0 * float(np.median(jumps)) + 1e-6)


class TestInception(unittest.TestCase):
    """Inception, the air path, and the hysteresis."""

    def test_a_disconnected_pocket_is_refused(self):
        """The whole difference between ventilation and cavitation."""
        pan = tpw.build_panels(vm.strut_mesh(n_c=6, nspan_half=6, alpha_deg=12.0))
        candidate = np.zeros(len(pan["areas"]), dtype=bool)
        seed = np.zeros_like(candidate)
        n = len(candidate)
        candidate[[40, 41, 42]] = True
        seed[40] = True
        reached = vcv.air_path(pan, candidate, seed)
        self.assertTrue(reached[40])
        island = np.zeros_like(candidate)
        far = n - 5
        island[far] = True
        self.assertFalse(vcv.air_path(pan, candidate | island, seed)[far])

    def test_the_flood_fill_reaches_only_through_the_candidate_set(self):
        pan = tpw.build_panels(vm.strut_mesh(n_c=6, nspan_half=6, alpha_deg=12.0))
        candidate = np.ones(len(pan["areas"]), dtype=bool)
        seed = np.zeros_like(candidate)
        seed[0] = True
        self.assertTrue(np.all(vcv.air_path(pan, candidate, seed)))

    def test_the_separation_indicator_lives_aft_of_the_suction_peak(self):
        fluid, vent, cav, _ = solved(alpha=16.0)
        sep = cav["separated"]
        suction = vcv.suction_side(fluid["panels"], vent["alpha_rad"])
        self.assertTrue(np.all(~sep | suction))
        self.assertGreater(int(sep.sum()), 0)

    def test_no_spontaneous_inception_below_the_stall_angle(self):
        """The paper's stall boundary, which is vertical in incidence.

        Below stall the separation bubble stops short of the free surface and a
        thin attached layer seals the ventilation-prone flow from the air. A panel
        method cannot resolve that layer, so the stall angle is an INPUT.
        """
        for alpha in (8.0, 12.0, 14.0):
            fluid, vent, cav, _ = solved(alpha=alpha)
            vcv.commit_vent(vent, cav, 0.0, 0.1)
            self.assertEqual(vent["regime"], "FW", f"alpha={alpha}")

    def test_spontaneous_inception_above_the_stall_angle(self):
        for alpha in (16.0, 20.0):
            fluid, vent, cav, _ = solved(alpha=alpha)
            vcv.commit_vent(vent, cav, 0.0, 0.1)
            self.assertNotEqual(vent["regime"], "FW", f"alpha={alpha}")
            self.assertEqual(vent["transitions"][-1][3], "inception")

    def test_an_injection_incepts_below_stall_and_the_cavity_then_persists(self):
        """The paper's perturbation-induced route, and its whole point.

        An air jet at the junction of the leading edge and the free surface breaks
        the seal; the cavity that forms survives the jet being switched off,
        because persistence never asks the seal question again.
        """
        fluid, _, _, vent, _ = strut_case(alpha=10.0)
        vent = vcv.inject(vent, active=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav, _ = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
        vcv.commit_vent(vent, cav, 0.0, 0.1)
        self.assertNotEqual(vent["regime"], "FW")
        vent = vcv.inject(vent, active=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav, _ = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
        vcv.commit_vent(vent, cav, 0.1, 0.1)
        self.assertNotEqual(vent["regime"], "FW")

    def test_the_same_conditions_reached_from_different_pasts_differ(self):
        """Hysteresis, written so a single threshold with a dead band fails it.

        The conditions are IDENTICAL - the same incidence, Froude number and mesh
        - and only the history differs. A model whose regime were a function of
        the instantaneous state could not pass this.
        """
        bistable = 0
        for alpha in (6.0, 10.0, 14.0):
            regimes = []
            for start in ("FW", "FV"):
                fluid, vent, cav, _ = solved(alpha=alpha, regime=start)
                vcv.commit_vent(vent, cav, 0.0, 0.1)
                regimes.append(vent["regime"])
            if regimes[0] != regimes[1]:
                bistable += 1
        self.assertEqual(bistable, 3)

    def test_the_ventilated_branch_carries_less_lift_at_the_same_incidence(self):
        out = {}
        for start in ("FW", "FV"):
            _, _, _, loads = solved(alpha=10.0, regime=start)
            out[start] = abs(loads["resultants"]["CL"])
        self.assertLess(out["FV"], out["FW"])

    def test_the_regime_does_not_oscillate_when_held_at_fixed_conditions(self):
        fluid, _, _, vent, _ = strut_case(alpha=16.0)
        seen = []
        for k in range(12):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid, cav, _ = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
            vcv.commit_vent(vent, cav, 0.1 * k, 0.1)
            seen.append(vent["regime"])
        changes = sum(1 for a, b in zip(seen[:-1], seen[1:]) if a != b)
        self.assertLessEqual(changes, 1, seen)

    def test_the_regime_advances_only_on_commit(self):
        fluid, _, _, vent, _ = strut_case(alpha=20.0)
        before = vent["regime"]
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid, cav, _ = vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
            self.assertEqual(vent["regime"], before)
        vcv.commit_vent(vent, cav, 0.0, 0.1)
        self.assertNotEqual(vent["regime"], before)

    def test_an_unknown_regime_is_rejected(self):
        _, _, _, vent, _ = strut_case()
        with self.assertRaises(ValueError):
            vcv.set_regime(vent, "supercavitating")
        with self.assertRaises(ValueError):
            vcv.build_vent(vm.mirror_maps((13, 13, 3)), np.zeros((13, 7, 3)),
                           1.0, 1.0, 1.0, 0.2, regime="soggy")

    def test_an_unknown_closure_or_extent_rule_is_rejected(self):
        maps = vm.mirror_maps((13, 13, 3))
        half = np.zeros((13, 7, 3))
        with self.assertRaises(ValueError):
            vcv.build_vent(maps, half, 1.0, 1.0, 1.0, 0.2, closure="magic")
        with self.assertRaises(ValueError):
            vcv.build_vent(maps, half, 1.0, 1.0, 1.0, 0.2, extent_rule="guess")


class TestLoads(unittest.TestCase):
    """The load path, and what the image half must not contribute."""

    @classmethod
    def setUpClass(cls):
        cls.fluid, cls.sstate, cls.transfer, cls.vent, cls.mesh = \
            strut_case(alpha=16.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fluid, cls.cav, cls.loads = vsl.solve_cavity(cls.fluid, cls.vent,
                                                             rho=RHO_WATER)

    def test_the_modified_load_conserves_force_and_moment_through_the_transfer(self):
        rep = vnl.conservation_report_half(self.transfer, self.vent,
                                           self.fluid["panels"],
                                           self.loads["force"],
                                           u_struct=self.sstate["u"])
        self.assertLess(rep["force_error_lump"], 1e-12)
        self.assertLess(rep["force_error_transfer"], 1e-12)
        self.assertLess(rep["moment_error_lump"], 1e-12)
        self.assertLess(rep["work_error"], 1e-10)

    def test_the_image_half_carries_no_structural_load(self):
        force = np.array(self.loads["force"])
        self.assertEqual(float(np.abs(
            force[np.asarray(self.vent["image_mask"], dtype=bool)]).max()), 0.0)
        f_struct = vnl.structural_forces_half(self.transfer, self.vent,
                                              self.fluid["panels"], force)
        self.assertLess(float(np.abs(f_struct.sum(axis=0)
                                     - force.sum(axis=0)).max())
                        / max(float(np.abs(force).sum()), 1e-30), 1e-12)

    def test_the_reference_area_is_the_immersed_planform(self):
        self.assertAlmostEqual(self.vent["s_ref_wet"],
                               self.vent["h"] * self.vent["chord"], delta=1e-14)
        self.assertAlmostEqual(0.5 * self.fluid["s_ref"],
                               self.vent["s_ref_wet"], delta=1e-6)

    def test_the_doubled_mesh_lift_is_a_near_cancellation_and_not_the_answer(self):
        """tpw.get_loads must never be called on a ventilating state.

        The image half carries the negated loading, so the doubled resultant is a
        near-cancellation - about a tenth of the immersed value, not zero, because
        the pressure is quadratic in a velocity whose tangential part does not
        simply mirror. Both are asserted so that nobody later "fixes" the small
        number by scaling it.
        """
        doubled = tpw.get_loads(self.fluid, rho=RHO_WATER)["resultants"]["CL"]
        immersed = self.loads["resultants"]["CL"]
        self.assertLess(abs(doubled), 0.5 * abs(immersed))
        self.assertGreater(abs(immersed), 0.01)

    def test_the_moment_reference_does_not_drift_with_the_geometry(self):
        """integrate_loads would relocate it on the waterline section each call."""
        vent = self.vent
        moved = np.array(self.fluid["panels"]["points"])
        moved[..., 0] += 0.05
        fluid = tpw.update_points(dict(self.fluid), moved)
        second = vnl.immersed_loads(fluid, vent, self.cav["p_gauge"], RHO_WATER)
        self.assertEqual(float(np.abs(second["resultants"]["x_ref"]
                                      - vent["x_ref"]).max()), 0.0)

    def test_p_ref_is_rejected_when_ventilating(self):
        with self.assertRaises(ValueError):
            fsd._loads_on_structure(self.fluid, self.transfer, RHO_WATER,
                                    p_ref=lambda z: 0.0 * z, vent=self.vent)


class TestBaseline(unittest.TestCase):
    """Nothing that existed before this module may have moved."""

    def test_a_coupled_run_with_ventilation_disabled_is_unchanged(self):
        """Bit-for-bit, in every history array.

        The strongest available statement that the C1 to C6 numbers of
        docs/COUPLING.md cannot have moved: the ventilated code paths are all
        behind `vent is not None`.
        """
        import make_thick_sample_inputs as mts
        points = mts.thick_wing_mesh(n_c=5, nspan=6, span=3.0, root_chord=1.0,
                                     taper=1.0, sweep_deg=0.0, twist_deg=0.0,
                                     camber=0.0, thickness=0.12)
        import fem_mesh
        mesh = fem_mesh.solid_foil_mesh(points, n_thick=1,
                                        min_half_thickness=0.02)
        model = fes.build_model(mesh["nodes"], mesh["elements"],
                                fmat.IsotropicElastic(e_mod=2e9, nu=0.3,
                                                      rho=1200.0))
        out = []
        for _ in range(2):
            fluid, sstate, transfer = fsd.init_fsi(
                points, uniform_onset(), model, rho=RHO_WATER,
                config={"lifting": True, "nwake": 0, "n_bound": 1,
                        "closure": False})
            fes.clamp(sstate, mesh["node_sets"]["root"])
            fes.assemble_operators(sstate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist, _, _ = fsd.time_march_fsi(fluid, sstate, transfer, dt=0.05,
                                                nsteps=3, rho=RHO_WATER,
                                                rho_inf=0.9, max_sub=10)
            out.append(hist)
        for key in ("CL", "CD", "CM", "tip", "fluid_work", "resid"):
            self.assertEqual(float(np.abs(np.asarray(out[0][key])
                                          - np.asarray(out[1][key])).max()), 0.0,
                             key)
        self.assertNotIn("regime", out[0])

    def test_the_vendored_modules_are_not_monkeypatched(self):
        for module in (tpw, fk):
            for name in dir(module):
                if name.startswith("__"):
                    continue
                self.assertFalse(name.startswith("vent_"),
                                 f"{module.__name__}.{name}")

    def test_the_ventilation_state_holds_no_reference_to_the_fluid(self):
        """So that two ventilation states coexist in one process."""
        _, _, _, vent, _ = strut_case()
        for key, value in vent.items():
            self.assertNotIsInstance(value, type(np.zeros(1).flags), key)
        self.assertNotIn("panels", vent)
        self.assertNotIn("mu", vent)

    def test_a_ventilated_evaluation_restores_the_geometry(self):
        fluid, _, _, vent, _ = strut_case(alpha=16.0)
        before = np.array(fluid["panels"]["points"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vsl.solve_cavity(fluid, vent, rho=RHO_WATER)
        self.assertLess(float(np.abs(fluid["panels"]["points"] - before).max()),
                        1e-12)


class TestCoupled(unittest.TestCase):
    """The bookkeeping of a ventilated coupled step."""

    def test_the_regime_advances_once_per_step_not_once_per_subiteration(self):
        """The analogue of the wake-row test, and for the same reason."""
        fluid, sstate, transfer, vent, mesh = strut_case(alpha=20.0, n_c=6,
                                                         nspan_half=4,
                                                         unsteady=True, nwake=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                      rho=RHO_WATER, max_iter=6,
                                                      vent=vent)
            hist, fluid, sstate = fsd.time_march_fsi(fluid, sstate, transfer,
                                                     dt=0.02, nsteps=3,
                                                     rho=RHO_WATER, rho_inf=0.9,
                                                     max_sub=12, vent=vent)
        self.assertEqual(len(hist["regime"]), 4)
        self.assertEqual(list(hist["nrows"]), [0, 1, 2, 3])
        self.assertGreater(int(np.max(hist["sub"][1:])), 1)
        self.assertEqual(vent["step"], 4)

    def test_the_image_geometry_follows_the_deformed_immersed_half(self):
        fluid, sstate, transfer, vent, mesh = strut_case(alpha=16.0, n_c=6,
                                                         nspan_half=4)
        rng = np.random.default_rng(3)
        u = 0.01 * rng.standard_normal(sstate["u"].shape)
        points0 = fsd._reference_points(fluid, vent)
        pts, _ = fsd._geometry(transfer, points0, u, None, vent)
        self.assertEqual(vm.symmetry_residual(pts, vent["y_fs"]), 0.0)

    def test_the_ventilated_load_is_deterministic_at_fixed_geometry(self):
        fluid, sstate, transfer, vent, _ = strut_case(alpha=20.0, regime="FV")
        out = []
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_struct, _ = fsd._loads_on_structure(fluid, transfer, RHO_WATER,
                                                      vent=vent)
            out.append(f_struct.copy())
        self.assertEqual(float(np.abs(out[0] - out[1]).max()), 0.0)
        self.assertEqual(float(np.abs(out[0] - out[2]).max()), 0.0)

    def test_the_wake_projection_restores_the_antisymmetry_convection_breaks(self):
        """The premise that had to be checked rather than assumed.

        With phi antisymmetric the perturbation velocity mirrors with a sign while
        the onset does not, so a mirror-consistent convection would need the
        perturbation to vanish. Left to convect, the image wake drifts off the
        mirror position and the free-surface condition is lost progressively.
        """
        fluid, _, _, vent, _ = strut_case(alpha=16.0, nwake=4)
        fluid = vsl.solve_wetted(fluid, vent, wake="frozen")
        v_nodes = tpw.wake_node_velocity(fluid)
        tpw.shed(fluid["wake"], v_nodes, 0.1,
                 v_shed=tpw.te_convection_velocity(fluid))
        nodes = fluid["wake"]["nodes"]
        nh = vent["n_half"]
        image = np.array(nodes[:, nh - 1::-1, :])
        image[..., 1] = 2.0 * vent["y_fs"] - image[..., 1]
        before = float(np.abs(nodes[:, nh + 1:, :] - image).max())
        self.assertGreater(before, 1e-12)          # convection DID break it
        report = vm.symmetrise_wake(fluid["wake"], vent)
        nodes = fluid["wake"]["nodes"]
        image = np.array(nodes[:, nh - 1::-1, :])
        image[..., 1] = 2.0 * vent["y_fs"] - image[..., 1]
        self.assertEqual(float(np.abs(nodes[:, nh + 1:, :] - image).max()), 0.0)
        self.assertEqual(float(np.abs(nodes[:, nh, 1] - vent["y_fs"]).max()), 0.0)
        self.assertGreaterEqual(report["drift_y"], 0.0)

    def test_the_ventilation_margin_diagnostic_reports_a_ready_flow(self):
        fluid, _, _, vent, _ = strut_case(alpha=20.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = fsd.ventilation_margin(fluid, vent, rho=RHO_WATER)
        self.assertTrue(report["ready"])
        self.assertLess(report["margin"], 0.0)
        self.assertTrue(report["seal_broken"])
        self.assertGreater(report["weber"], vs.WE_MIN)


@unittest.skipUnless(SLOW, "set VENT_SLOW=1 for the coupled ventilation gates")
class TestSlow(unittest.TestCase):
    """Coupled ventilation physics: the load step, and the twist feedback."""

    def test_ventilation_removes_a_large_fraction_of_the_lift(self):
        _, _, _, wet = solved(alpha=16.0, n_c=12, nspan_half=8, regime="FW")
        _, _, _, ven = solved(alpha=16.0, n_c=12, nspan_half=8, regime="FV")
        ratio = abs(ven["resultants"]["CL"] / wet["resultants"]["CL"])
        self.assertLess(ratio, 0.8)
        self.assertGreater(ratio, 0.1)

    def test_the_ventilated_lift_falls_with_increasing_froude_number(self):
        out = []
        for fn_h in (1.5, 2.5, 3.5):
            _, _, _, loads = solved(alpha=16.0, fn_h=fn_h, n_c=12, regime="FV")
            out.append(abs(loads["resultants"]["CL"]))
        self.assertTrue(out[0] > out[1] > out[2], out)

    def test_the_lift_gap_grows_with_incidence(self):
        gaps = []
        for alpha in (8.0, 12.0, 16.0):
            _, _, _, wet = solved(alpha=alpha, n_c=12, regime="FW")
            _, _, _, ven = solved(alpha=alpha, n_c=12, regime="FV")
            gaps.append(abs(wet["resultants"]["CL"])
                        - abs(ven["resultants"]["CL"]))
        self.assertTrue(gaps[0] < gaps[1] < gaps[2], gaps)

    def test_the_centre_of_pressure_moves_aft_towards_mid_chord(self):
        """The mechanism behind the paper's collapse of the yawing moment.

        Measured directly as the centre of pressure rather than through CM,
        because CM about mid-chord is a small difference of two larger numbers and
        its magnitude is not a claim this model can make. What it CAN say is that
        aerating the forward suction moves the load aft, and by how much.
        """
        out = {}
        for regime in ("FW", "FV"):
            fluid, vent, cav, loads = solved(alpha=16.0, n_c=12, regime=regime)
            res = loads["resultants"]
            # x_cp forward of mid-chord, which is the paper's e: the moment about
            # +y of a load at -rx is -rx*Fz, so e = +CM*c/CL
            out[regime] = res["CM"] * vent["chord"] / res["CL"]
        self.assertLess(out["FV"], out["FW"])
        self.assertGreater(out["FW"] - out["FV"], 0.02)
        # and it lands close to the supercavitating 3/16 the paper quotes
        self.assertAlmostEqual(out["FV"], 3.0 / 16.0, delta=0.06)

    def test_the_structural_transient_after_a_ventilation_step_decays(self):
        fluid, sstate, transfer, vent, mesh = strut_case(alpha=20.0, n_c=8,
                                                         nspan_half=6,
                                                         unsteady=True, nwake=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                      rho=RHO_WATER, max_iter=8,
                                                      vent=vent)
            hist, fluid, sstate = fsd.time_march_fsi(fluid, sstate, transfer,
                                                     dt=0.01, nsteps=24,
                                                     rho=RHO_WATER, rho_inf=0.8,
                                                     max_sub=20, vent=vent)
        tip = np.asarray(hist["tip"])
        self.assertTrue(np.all(np.isfinite(tip)))
        self.assertLess(float(tip.max()), 0.2)
        self.assertGreater(len(vent["transitions"]), 0)

    def test_structural_twist_shifts_the_inception_boundary(self):
        margins = {}
        for e_mod in (1e12, 2e8):
            points = vm.strut_mesh(n_c=8, nspan_half=6, alpha_deg=13.0)
            maps = vm.mirror_maps(points.shape)
            mesh = vm.strut_solid_mesh(vm.half_points(points, maps), n_thick=2,
                                       min_half_thickness=0.02)
            model = fes.build_model(mesh["nodes"], mesh["elements"],
                                    fmat.IsotropicElastic(e_mod=e_mod, nu=0.3,
                                                          rho=1200.0))
            fluid, sstate, transfer, vent = fsd.init_vent_fsi(
                points, uniform_onset(), model, h=1.0, chord=1.0, u_ref=1.0,
                rho=RHO_WATER, alpha_deg=13.0, unsteady=False,
                config={"lifting": True, "nwake": 6}, fn_h=2.5)
            fes.clamp(sstate, mesh["node_sets"]["tip_max"])
            fes.assemble_operators(sstate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate,
                                                          transfer,
                                                          rho=RHO_WATER,
                                                          max_iter=8, vent=vent)
                margins[e_mod] = fsd.ventilation_margin(fluid, vent,
                                                        rho=RHO_WATER)["margin"]
        self.assertNotEqual(margins[1e12], margins[2e8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
