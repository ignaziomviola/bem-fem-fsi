"""Tests for the time-dependent driver (standard-library unittest, no pytest).

Run the fast set:      python3 test_unsteady_wing.py
Include the slow set:  UNSTEADY_SLOW=1 python3 test_unsteady_wing.py

The fast set pins the bookkeeping of the wake time step, the transfer of body
motion onto the collocation points, the sign and magnitude of the dmu/dt
pressure term, and the recovery of the steady solution in the long-time limit.
The slow set holds the analytical unsteady gates: the added mass of a sphere,
the Wagner indicial response and the Theodorsen response in plunge. No files
are written and no figures are created.
"""

import os
import unittest

import numpy as np

import thick_panel_wing as tpw
import unsteady_wing as usw
import make_thick_sample_inputs as mts
from freewake_kernels import induced_velocity, pitch_mesh

SLOW = os.environ.get("UNSTEADY_SLOW", "") not in ("", "0")


def uniform_onset(points):
    pts = np.atleast_2d(points)
    v = np.zeros((len(pts), 3))
    v[:, 0] = 1.0
    return v


def still_fluid(points):
    return np.zeros((len(np.atleast_2d(points)), 3))


def small_wing(n_c=6, nspan=6, **kw):
    args = dict(n_c=n_c, nspan=nspan, span=8.0, root_chord=1.0, taper=1.0,
                sweep_deg=0.0, twist_deg=0.0, camber=0.0, thickness=0.12)
    args.update(kw)
    return mts.thick_wing_mesh(**args)


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


class TestWakeTimeStepping(unittest.TestCase):
    """The array bookkeeping the whole time loop rests on."""

    def _wake(self, points, nrows=3):
        steps = np.full(nrows, 0.3)
        wake = tpw.init_wake(tpw.te_nodes(points), uniform_onset, steps,
                             n_bound=1, closure=False, core=0.05)
        wake["mu"][:] = np.arange(1, nrows + 1)[:, None]
        return wake

    def test_shed_invariants(self):
        points = small_wing()
        wake = self._wake(points, nrows=3)
        mu_before = wake["mu"].copy()
        nodes_before = wake["nodes"].copy()
        v = np.tile(np.array([1.0, 0.0, 0.0]),
                    (wake["nodes"].size // 3, 1))
        tpw.shed(wake, v, dt=0.2)
        nrows = wake["mu"].shape[0]
        self.assertEqual(nrows, 4)
        self.assertEqual(wake["nodes"].shape[0], nrows + 1)
        self.assertEqual(len(wake["steps"]), nrows)
        # the history keeps the strengths it was shed with; the new row is zero
        self.assertTrue(np.array_equal(wake["mu"][1:], mu_before))
        self.assertTrue(np.array_equal(wake["mu"][0], np.zeros(wake["mu"].shape[1])))
        # row 0 is untouched (update_points re-pins it); row r+1 is row r moved
        self.assertTrue(np.array_equal(wake["nodes"][0], nodes_before[0]))
        for r in range(nrows):
            self.assertLess(np.abs(wake["nodes"][r + 1]
                                   - (nodes_before[r] + 0.2 * np.array([1, 0, 0]))
                                   ).max(), 1e-14)

    def test_shed_is_material(self):
        """The newly shed node is the fluid particle that was at the trailing
        edge: with a unit onset and a small wing it moves by dt*U downstream."""
        points = small_wing()
        state = usw.init_unsteady_state(points, uniform_onset, 1.0)
        state = tpw.solve(state, wake="frozen")
        te = tpw.te_nodes(points).copy()
        v = tpw.wake_node_velocity(state)
        v_te = tpw.te_convection_velocity(state)
        dt = 0.1
        tpw.shed(state["wake"], v, dt, v_shed=v_te)
        moved = state["wake"]["nodes"][1] - te
        # dominated by the onset; the induced part is a fraction of it
        self.assertLess(abs(moved[3, 0] / dt - 1.0), 0.25)
        self.assertLess(abs(moved[3, 1] / dt), 0.25)
        # and it is not what the singular kernels return at the node itself
        self.assertGreater(np.abs(v_te - v.reshape(-1, len(te), 3)[0]).max(),
                           0.1)

    def test_convect_pins_row_zero(self):
        points = small_wing()
        wake = self._wake(points)
        row0 = wake["nodes"][0].copy()
        v = np.ones((wake["nodes"].size // 3, 3))
        d = tpw.convect(wake, v, dt=0.5)
        self.assertTrue(np.array_equal(wake["nodes"][0], row0))
        self.assertAlmostEqual(d, 0.5 * np.sqrt(3.0), places=12)

    def test_empty_wake_paths(self):
        """The state the driver starts in: nodes only, no rows."""
        points = small_wing()
        state = usw.init_unsteady_state(points, uniform_onset, 1.0)
        wake = state["wake"]
        self.assertEqual(wake["mu"].shape[0], 0)
        self.assertEqual(tpw.wake_quads(wake, uniform_onset).shape, (0, 4, 3))
        p_bound, phi_known = tpw.wake_influence(
            state["panels"]["centroids"], wake, uniform_onset)
        self.assertTrue(np.array_equal(p_bound, np.zeros_like(p_bound)))
        self.assertTrue(np.array_equal(phi_known, np.zeros_like(phi_known)))
        state = tpw.solve(state, wake="frozen")          # must not raise
        self.assertTrue(np.all(np.isfinite(state["mu"])))
        # with no wake there is no circulation: the trailing-edge jump is free
        # and the solution is the acyclic one, so set_bound_strengths is a no-op
        self.assertEqual(state["wake"]["mu"].shape[0], 0)

    def test_truncate_closes_the_cut(self):
        points = small_wing()
        wake = self._wake(points, nrows=6)
        tpw.truncate(wake, 3)
        self.assertEqual(wake["mu"].shape[0], 3)
        self.assertEqual(wake["nodes"].shape[0], 4)
        self.assertEqual(len(wake["steps"]), 3)
        self.assertTrue(wake["closure"])

    def test_views_agree_for_a_row_varying_wake(self):
        """The unsteady path of wake_segments: with history rows of differing
        strength the Biot-Savart velocity must be the gradient of the quad-sum
        potential, once the analytically dropped row-0 segments are restored."""
        points = small_wing()
        wake = self._wake(points, nrows=4)
        rng = np.random.default_rng(11)
        wake["mu"] = rng.standard_normal(wake["mu"].shape)
        wake["core"] = 1e-9
        quads = tpw.wake_quads(wake, uniform_onset)
        mu_flat = wake["mu"].T.ravel()                   # strip-major
        p1, p2, g, core = tpw.wake_segments(wake, uniform_onset)
        n0 = wake["nodes"][0]
        p1 = np.vstack([p1, n0[:-1]])
        p2 = np.vstack([p2, n0[1:]])
        g = np.concatenate([g, wake["mu"][0]])
        core = np.concatenate([core, np.full(wake["mu"].shape[1], 1e-9)])
        eps = 1e-6
        for probe in ([1.6, 0.3, 0.45], [3.0, -1.7, -0.3]):
            probe = np.array(probe)
            grad = np.zeros(3)
            for ax in range(3):
                pp, pm = probe.copy(), probe.copy()
                pp[ax] += eps
                pm[ax] -= eps
                grad[ax] = (
                    (tpw.doublet_potential_matrix(pp[None], quads) @ mu_flat
                     - tpw.doublet_potential_matrix(pm[None], quads) @ mu_flat)
                    / (2.0 * eps))[0]
            v = induced_velocity(probe[None], p1, p2, g, core)[0]
            self.assertLess(np.linalg.norm(grad - v), 1e-8)


class TestMotionTransfer(unittest.TestCase):
    """Body motion reaching u_rel, and the rigid fast path."""

    def test_centroid_weights_reproduce_centroids(self):
        points = small_wing(n_c=8, nspan=8, sweep_deg=20.0, camber=0.02)
        pan = tpw.build_panels(points)
        got = np.einsum("kv,kvc->kc", tpw.centroid_weights(pan["corners"]),
                        pan["corners"])
        self.assertLess(np.abs(got - pan["centroids"]).max(), 1e-13)

    def test_rigid_velocity_field_is_exact(self):
        """Translation plus rotation is linear over a panel, so the centroid
        operator transfers it without error."""
        points = small_wing(n_c=8, nspan=8, sweep_deg=20.0)
        pan = tpw.build_panels(points)
        omega = np.array([0.3, -0.7, 0.2])
        v0 = np.array([1.1, -0.4, 0.6])
        nodal = v0 + np.cross(omega, points.reshape(-1, 3)).reshape(points.shape)
        got = tpw.sample_at_centroids(pan, nodal)
        want = v0 + np.cross(omega, pan["centroids"])
        self.assertLess(np.abs(got - want).max(), 1e-12)

    def test_u_rel_subtracts_body_motion(self):
        points = small_wing()
        state = tpw.init_state(points, uniform_onset, 1.0, config={"nwake": 4})
        vel = np.zeros_like(points)
        vel[..., 2] = 0.25
        state = tpw.update_points(state, points, vel, rigid=True)
        self.assertLess(np.abs(state["u_body"][:, 2] - 0.25).max(), 1e-12)
        self.assertLess(np.abs(state["u_rel"][:, 0] - 1.0).max(), 1e-12)
        self.assertLess(np.abs(state["u_rel"][:, 2] + 0.25).max(), 1e-12)

    def test_rigid_update_keeps_valid_blocks(self):
        """The doublet and source blocks are invariant under a rigid motion, so
        the O(N^2) reassembly - the dominant cost of a time step - is skipped."""
        points = small_wing(n_c=8, nspan=8, camber=0.02)
        state = tpw.init_state(points, uniform_onset, 1.0, config={"nwake": 4})
        state = tpw.solve(state, wake="frozen")
        d0 = state["blocks"]["D"].copy()
        s0 = state["blocks"]["S"].copy()
        th, ph = 0.37, 0.21
        r_y = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)]])
        r_z = np.array([[np.cos(ph), -np.sin(ph), 0],
                        [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
        moved = (points.reshape(-1, 3) @ (r_y @ r_z).T
                 + np.array([3.1, -2.0, 0.7])).reshape(points.shape)
        state = tpw.update_points(state, moved, rigid=True)
        self.assertEqual(state["blocks_version"], state["geom_version"])
        fresh = tpw.assemble_body(state["panels"])
        self.assertLess(np.abs(fresh["D"] - d0).max(), 1e-11)
        self.assertLess(np.abs(fresh["S"] - s0).max(), 1e-12)

    def test_rigid_claim_is_checked(self):
        points = small_wing()
        state = tpw.init_state(points, uniform_onset, 1.0, config={"nwake": 4})
        state = tpw.solve(state, wake="frozen")
        stretched = points.copy()
        stretched[..., 2] *= 1.5                        # not a rigid motion
        with self.assertRaises(ValueError) as ctx:
            tpw.update_points(state, stretched, rigid=True)
        self.assertIn("deformed", str(ctx.exception))


class TestMotions(unittest.TestCase):
    """Every motion states its velocity analytically; it had better be one."""

    def _check(self, motion, times=(0.3, 1.7)):
        eps = 1e-6
        for t in times:
            _, v = motion(t)
            fd = (motion(t + eps)[0] - motion(t - eps)[0]) / (2.0 * eps)
            self.assertLess(np.abs(v - fd).max(), 1e-8)

    def test_analytic_velocities(self):
        points = small_wing()
        chord = float(tpw.section_chords(points).mean())
        te = tpw.te_nodes(points)
        pivot = te[len(te) // 2] + np.array([-0.75 * chord, 0.0, 0.0])
        self._check(usw.still(points))
        self._check(usw.translate(points, (0.7, 0.0, -0.2)))
        self._check(usw.accelerate(points, (0.3, 0.0, 0.1), (0.5, 0.0, 0.0)))
        self._check(usw.heave(points, 0.05, 2.0))
        self._check(usw.pitch(points, np.radians(3.0), 2.0, pivot))
        self._check(usw.pitch_and_heave(points, np.radians(3.0), 0.05, 2.0,
                                        pivot))

    def test_rigid_motions_preserve_the_body(self):
        """A rigid motion must keep the mesh watertight and the blocks valid,
        which update_points(rigid=True) checks and would refuse otherwise."""
        points = small_wing()
        chord = float(tpw.section_chords(points).mean())
        pivot = np.array([0.25 * chord, 0.0, 0.0])
        state = tpw.init_state(points, uniform_onset, 1.0, config={"nwake": 4})
        state = tpw.solve(state, wake="frozen")
        motion = usw.pitch(points, np.radians(6.0), 1.0, pivot)
        for t in (0.4, 1.1, 2.3):
            pts, vel = motion(t)
            state = tpw.update_points(state, pts, vel, rigid=True)
        self.assertEqual(state["blocks_version"], state["geom_version"])


class TestUnsteadyPressure(unittest.TestCase):

    def test_dphi_dt_sign_and_size(self):
        """p_gauge shifts by exactly -rho dmu/dt, the added-mass term."""
        points = small_wing()
        pan = tpw.build_panels(points)
        n = len(pan["areas"])
        rng = np.random.default_rng(2)
        mu = 0.1 * rng.standard_normal(n)
        u_rel = uniform_onset(pan["centroids"])
        rate = 0.37 * np.ones(n)
        rho = 1025.0
        p_steady = tpw.pressure_fields(pan, mu, u_rel, 1.0, rho=rho)["p_gauge"]
        p_unsteady = tpw.pressure_fields(pan, mu, u_rel, 1.0, dphi_dt=rate,
                                         rho=rho)["p_gauge"]
        self.assertLess(np.abs(p_unsteady - p_steady + rho * rate).max(), 1e-9)

    def test_backward_difference_orders(self):
        dt = 0.01
        t = np.array([-2 * dt, -dt, 0.0])
        f = [np.array([1.0 + 3.0 * s + 5.0 * s ** 2]) for s in t]
        self.assertAlmostEqual(usw.backward_difference(f, dt, 2)[0], 3.0,
                               places=10)
        self.assertAlmostEqual(usw.backward_difference(f[1:], dt, 1)[0],
                               3.0 + 5.0 * (-dt), places=10)
        self.assertEqual(usw.backward_difference(f[:1], dt, 2)[0], 0.0)


class TestSteadyLimit(unittest.TestCase):

    def test_march_recovers_the_steady_solution(self):
        """A stationary wing in a uniform onset must march back to the answer
        the steady driver gives on the same wake length."""
        points = pitch_mesh(small_wing(n_c=8, nspan=8), np.radians(5.0))
        dt = 0.25
        nsteps = 32                                     # wake reaches ~8 chords
        history, state = usw.time_march(points, uniform_onset, dt=dt,
                                        nsteps=nsteps, u_ref=1.0)
        steady = tpw.init_state(points, uniform_onset, 1.0,
                                config={"nwake": nsteps,
                                        "wake_length_spans":
                                        nsteps * dt / 8.0})
        steady = tpw.solve(steady, wake="frozen")
        cl_steady = tpw.get_loads(steady)["resultants"]["CL"]
        cl_marched = history["CL"][-1]
        # the march is still on the indicial curve at s = 8; Wagner gives 0.92
        self.assertGreater(cl_marched / cl_steady, 0.85)
        self.assertLess(cl_marched / cl_steady, 1.02)
        # monotone growth of the shed strength, no oscillation
        mu_w = history["mu_w_mid"][1:]
        self.assertTrue(np.all(np.diff(mu_w) > -1e-6))

    def test_march_is_deterministic(self):
        points = pitch_mesh(small_wing(), np.radians(5.0))
        a, _ = usw.time_march(points, uniform_onset, dt=0.3, nsteps=5)
        b, _ = usw.time_march(points, uniform_onset, dt=0.3, nsteps=5)
        self.assertTrue(np.array_equal(a["CL"], b["CL"]))


@unittest.skipUnless(SLOW, "slow analytical set: UNSTEADY_SLOW=1 to enable")
class TestAnalytical(unittest.TestCase):
    """The gates that a wrong dmu/dt or a wrong shedding rule cannot pass."""

    def test_sphere_added_mass(self):
        """A sphere accelerated in still fluid feels -m_a a with m_a = 2/3 pi
        rho a^3. No wake, no circulation: this isolates the dmu/dt term, and
        the reference is dimensional, so no normalisation can hide an error."""
        import verify_unsteady as vu
        got, exact, err = vu.case_added_mass(nwrap=20, nspan=20, verbose=False)
        self.assertLess(err, 0.01)                 # measured 0.0016

    def test_wagner_circulation(self):
        """The bound circulation is what the shedding rule sets, so it is the
        gate; the lift carries the added mass, which Wagner's circulatory
        function excludes, and runs above it (docs/UNSTEADY.md)."""
        import verify_unsteady as vu
        res = vu.case_wagner(n_c=12, nspan=12, span=20.0, nsteps=40,
                             verbose=False)
        # measured over 1 <= s <= 4: circulation 0.0284, lift 0.1100
        self.assertLess(res["rms_circulation"], 0.045)
        i2 = int(np.argmin(np.abs(res["s"] - 2.0)))
        self.assertLess(abs(res["circulation"][i2] - res["wagner_at_s2"]), 0.05)

    def test_theodorsen_plunge(self):
        """Amplitude and phase in plunge at 24 steps per cycle, where the time
        discretisation still costs about 12% and 13 degrees; both fall under
        time refinement and the gates are set on the measured values."""
        import verify_unsteady as vu
        res = vu.case_theodorsen(k=0.2, n_c=12, nspan=12, span=20.0,
                                 cycles=3, steps_per_cycle=24, verbose=False)
        self.assertLess(abs(res["amplitude_ratio"] - 1.0), 0.16)   # 1.1172
        self.assertLess(abs(res["phase_error_deg"]), 16.0)         # +13.31
        self.assertGreater(res["phase_error_deg"], 0.0)   # the sign is a lead


if __name__ == "__main__":
    unittest.main(verbosity=2)
