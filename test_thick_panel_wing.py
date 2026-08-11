"""Tests for thick_panel_wing.py (standard-library unittest; no pytest needed).

Run the fast set:      python3 test_thick_panel_wing.py
Include the slow set:  THICK_SLOW=1 python3 test_thick_panel_wing.py

The fast set holds the five sign-pinning unit checks of THICK_PANEL_DESIGN.md
plus the architecture invariants of THICK_PANEL_ARCHITECTURE.md; the slow set
holds the cheap regression subset of the verification programme. No files are
written and no figures are created.
"""

import os
import unittest
import warnings

import numpy as np

import thick_panel_wing as tpw
import make_thick_sample_inputs as mts
from freewake_kernels import segment_velocities, make_onset, pitch_mesh

SLOW = os.environ.get("THICK_SLOW", "") not in ("", "0")


def sphere_wrap_mesh(nwrap=24, nspan=24, radius=1.0):
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


def uniform_onset(points):
    pts = np.atleast_2d(points)
    v = np.zeros((len(pts), 3))
    v[:, 0] = 1.0
    return v


def small_wing(n_c=6, nspan=6, **kw):
    args = dict(n_c=n_c, nspan=nspan, span=8.0, root_chord=1.0, taper=1.0,
                sweep_deg=0.0, twist_deg=0.0, camber=0.0, thickness=0.12)
    args.update(kw)
    return mts.thick_wing_mesh(**args)


class TestKernels(unittest.TestCase):
    """Unit checks 1-3: the sign authorities."""

    def test_closed_constant_mu_surface(self):
        pts = sphere_wrap_mesh(16, 16)
        pan = tpw.build_panels(pts)
        self.assertGreater(tpw.signed_volume(pan), 0.0)
        d_blk = tpw.doublet_potential_matrix(pan["centroids"], pan["corners"])
        np.fill_diagonal(d_blk, -0.5)
        rows = d_blk.sum(axis=1)
        self.assertLess(np.abs(rows + 1.0).max(), 1e-10)
        outside = np.array([[3.0, 0.4, 0.2], [0.0, 2.5, -1.0]])
        phi_out = tpw.doublet_potential_matrix(outside, pan["corners"]).sum(axis=1)
        self.assertLess(np.abs(phi_out).max(), 1e-12)
        inside = np.array([[0.2, -0.1, 0.15]])
        phi_in = tpw.doublet_potential_matrix(inside, pan["corners"]).sum(axis=1)
        self.assertLess(np.abs(phi_in + 1.0).max(), 1e-12)

    def test_source_panel_jump_and_far_field(self):
        corners = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
        centre = np.array([0.5, 0.5, 0.0])
        sigma = np.array([1.0])
        for eps in (1e-4, 1e-6):
            pair = np.array([centre + [0, 0, eps], centre - [0, 0, eps]])
            vn = tpw.source_velocity(pair, corners, sigma)[:, 2]
            self.assertAlmostEqual(vn[0], +0.5, delta=2e-3)
            self.assertAlmostEqual(vn[1], -0.5, delta=2e-3)
        far = np.array([[20.0, -13.0, 9.0]])
        r = np.linalg.norm(far[0] - centre)
        phi = tpw.source_potential_matrix(far, corners)[0, 0]
        self.assertAlmostEqual(phi, -1.0 / (4.0 * np.pi * r),
                               delta=abs(phi) * 0.01)
        near_edge = np.array([[1.2, 0.5, 0.0]])
        v = tpw.source_velocity(near_edge, corners, sigma, far_diag=50.0)[0]
        self.assertGreater(v[0], 0.0)          # outward through the +x edge

    def test_doublet_panel_ring_equivalence(self):
        corners = np.array([[[0.0, 0.0, 0.00], [1.1, 0.1, 0.02],
                             [1.0, 1.0, -0.01], [-0.1, 0.9, 0.03]]])
        order = (0, 3, 2, 1, 0)                # reversed corner order
        p1 = corners[0, order[:-1], :]
        p2 = corners[0, order[1:], :]
        eps = 1e-5
        for probe in ([0.4, 0.5, 0.8], [0.4, 0.5, -0.7], [2.5, 1.5, 0.6]):
            probe = np.array(probe)
            grad = np.zeros(3)
            for ax in range(3):
                pp, pm = probe.copy(), probe.copy()
                pp[ax] += eps
                pm[ax] -= eps
                fp = tpw.doublet_potential_matrix(pp[None], corners)[0, 0]
                fm = tpw.doublet_potential_matrix(pm[None], corners)[0, 0]
                grad[ax] = (fp - fm) / (2.0 * eps)
            ring = segment_velocities(probe[None], p1, p2,
                                      np.full(4, 1e-9)).sum(axis=1)[0]
            self.assertLess(np.linalg.norm(grad - ring), 1e-8)


class TestTECancellation(unittest.TestCase):
    """Unit check 4: the analytic Morino drop equals the explicit sum."""

    def test_te_cancellation(self):
        points = small_wing()
        pan = tpw.build_panels(points)
        n = pan["nwrap"] * pan["nspan"]
        rng = np.random.default_rng(3)
        mu = np.sin(np.linspace(0.0, 2.0, n)) + 0.1 * rng.standard_normal(n)
        onset = uniform_onset
        steps = np.full(6, 0.5)
        wake = tpw.init_wake(tpw.te_nodes(points), onset, steps,
                             n_bound=6, closure=True, core=0.05)
        tpw.set_bound_strengths(wake, mu, pan["k_up"], pan["k_low"])

        bp1, bp2, bg, bc = tpw.body_segments(pan, mu)
        wp1, wp2, wg, wc = tpw.wake_segments(wake, onset)
        probes = tpw.te_nodes(points)[1:-1] + np.array([0.3, 0.05, 0.2])
        v_a = segment_velocities(probes, np.vstack([bp1, wp1]),
                                 np.vstack([bp2, wp2]),
                                 np.concatenate([bc, wc]))
        v_a = np.einsum("msk,s->mk", v_a,
                        np.concatenate([bg, wg]))

        # explicit: add the three dropped TE segments with one shared core
        te = tpw.te_nodes(points)
        nspan = pan["nspan"]
        mu_up = mu[pan["k_up"]]
        mu_low = mu[pan["k_low"]]
        mu_w = mu_up - mu_low
        extra_p1, extra_p2, extra_g = [], [], []
        for j in range(nspan):
            a, b = te[j], te[j + 1]
            extra_p1 += [a, a, a]
            extra_p2 += [b, b, b]
            extra_g += [-mu_up[j], +mu_low[j], +mu_w[j]]
        p1 = np.vstack([bp1, wp1, np.array(extra_p1)])
        p2 = np.vstack([bp2, wp2, np.array(extra_p2)])
        g = np.concatenate([bg, wg, np.array(extra_g)])
        core = np.concatenate([bc, wc, np.full(3 * nspan, tpw.CORE_WING)])
        v_b = segment_velocities(probes, p1, p2, core)
        v_b = np.einsum("msk,s->mk", v_b, g)
        self.assertLess(np.abs(v_a - v_b).max(), 1e-11)


class TestSphere(unittest.TestCase):
    """Unit check 5 and the no-wake verification gates."""

    def test_sphere_dirichlet_solve(self):
        pts = sphere_wrap_mesh(24, 24)
        state = tpw.init_state(pts, uniform_onset, u_ref=1.0,
                               config={"lifting": False})
        state = tpw.solve(state, wake="none")
        loads = tpw.get_loads(state)
        pan = state["panels"]
        c = pan["centroids"]
        rel = c / np.linalg.norm(c, axis=1, keepdims=True)
        theta = np.arccos(np.clip(rel[:, 0], -1.0, 1.0))
        cp_exact = 1.0 - 2.25 * np.sin(theta) ** 2
        err = np.abs(loads["cp"] - cp_exact)
        nspan = pan["nspan"]
        j_of = np.arange(len(c)) % nspan
        pole = (j_of == 0) | (j_of == nspan - 1)
        # measured at 24x24: nonpole 0.031, pole 0.060, rms 0.0131 (the
        # rms corresponds to a speed error of 0.004 U, inside the design's
        # 0.01 U criterion; dCp ~ 3 dV/U at the equator)
        self.assertLess(err[~pole].max(), 0.035)
        self.assertLess(err[pole].max(), 0.07)
        self.assertLess(np.sqrt(np.mean(err ** 2)), 0.018)
        speed = np.linalg.norm(loads["v_surf"], axis=1)
        self.assertAlmostEqual(speed.max(), 1.5, delta=0.01)
        res = loads["resultants"]
        self.assertLess(abs(res["CL"]), 5e-3)
        self.assertLess(abs(res["CD_pressure"]), 5e-3)
        lk_max, lk_rms = tpw.leakage(state)
        self.assertLess(lk_rms, 0.1)           # reported diagnostic, loose gate


class TestArchitectureInvariants(unittest.TestCase):

    def test_mask_passthrough(self):
        pts = sphere_wrap_mesh(10, 10)
        pan = tpw.build_panels(pts)
        blocks = tpw.assemble_body(pan)
        n = len(pan["areas"])
        sigma = tpw.source_strengths(pan["normals"],
                                     uniform_onset(pan["centroids"]))
        a_mat, rhs = tpw.assemble_system(
            blocks["D"], blocks["S"], None, None, pan["k_up"], pan["k_low"],
            np.zeros(n, dtype=np.int8), np.zeros(n), sigma)
        self.assertTrue(np.array_equal(a_mat, blocks["D"]))
        self.assertTrue(np.array_equal(rhs, -(blocks["S"] @ sigma)))

    def test_mixed_mask(self):
        rng = np.random.default_rng(5)
        n = 8
        d_blk = np.eye(n) * -0.5 + 0.01 * rng.standard_normal((n, n))
        s_blk = 0.1 * rng.standard_normal((n, n))
        sigma = rng.standard_normal(n)
        mu_fixed = np.zeros(n)
        mask = np.zeros(n, dtype=np.int8)
        k = 3
        mask[k] = 1
        mu_fixed[k] = 0.3
        a_mat, rhs = tpw.assemble_system(d_blk, s_blk, None, None,
                                         np.array([0]), np.array([1]),
                                         mask, mu_fixed, sigma)
        self.assertTrue(np.array_equal(a_mat[:, k], s_blk[:, k]))
        keep = np.ones(n, dtype=bool)
        keep[k] = False
        expect = -(s_blk[:, keep] @ sigma[keep]) - d_blk[:, k] * 0.3
        self.assertLess(np.abs(rhs - expect).max(), 1e-14)

    def test_source_velocity_te_node(self):
        points = small_wing()
        pan = tpw.build_panels(points)
        te = tpw.te_nodes(points)
        v = tpw.source_velocity(te, pan["corners"],
                                np.ones(len(pan["areas"])))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_mesh_validation(self):
        # open TE (the standard -0.1015 coefficient) must be rejected
        points = small_wing()
        gap = points.copy()
        gap[0, :, 2] -= 2.0e-3                 # ~0.2% chord opening
        gap[-1, :, 2] += 2.0e-3
        with self.assertRaises(ValueError) as ctx:
            tpw.validate_mesh(gap)
        self.assertIn("trailing edge not closed", str(ctx.exception))
        self.assertIn("-0.1036", str(ctx.exception))
        # reversed wrap caught by the signed volume
        with self.assertRaises(ValueError) as ctx:
            tpw.validate_mesh(points[::-1].copy())
        self.assertIn("reversed", str(ctx.exception))
        # sub-tolerance TE gap welded with a warning
        tiny = points.copy()
        tiny[0, :, 2] += 1e-11
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            welded = tpw.validate_mesh(tiny)
        self.assertTrue(any("welded" in str(w.message) for w in rec))
        self.assertTrue(np.array_equal(welded[0], welded[-1]))
        # bowtie panel caught by rule 9 (crossed corners on one edge)
        bow = points.copy()
        bow[3, 2], bow[3, 3] = points[3, 3].copy(), points[3, 2].copy()
        with self.assertRaises(ValueError):
            tpw.validate_mesh(bow)

    def test_solve_determinism_and_warm_start(self):
        points = small_wing()
        state = tpw.init_state(points, uniform_onset, u_ref=1.0,
                               config={"nwake": 8})
        state = tpw.solve(state, wake="frozen")
        cl1 = tpw.get_loads(state)["resultants"]["CL"]
        mu1 = state["mu"].copy()
        state = tpw.solve(state, wake="frozen")
        self.assertTrue(np.array_equal(state["mu"], mu1))
        topo_id = id(state["panels"]["panel_nodes"])
        state = tpw.update_points(state, points)
        self.assertEqual(state["geom_version"], 1)
        self.assertEqual(id(state["panels"]["panel_nodes"]), topo_id)
        self.assertTrue(np.array_equal(state["mu"], mu1))   # warm start kept
        state = tpw.solve(state, wake="frozen")
        cl3 = tpw.get_loads(state)["resultants"]["CL"]
        self.assertAlmostEqual(cl3, cl1, delta=1e-12)

    def test_te_seam_and_tip_stencils(self):
        # dual-basis trap: linear field on a strongly swept thick wing
        points = small_wing(n_c=12, nspan=10, sweep_deg=30.0)
        pan = tpw.build_panels(points)
        g_true = np.array([2.0, 3.0, -1.0])
        f = pan["centroids"] @ g_true
        grad = tpw.surface_gradient(pan, f)
        nrm = pan["normals"]
        g_tan = g_true[None, :] - (nrm @ g_true)[:, None] * nrm
        nwrap, nspan = pan["nwrap"], pan["nspan"]
        i_of = np.arange(len(f)) // nspan
        j_of = np.arange(len(f)) % nspan
        interior = ((i_of > 1) & (i_of < nwrap - 2)
                    & (np.abs(i_of - nwrap // 2) > 1)      # away from the LE
                    & (j_of > 0) & (j_of < nspan - 1))
        err = np.linalg.norm(grad - g_tan, axis=1)[interior]
        scale = np.linalg.norm(g_true)
        # measured 0.033 (without the dual basis the skew alone gives ~0.3)
        self.assertLess(err.max() / scale, 0.045)
        # seam isolation: centroid arc-length field, discontinuous only across
        # the TE seam (jump = the whole wrap length ~ 2 chords)
        ds = pan["ds_wrap"].reshape(nwrap, nspan)
        s_wrap = np.cumsum(ds, axis=0) - 0.5 * ds     # centroid arc positions
        f2 = s_wrap.ravel()
        d1 = tpw._directional_derivative(f2, pan["centroids"],
                                         pan["wrap_prev"], pan["wrap_next"])
        seam = (i_of == 0) | (i_of == nwrap - 1)
        mid = (j_of > 0) & (j_of < nspan - 1)
        # a two-sided stencil across the seam would see the jump and produce
        # derivatives of order jump/h ~ 100; the one-sided stencil stays ~ 1
        self.assertLess(np.abs(np.abs(d1[seam & mid]) - 1.0).max(), 0.2)


@unittest.skipUnless(SLOW, "slow regression set: THICK_SLOW=1 to enable")
class TestSlowRegressions(unittest.TestCase):
    """Cheap regression subset of the verification programme."""

    def test_thin_limit_6pc(self):
        points = mts.thick_wing_mesh(n_c=24, nspan=20, span=8.0,
                                     root_chord=1.0, taper=1.0, sweep_deg=0.0,
                                     twist_deg=0.0, camber=0.0, thickness=0.06)
        points = pitch_mesh(points, np.radians(5.0))
        state = tpw.init_state(points, uniform_onset, u_ref=1.0)
        state = tpw.solve(state)
        cl = tpw.get_loads(state)["resultants"]["CL"]
        # pinned regression at 48 x 20 (converged value 0.4233 at 80 x 20;
        # against the thin code's converged 0.4114 the thickness ratio is
        # 1.029, within 0.8% of the lifting-line-damped factor 1.037)
        self.assertAlmostEqual(cl, 0.4218, delta=0.006)

    def test_loading_symmetry(self):
        points = small_wing(n_c=10, nspan=12)
        points = pitch_mesh(points, np.radians(5.0))
        state = tpw.init_state(points, uniform_onset, u_ref=1.0,
                               config={"nwake": 20})
        state = tpw.solve(state)
        loads = tpw.get_loads(state)
        _, cl_span = tpw.spanwise_loading(state["panels"], loads["p_gauge"],
                                          state["u_ref"])
        asym = np.abs(cl_span - cl_span[::-1]).max()
        self.assertLess(asym, 5e-10)          # measured 1.0e-10 at 20x12

    def test_shear_ratio(self):
        # the thick sample wing at alpha = 5 deg, as in the HANDOVER.md case
        # (0.4848 uniform against 0.4630 sheared for the thin code)
        points = pitch_mesh(mts.thick_wing_mesh(), np.radians(5.0))
        z = np.linspace(-4.0, 4.0, 17)
        u = np.maximum(1.0 + 0.3 * z, 0.1)
        shear = make_onset(z, u)
        state_u = tpw.init_state(points, uniform_onset, u_ref=1.0)
        state_u = tpw.solve(state_u)
        cl_u = tpw.get_loads(state_u)["resultants"]["CL"]
        state_s = tpw.init_state(points, shear, u_ref=1.0)
        state_s = tpw.solve(state_s)
        cl_s = tpw.get_loads(state_s)["resultants"]["CL"]
        # measured 0.9705; the thin code gives 0.9550. The offset is the
        # streamline-height Bernoulli tagging across the 12% thickness (upper
        # and lower panels sit at different z, so their onset dynamic
        # pressures differ by O(U' f)), a documented convention difference
        # from the thin code's single-surface Kutta-Joukowski loads.
        self.assertAlmostEqual(cl_s / cl_u, 0.955, delta=0.020)


if __name__ == "__main__":
    unittest.main(verbosity=2)
