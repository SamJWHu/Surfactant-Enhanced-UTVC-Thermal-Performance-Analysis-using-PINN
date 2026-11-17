"""Regression checks for the 3D derivative path using analytic solutions.

Run from the project directory: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import tensorflow as tf

from utvc.config import build_config, sample_points
from utvc.model import PhysicsModel, energy_residual, scalar_jet


class DifferentialOperatorTests(unittest.TestCase):
    def setUp(self):
        self.points = tf.constant([[0.13, 0.23, 0.38, 0.09],
                                   [0.62, 0.77, 0.42, 0.54]], tf.float32)
        self.cfg = build_config()

    def test_gradient_and_diagonal_hessian_include_thickness_and_time(self):
        def manufactured(points):
            x, y, z, s = tf.split(points, 4, axis=1)
            return x**2 + 3*x*y + 2*y**2 + 4*z**3 + s*z

        _, first, diagonal = scalar_jet(manufactured, self.points)
        x, y, z, s = self.points.numpy().T
        exact_first = np.stack([2*x + 3*y, 3*x + 4*y, 12*z**2 + s, z], axis=1)
        exact_diagonal = np.stack([2 + 0*x, 4 + 0*y, 24*z], axis=1)
        np.testing.assert_allclose(first, exact_first, rtol=1e-5)
        np.testing.assert_allclose(diagonal, exact_diagonal, rtol=1e-5)

    def test_anisotropic_transient_heat_equation(self):
        # theta = sum(a_i * coordinate_i^2) + 2*tau*sum(a_i*Fo_i)
        # solves d(theta)/d(tau) = sum Fo_i*d2(theta)/d(coordinate_i)^2.
        cfg = self.cfg
        fourier = np.array([cfg.k_xy/cfg.L**2, cfg.k_xy/cfg.W**2, cfg.k_z/cfg.H**2]) * cfg.t_end/cfg.rho_cp
        # Equal O(1) contributions make a missing axis readily detectable.
        coeff = tf.constant(1.0/fourier, tf.float32)

        def exact(points):
            return tf.reduce_sum(coeff*points[:, :3]**2, axis=1, keepdims=True) + 6.0*points[:, 3:4]

        _, first, diagonal = scalar_jet(exact, self.points)
        exact_residual = energy_residual(first, diagonal, cfg).numpy()
        # The negative control has no temporal derivative and must fail the PDE.
        no_time = tf.concat([first[:, :3], tf.zeros_like(first[:, 3:4])], axis=1)
        negative_control = energy_residual(no_time, diagonal, cfg).numpy()
        self.assertTrue(np.all(np.abs(negative_control) > 0))
        np.testing.assert_allclose(exact_residual / np.abs(negative_control), 0.0, atol=2e-6)


class ModelExecutionTests(unittest.TestCase):
    def test_evaporation_and_condensation_have_balanced_mass_flux(self):
        cfg = build_config(case='PEG600_0.30')
        model = PhysicsModel(cfg, width=8, depth=2)
        nodes, weights = [], []
        for order in (40, 20, 3):
            n, w = np.polynomial.legendre.leggauss(order)
            nodes.append((n + 1.) / 2.)
            weights.append(w / 2.)
        space = np.stack(np.meshgrid(*nodes, indexing='ij'), axis=-1).reshape(-1, 3)
        wt = np.prod(np.meshgrid(*weights, indexing='ij'), axis=0).ravel()
        points = np.column_stack((space, np.ones(len(space)) * 0.5))
        f = model.fields(points)
        rate = f['phase_rate_kg_m3_s'].numpy().ravel()
        self.assertGreater(rate.max(), 0.)
        self.assertLess(rate.min(), 0.)
        self.assertLess(abs(wt @ rate), 1e-5 * (wt @ abs(rate)))
        points[:, 3] = 0.
        initial = model.fields(points[:8])
        np.testing.assert_allclose(initial['c_wt_pct'], 0.30, rtol=1e-5)
        for name in ('T_C', 'p_Pa', 'u_m_s', 'v_m_s', 'w_m_s', 'phase_rate_kg_m3_s'):
            target = cfg.T_amb - 273.15 if name == 'T_C' else 0.
            np.testing.assert_allclose(initial[name], target, atol=1e-5)

    def test_heater_flux_and_outward_robin_signs(self):
        cfg = build_config()
        batch = sample_points(cfg, n_int=8, n_bc=6, n_ic=3, seed=81)
        # Linear theta in z imposes the physical CPU heat flux on its underside.
        # At both end faces use a quadratic such that theta=1 there and
        # -n k dT/dx = h_side * dT in the second manufactured field.
        class ManufacturedWall(PhysicsModel):
            def _theta(self, pts, raw):
                x, z = pts[:, 0:1], pts[:, 2:3]
                if self.linear_z:
                    return -(cfg.q_cpu * cfg.H / (cfg.k_z * cfg.dT)) * z
                return 1. + cfg.h_side * cfg.L / cfg.k_xy * x * (1. - x)

        model = ManufacturedWall(cfg, width=6, depth=1)
        model.linear_z = True
        r = model.boundary_residuals(batch)
        np.testing.assert_allclose(r['heat_cpu'], 0., atol=2e-6)
        np.testing.assert_allclose(r['heat_gpu'], 1. - cfg.q_cpu/cfg.q_gpu, atol=2e-6)
        model.linear_z = False
        r = model.boundary_residuals(batch)
        for end in ('heat_x0', 'heat_x1'):
            np.testing.assert_allclose(r[end], 0., atol=2e-6)

    def test_global_storage_integrates_exact_heater_powers(self):
        cfg = build_config(load='150W')
        # At the power step a spatially uniform ambient field rejects no heat.
        # A known temporal slope stores exactly the imposed CPU+GPU power.
        class InitialRamp(PhysicsModel):
            def _theta(self, pts, raw):
                capacity = cfg.rho_cp * cfg.volume
                return self.store_fraction * cfg.power * cfg.t_end / (capacity * cfg.dT) * pts[:, 3:4]

        model = InitialRamp(cfg, width=5, depth=1)
        points = tf.constant([[.31, .65, .19, 0.], [.72, .24, .54, 0.]], tf.float32)
        model.store_fraction = 1.
        _, global_balance = model.control_volume_energy(points)
        np.testing.assert_allclose(global_balance, 0., atol=1e-5)
        model.store_fraction = 0.
        _, missing_storage = model.control_volume_energy(points)
        np.testing.assert_allclose(missing_storage, -1., atol=1e-5)

    def test_hard_fluxes_on_actual_network(self):
        cfg = build_config()
        model = PhysicsModel(cfg, width=8, depth=2)
        batch = sample_points(cfg, n_int=8, n_bc=97, n_ic=3, seed=23)
        residuals = model.boundary_residuals(batch)
        for face in ('cpu', 'gpu', 'z0', 'z1', 'y0', 'y1'):
            with self.subTest(face=face):
                np.testing.assert_allclose(residuals['heat_' + face], 0.0, atol=3e-5)

    def test_initial_temperature_and_traced_optimizer_update(self):
        tf.keras.utils.set_random_seed(8901)
        cfg = build_config()
        model = PhysicsModel(cfg, width=8, depth=2)
        batch = {name: tf.convert_to_tensor(points, tf.float32) for name, points in
                 sample_points(cfg, n_int=8, n_bc=4, n_ic=3, seed=8).items()}
        initial_T = model.temperature_C(batch['initial']).numpy()
        np.testing.assert_allclose(initial_T, cfg.T_amb - 273.15, atol=1e-4)
        opt = tf.keras.optimizers.Adam(2e-4)

        @tf.function
        def update(points):
            with tf.GradientTape() as tape:
                terms = model.losses_for(points)
                total = tf.add_n(list(terms.values()))
            grads = tape.gradient(total, model.trainable_variables)
            if not grads or any(g is None for g in grads):
                raise AssertionError('Disconnected loss/weight in traced training')
            tf.debugging.assert_all_finite(total, 'Non-finite optimizer input')
            for g in grads:
                tf.debugging.assert_all_finite(g, 'Non-finite weight gradient')
            norm = tf.linalg.global_norm(grads)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return norm

        self.assertGreater(float(update(batch)), 0.)
        self.assertEqual(int(opt.iterations), 1)
        fields = model.fields(batch['interior'])
        for name, value in fields.items():
            with self.subTest(field=name):
                self.assertEqual(value.shape[0], 8)
                self.assertTrue(np.isfinite(value.numpy()).all())


if __name__ == '__main__':
    unittest.main()
