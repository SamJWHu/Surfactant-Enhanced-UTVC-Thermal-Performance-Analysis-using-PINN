"""Manufactured discrete cosine mode for a sealed, depth-resolved vapor core."""
import unittest

import numpy as np

from utvc.config import build_config
from utvc.darcy_reference import solve_vapor_darcy, vapor_darcy_check


class SealedVaporTest(unittest.TestCase):
    def test_pressure_gauge_neumann_and_intrinsic_velocity(self):
        cfg = build_config()
        shape = (12, 10, 6)
        ds = np.array([cfg.L, cfg.W, cfg.H]) / shape
        masses = cfg.rho_v * (1.-cfg.fill) * cfg.void_fraction
        mobility = cfg.vapor_permeability / cfg.mu_v
        exact_p, rhs = np.zeros(shape), np.zeros(shape)
        exact_v = np.zeros((*shape, 3))
        for axis, (n, dx, amp) in enumerate(zip(shape, ds, (0.65, -0.25, 0.31))):
            expand = [1, 1, 1]
            expand[axis] = n
            phase = (np.pi * (np.arange(n) + .5)/n).reshape(expand)
            p_mode = amp * np.cos(phase)
            exact_p += p_mode
            rhs += p_mode * (2 * np.sin(np.pi/(2*n)) / dx)**2
            exact_v[..., axis] = mobility * amp * np.sin(np.pi/n) * np.sin(phase) / dx
        imposed_mean = 0.32
        source = rhs * mobility * masses + imposed_mean
        pressure, vel, removed_mean = solve_vapor_darcy(cfg, source)
        np.testing.assert_allclose(pressure, exact_p, atol=1e-10)
        np.testing.assert_allclose(vel, exact_v, atol=1e-11)
        self.assertAlmostEqual(removed_mean, imposed_mean, places=8)
        audit = vapor_darcy_check(cfg, source, exact_p + 729., exact_v)
        self.assertLess(audit['darcy_pressure_gauge_aligned_RMSE_Pa'], 1.e-10)
        self.assertLess(audit['darcy_velocity_vector_RMSE_m_s'], 1.e-10)


if __name__ == '__main__':
    unittest.main()
