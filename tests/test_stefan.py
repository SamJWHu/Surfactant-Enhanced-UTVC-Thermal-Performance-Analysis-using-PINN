"""Moving boundary and interfacial mass-flux regression with no training labels."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utvc.config import Config
from utvc.stefan import run


class StefanInterfaceTest(unittest.TestCase):
    def test_profiles_front_trajectory_and_mass_jump(self):
        with tempfile.TemporaryDirectory() as path, contextlib.redirect_stdout(io.StringIO()):
            stats = run(path, seed=42)
            out = np.genfromtxt(Path(path) / "moving_interface.csv", delimiter=",", names=True)
        self.assertLess(stats["front_relative_error_pct"], 2.0)
        self.assertLess(stats["vapor_profile_RMSE_K"], .035)
        self.assertLess(stats["liquid_profile_RMSE_K"], .025)
        np.testing.assert_allclose(out["time_s"], [1, 5, 10, 30, 60])
        self.assertTrue(np.all(np.diff(out["s_PINN_m"]) > 0))
        cfg = Config()
        # Stefan mass-flux magnitude on each side in the moving-front frame.
        np.testing.assert_allclose(out["J_evap_kg_m2_s"], cfg.rho_v * out["front_speed_m_s"])
        np.testing.assert_allclose(out["J_evap_kg_m2_s"],
                                   cfg.rho * (out["front_speed_m_s"] - out["liquid_velocity_m_s"]))
        np.testing.assert_allclose(out["s_PINN_m"] / (2*out["time_s"]), out["front_speed_m_s"])


if __name__ == "__main__":
    unittest.main()
