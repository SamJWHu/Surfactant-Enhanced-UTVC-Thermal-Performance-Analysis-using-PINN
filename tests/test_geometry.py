"""Boundary geometry, SI heat rates, seed repeatability and input provenance."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utvc.config import build_config, cases, sample_points, water_psat


class GeometryTests(unittest.TestCase):
    def test_sources_carry_exact_applied_power_for_both_loads(self):
        for scenario, powers in (("80W", (30.0, 50.0)), ("150W", (35.0, 115.0))):
            cfg = build_config(load=scenario)
            for rect, q, watts in zip(
                (cfg.cpu, cfg.gpu), (cfg.q_cpu, cfg.q_gpu), powers
            ):
                a, b, c, d = rect
                self.assertAlmostEqual(q * (b - a) * (d - c) * cfg.A_top, watts)
            self.assertAlmostEqual((cfg.gpu[0] - cfg.cpu[1]) * cfg.L, 0.001)
            self.assertAlmostEqual(cfg.audit()["end_fin_UA_W_per_K"], 3.456)

    def test_invalid_layout_and_fill_are_rejected(self):
        for overrides in (
            {"L": 0.025, "W": 0.010},
            {"gpu_x": 0.40},
            {"H": 0.00050},
            {"fill": 1.0},
            {"mu": -0.001},
        ):
            with self.assertRaises(ValueError):
                build_config(**overrides)

    def test_formulation_mass_fraction_and_initial_pressure(self):
        self.assertEqual(len(cases()), 11)
        cfg = build_config(case="PEG600_0.30")
        self.assertAlmostEqual(cfg.c0, 0.003)
        self.assertAlmostEqual(water_psat(298.15), 3169.8, delta=0.2)
        self.assertAlmostEqual(water_psat(333.15), 19946.4, delta=1.0)
        baseline = build_config()
        for key in ("k_xy", "k_z", "interface_h_vol", "rho_cp", "mu"):
            self.assertEqual(getattr(cfg, key), getattr(baseline, key))

    def test_fill_changes_holdup_and_vapor_gap(self):
        a, b = build_config(fill=0.3), build_config(fill=0.5)
        self.assertLess(a.audit()["charge_mass_g"], b.audit()["charge_mass_g"])
        self.assertGreater(a.vapor_permeability, b.vapor_permeability)
        self.assertLess(a.rho_cp, b.rho_cp)

    def test_samples_cover_exact_disjoint_faces_at_positive_times(self):
        cfg = build_config()
        pts = sample_points(cfg, 41, 193, 19, 2026)
        repeat = sample_points(cfg, 41, 193, 19, 2026)
        for name, val in pts.items():
            np.testing.assert_array_equal(val, repeat[name])
            self.assertTrue(np.all((0 <= val) & (val <= 1)))
            self.assertTrue(
                np.all(val[:, 3] == 0 if name == "initial" else val[:, 3] > 0)
            )
        for name, axis, value in (
            ("x0", 0, 0),
            ("x1", 0, 1),
            ("y0", 1, 0),
            ("y1", 1, 1),
            ("z1", 2, 1),
            ("z0", 2, 0),
            ("cpu", 2, 0),
            ("gpu", 2, 0),
        ):
            self.assertTrue((pts[name][:, axis] == value).all())
        for name, (a, b, c, d) in (("cpu", cfg.cpu), ("gpu", cfg.gpu)):

            def inside(p):
                return (p[:, 0] > a) & (p[:, 0] < b) & (p[:, 1] > c) & (p[:, 1] < d)

            self.assertTrue(inside(pts[name]).all())
            self.assertFalse(inside(pts["z0"]).any())

    def test_calibration_is_scoped_and_requires_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "property.csv"

            def write(source):
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=["case", "source", "k_xy", "rho_cp"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "case": "PEG600_0.30",
                            "source": source,
                            "k_xy": 3500,
                            "rho_cp": 5.1e6,
                        }
                    )

            write("unit-a run 1")
            unmodified = build_config("water", calibration=path)
            calibrated = build_config("PEG600_0.30", calibration=path)
            self.assertEqual(unmodified.k_xy, 3000)
            self.assertEqual(calibrated.k_xy, 3500)
            self.assertEqual(calibrated.rho_cp, 5.1e6)
            self.assertEqual(calibrated.source, "unit-a run 1")
            self.assertEqual(set(calibrated.measured_fields), {"k_xy", "rho_cp"})
            write("")
            with self.assertRaisesRegex(ValueError, "source"):
                build_config("PEG600_0.30", calibration=path)


if __name__ == "__main__":
    unittest.main()
