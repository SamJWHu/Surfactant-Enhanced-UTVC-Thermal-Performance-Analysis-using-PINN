"""Synthetic inverse-recovery checks, independent of TensorFlow and experiments."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from utvc.config import build_config
from utvc.fit import COLUMNS, calibrate, read_runs, write_demo


class FitTest(unittest.TestCase):
    def test_recovers_two_conductivities_and_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = path / "synthetic.csv"
            shape = (16, 8, 3)
            write_demo(data, shape, .5)
            report = calibrate(data, path / "fitted", shape, .5)
            self.assertEqual(report["provenance"], "SYNTHETIC self-test")
            for key, known in (("water", 2400.), ("PEG600_0.30", 1900.)):
                case = report["cases"][key]
                self.assertTrue(case["identifiable_local"])
                self.assertTrue(case["fixture_identifiable_local"])
                self.assertLess(abs(case["k_xy"] / known - 1), .03)
                self.assertLess(abs(case["h_side"] / 36000. - 1), .03)
                self.assertLess(case["metrics"]["holdout"]["wall_RMSE_K"], .25)
                self.assertFalse(set(case["fit_runs"]).intersection(case["holdout_runs"]))
                fixture = json.loads((path / "fitted" / "fixture_design.json").read_text())
                cfg = build_config(key, calibration=path / "fitted" / "effective_k_fit.csv", **fixture)
                self.assertAlmostEqual(cfg.k_xy, case["k_xy"])
                self.assertAlmostEqual(cfg.h_side, case["h_side"])

    def test_rejects_splitting_one_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "leak.csv"
            with data.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                for i, split in enumerate(("fit", "holdout"), start=1):
                    writer.writerow(("one-device-run", split, "water", "80W", .5, 25, i, 28, 29, .3))
            with self.assertRaisesRegex(ValueError, "split"):
                read_runs(data)
