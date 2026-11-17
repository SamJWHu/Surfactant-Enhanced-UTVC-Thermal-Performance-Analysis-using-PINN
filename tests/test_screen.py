"""Pressure sign, Darcy units, purchase concentration and unknown-data handling."""

import unittest

from utvc.screen import COLUMNS, screen_case


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.row = dict.fromkeys(COLUMNS, "")
        self.row.update(case="PEG600_0.30", load="80W", specimen="synthetic")

    def test_empty_input_never_ranks(self):
        result = screen_case(self.row)
        self.assertIsNone(result["pressure_margin_Pa"])
        self.assertIsNone(result["additive_material_USD_unit"])

    def test_darcy_and_purchased_mass(self):
        self.row.update(
            hydraulic_source="synthetic", test_temp_C="60", net_capillary_head_Pa="3000",
            vapor_drop_Pa="120", gravity_head_Pa="0", other_drop_Pa="30",
            latent_fraction="0.5", h_fg_J_kg="2000000", mu_Pa_s="0.001",
            liquid_path_m="0.1", rho_kg_m3="1000", wick_K_m2="1e-10",
            wick_flow_area_m2="0.0001", cost_source="synthetic price", charge_mass_g="1",
            purchased_active_fraction="0.5", price_USD_kg="5",
            other_increment_USD_per_shipped_unit="0.01",
        )
        result = screen_case(self.row)
        self.assertAlmostEqual(result["circulating_kg_s"], 2e-5)
        self.assertAlmostEqual(result["liquid_drop_used_Pa"], 200.0)
        self.assertAlmostEqual(result["pressure_margin_Pa"], 2650.0)
        self.assertAlmostEqual(result["active_mass_mg"], 3.0)
        self.assertAlmostEqual(result["as_purchased_mass_mg"], 6.0)
        self.assertAlmostEqual(result["estimated_increment_USD_unit"], 0.01003)

    def test_direct_pressure_and_gravity_sign(self):
        self.row.update(
            hydraulic_source="synthetic", test_temp_C="60", net_capillary_head_Pa="3000",
            vapor_drop_Pa="120", gravity_head_Pa="-10", other_drop_Pa="30",
            liquid_drop_Pa="3000",
        )
        result = screen_case(self.row)
        self.assertEqual(result["pressure_margin_Pa"], -140.0)
        self.assertIsNone(result["circulating_kg_s"])
        self.row["hydraulic_source"] = ""
        self.assertIsNone(screen_case(self.row)["pressure_margin_Pa"])
