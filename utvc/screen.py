"""Screen measured capillary budgets and purchased additive cost for UTVC specimens.

Usage: python -m utvc.screen data/screen_template.csv results/screen.csv
Missing observations are reported; no thermal predictions or rankings are inferred.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .config import Config, FORMULATIONS, LOADS

IDENTITY = ("case", "load", "specimen")
PROVENANCE = ("hydraulic_source", "cost_source")
NUMERIC = (
    "test_temp_C", "net_capillary_head_Pa", "vapor_drop_Pa", "gravity_head_Pa",
    "other_drop_Pa", "liquid_drop_Pa", "latent_fraction", "h_fg_J_kg", "mu_Pa_s",
    "liquid_path_m", "rho_kg_m3", "wick_K_m2", "wick_flow_area_m2",
    "charge_mass_g", "purchased_active_fraction", "price_USD_kg",
    "other_increment_USD_per_shipped_unit",
)
COLUMNS = (*IDENTITY, *PROVENANCE, *NUMERIC)
NONNEGATIVE = {"vapor_drop_Pa", "other_drop_Pa", "liquid_drop_Pa", "price_USD_kg"}
SIGNED = {"test_temp_C", "net_capillary_head_Pa", "gravity_head_Pa",
          "other_increment_USD_per_shipped_unit"}
FRACTIONS = {"latent_fraction", "purchased_active_fraction"}
DARCY_INPUTS = ("latent_fraction", "h_fg_J_kg", "mu_Pa_s", "liquid_path_m",
                "rho_kg_m3", "wick_K_m2", "wick_flow_area_m2")


def numeric_values(row):
    values = {}
    for name in NUMERIC:
        raw = row[name].strip()
        if not raw:
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name == "purchased_active_fraction" and not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0,1]")
        if name == "latent_fraction" and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
        if name in NONNEGATIVE and value < 0:
            raise ValueError(f"{name} must be >= 0")
        if name not in NONNEGATIVE | SIGNED | FRACTIONS and value <= 0:
            raise ValueError(f"{name} must be > 0")
        if name == "test_temp_C" and value <= -273.15:
            raise ValueError("test_temp_C must exceed absolute zero")
        values[name] = value
    return values


def screen_case(row):
    case, load, specimen = [row[key].strip() for key in IDENTITY]
    if case not in FORMULATIONS or load not in LOADS or not specimen:
        raise ValueError("unknown case/load or blank specimen ID")
    config = Config(case=case, load=load)
    n = numeric_values(row)
    hydraulic_source, cost_source = [row[key].strip() for key in PROVENANCE]
    values = {
        "case": case, "load": load, "specimen": specimen,
        "test_temp_C": n.get("test_temp_C"), "hydraulic_source": hydraulic_source,
        "cost_source": cost_source, "nominal_q_cpu_W_m2": config.q_cpu,
        "nominal_q_gpu_W_m2": config.q_gpu, "circulating_kg_s": None,
        "liquid_drop_method": "", "liquid_drop_used_Pa": None,
        "net_capillary_head_Pa": n.get("net_capillary_head_Pa"),
        "vapor_drop_Pa": n.get("vapor_drop_Pa"), "gravity_head_Pa": n.get("gravity_head_Pa"),
        "other_drop_Pa": n.get("other_drop_Pa"), "pressure_margin_Pa": None,
        "pressure_status": "", "missing_pressure_inputs": "",
        "active_mass_mg": None, "as_purchased_mass_mg": None,
        "additive_material_USD_unit": None, "estimated_increment_USD_unit": None,
        "missing_cost_inputs": "",
    }
    pressure_missing = [key for key in ("test_temp_C", "net_capillary_head_Pa",
                                        "vapor_drop_Pa", "gravity_head_Pa", "other_drop_Pa")
                        if key not in n]
    if not hydraulic_source:
        pressure_missing.append("hydraulic_source")
    if "latent_fraction" in n and "h_fg_J_kg" in n and hydraulic_source:
        values["circulating_kg_s"] = config.power*n["latent_fraction"]/n["h_fg_J_kg"]
    if "liquid_drop_Pa" in n:
        values["liquid_drop_method"] = "direct"
        values["liquid_drop_used_Pa"] = n["liquid_drop_Pa"]
    else:
        missing_darcy = [key for key in DARCY_INPUTS if key not in n]
        if missing_darcy:
            pressure_missing.append("liquid_drop_Pa OR " + "+".join(missing_darcy))
        elif hydraulic_source:
            values["liquid_drop_method"] = "Darcy effective circuit"
            values["liquid_drop_used_Pa"] = (
                n["mu_Pa_s"] * n["liquid_path_m"] * values["circulating_kg_s"]
                / (n["rho_kg_m3"] * n["wick_K_m2"] * n["wick_flow_area_m2"])
            )
    values["missing_pressure_inputs"] = " | ".join(pressure_missing)
    if pressure_missing:
        values["pressure_status"] = "insufficient data"
    else:
        margin = (n["net_capillary_head_Pa"] - values["liquid_drop_used_Pa"]
                  - n["vapor_drop_Pa"] - n["gravity_head_Pa"] - n["other_drop_Pa"])
        values["pressure_margin_Pa"] = margin
        values["pressure_status"] = "positive headroom" if margin > 0 else "at or beyond budget"

    if case == "water":
        values.update(active_mass_mg=0.0, as_purchased_mass_mg=0.0,
                      additive_material_USD_unit=0.0, estimated_increment_USD_unit=0.0)
        return values

    cost_inputs = ("charge_mass_g", "purchased_active_fraction", "price_USD_kg")
    cost_missing = [key for key in cost_inputs if key not in n]
    if not cost_source:
        cost_missing.append("cost_source")
    if "charge_mass_g" in n:
        active_mg = 1000 * n["charge_mass_g"] * config.c0
        values["active_mass_mg"] = active_mg
        if "purchased_active_fraction" in n:
            values["as_purchased_mass_mg"] = active_mg/n["purchased_active_fraction"]
    if not cost_missing:
        material_cost = values["as_purchased_mass_mg"] * 1e-6 * n["price_USD_kg"]
        values["additive_material_USD_unit"] = material_cost
        if "other_increment_USD_per_shipped_unit" in n:
            values["estimated_increment_USD_unit"] = (
                material_cost + n["other_increment_USD_per_shipped_unit"]
            )
    if "other_increment_USD_per_shipped_unit" not in n:
        cost_missing.append("other_increment_USD_per_shipped_unit")
    values["missing_cost_inputs"] = " | ".join(cost_missing)
    return values


def run(input_path, output_path):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("choose an output path different from the measurement input")
    seen = set()
    results = []
    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS)
                or len(reader.fieldnames) != len(COLUMNS)):
            raise ValueError("input header must match data/screen_template.csv")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"row {number}: unexpected or missing CSV fields")
            try:
                result = screen_case(row)
            except ValueError as exc:
                raise ValueError(f"row {number}: {exc}") from exc
            key = tuple(result[field] for field in IDENTITY)
            if key in seen:
                raise ValueError(f"row {number}: duplicate specimen and condition {key}")
            seen.add(key)
            results.append(result)
    if not results:
        raise ValueError("input contains no conditions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    completed = sum(row["pressure_margin_Pa"] is not None for row in results)
    print(f"{len(results)} conditions, {completed} completed pressure budgets: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
