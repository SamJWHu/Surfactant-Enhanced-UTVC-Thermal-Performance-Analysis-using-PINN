"""Inverse calibration of the reduced thermal operator from measured wall temperatures.

Usage: python -m utvc.fit --demo --out runs/fit-demo
       python -m utvc.fit --data measurements.csv --out runs/fit
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .config import Config, FORMULATIONS, LOADS, build_config
from .reference import finite_volume


COLUMNS = ("run_id", "split", "case", "load", "fill", "ambient_C", "time_s",
           "cpu_wall_mean_C", "gpu_wall_mean_C", "sigma_T_K")
# Outer numerical search limits in W/(m K) and W/(m² K). Bound hits are reported.
SEARCH_BOUNDS = {"k_xy": (10.0, 100_000.0), "h_side": (50.0, 1_000_000.0)}
FIXED = {"case", "load", "fill", "T_amb", "k_xy", "h_side", "measured_fields", "source"}


@dataclass(frozen=True)
class Run:
    run_id: str
    split: str
    cfg: Config
    times: np.ndarray
    target: np.ndarray
    sigma: np.ndarray


def read_runs(path, design=None, calibration=None):
    """Each run is one uninterrupted step from equilibrium, on a common fixture."""
    design = design or {}
    if not isinstance(design, dict) or set(design) - ({f.name for f in fields(Config)} - FIXED):
        raise ValueError("Invalid fixed-design keys; k_xy and fixture h_side are inferred.")
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or (len(reader.fieldnames) != len(COLUMNS)
                                          or set(reader.fieldnames) != set(COLUMNS)):
            raise ValueError("Measurement columns must match: " + ",".join(COLUMNS))
        for row in reader:
            if None in row or any(not (row[key] or "").strip() for key in COLUMNS):
                raise ValueError(f"Missing/extra value at CSV line {reader.line_num}.")
            groups[row["run_id"].strip()].append(row)
    runs = []
    for run_id, rows in groups.items():
        meta = {(r["case"].strip(), r["load"].strip(), r["split"].strip(),
                 float(r["fill"]), float(r["ambient_C"])) for r in rows}
        if len(meta) != 1:
            raise ValueError(f"Run {run_id}: case/load/split/fill/ambient must be constant.")
        case, load, split, fill, ambient = meta.pop()
        if case not in FORMULATIONS or load not in LOADS or split not in {"fit", "holdout"}:
            raise ValueError(f"Run {run_id}: unknown case, load or split.")
        values = np.asarray([[float(r[key]) for key in COLUMNS[6:]] for r in rows])
        values = values[np.argsort(values[:, 0])]
        times = values[:, 0]
        if (not np.all(np.isfinite(values)) or np.any(times <= 0)
                or np.any(np.diff(times) <= 0) or np.any(values[:, 3] <= 0)):
            raise ValueError(f"Run {run_id}: require finite readings, unique positive times and positive sigma.")
        cfg = build_config(case, load, fill, calibration, T_amb=ambient + 273.15, **design)
        if "k_xy" in cfg.measured_fields:
            raise ValueError("Exclude k_xy from the fixed-property input; it is estimated here.")
        runs.append(Run(run_id, split, cfg, times, values[:, 1:3], values[:, 3:4]))
    if "water" not in {r.cfg.case for r in runs}:
        raise ValueError("Include a pure-water control for the common fixture h_side.")
    for case in {r.cfg.case for r in runs}:
        subset = [r for r in runs if r.cfg.case == case]
        if {r.split for r in subset} != {"fit", "holdout"}:
            raise ValueError(f"{case}: supply at least one fit run and one separate holdout run.")
        if len({r.cfg.fill for r in subset}) != 1:
            raise ValueError(f"{case}: one effective conductivity is fitted per fill; separate the charges.")
    return runs


def wall_prediction(run, pars, shape, dt):
    fv = finite_volume(replace(run.cfg, **pars), run.times, shape=shape, dt=dt)
    return np.array([[r["FV_Tcpu_wall_mean_C"], r["FV_Tgpu_wall_mean_C"]] for r in fv.diagnostics])


def infer_parameters(runs, shape, dt, fixture_h=None):
    """Fit only training runs. SVD diagnostics use sigma-weighted log-parameter sensitivities."""
    keys = ["k_xy", "h_side"] if fixture_h is None else ["k_xy"]
    known = {} if fixture_h is None else {"h_side": fixture_h}

    def physical(v):
        return {**known, **dict(zip(keys, np.exp(v)))}

    def residual(v):
        pars = physical(v)
        return np.concatenate([((wall_prediction(r, pars, shape, dt) - r.target) / r.sigma).ravel()
                               for r in runs])

    lo, hi = np.log(np.array([SEARCH_BOUNDS[k] for k in keys]).T)
    x0 = np.log([getattr(runs[0].cfg, k) for k in keys])
    fit = least_squares(residual, np.clip(x0, lo+1e-4, hi-1e-4), bounds=(lo, hi),
                        diff_step=2e-4, max_nfev=50)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = np.linalg.matrix_rank(fit.jac)
    condition = float(singular[0] / singular[-1]) if rank == len(keys) else None
    hits = [k for k, v, l, h in zip(keys, fit.x, lo, hi) if min(v-l, h-v) < .001]
    # A unit perturbation in a log parameter is approximately a relative change.
    weakest_log_se = 1 / float(singular[-1]) if rank == len(keys) else None
    identifiable = bool(fit.success and not hits and condition is not None and condition < 1e3
                        and weakest_log_se < 1.)
    return physical(fit.x), {
        "optimizer_success": bool(fit.success), "message": fit.message, "fit_parameter_names": keys,
        "weighted_log_jacobian_singular_values": singular.tolist(),
        "log_jacobian_condition": condition, "weakest_local_log_parameter_std": weakest_log_se,
        "near_search_bounds": hits, "identifiable_local": identifiable,
        "fit_runs": [r.run_id for r in runs], "nfev": fit.nfev,
    }


def calibrate(path, out, shape=(24, 12, 4), dt=.25, design=None, calibration=None, synthetic=False):
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = read_runs(path, design, calibration)
    cases = sorted({r.cfg.case for r in runs}, key=lambda case: (case != "water", case))
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if all(r.run_id.startswith("SYNTHETIC-") for r in runs):
        synthetic = True
    label = "SYNTHETIC self-test" if synthetic else "inverse fit to supplied wall-temperature CSV"
    fixed_rows = {}
    if calibration:
        with Path(calibration).open(encoding="utf-8-sig", newline="") as f:
            fixed_rows = {row["case"].strip(): row for row in csv.DictReader(f)}
    report = {"provenance": label, "input_sha256": fingerprint,
              "shape": list(shape), "dt_s": dt, "fixed_design": design or {},
              "fixed_properties_sha256": hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
              if calibration else None, "cases": {}}
    predictions, estimated = [], []
    fixture_h = None
    for case in cases:
        subset = [r for r in runs if r.cfg.case == case]
        train = [r for r in subset if r.split == "fit"]
        pars, diagnostics = infer_parameters(train, shape, dt, fixture_h)
        if case == "water":
            fixture_h = pars["h_side"]
        residuals = defaultdict(list)
        for r in subset:
            pred = wall_prediction(r, pars, shape, dt)
            error = pred - r.target
            residuals[r.split].append(error)
            for i, t in enumerate(r.times):
                predictions.append({**{k: getattr(r, k) for k in ("run_id", "split")},
                                    "case": case, "load": r.cfg.load, "time_s": float(t),
                                    "cpu_pred_C": pred[i, 0], "gpu_pred_C": pred[i, 1],
                                    "cpu_obs_C": r.target[i, 0], "gpu_obs_C": r.target[i, 1],
                                    "sigma_T_K": r.sigma[i, 0]})
        diagnostics["metrics"] = {}
        for split, chunks in residuals.items():
            error = np.concatenate(chunks, axis=0)
            diagnostics["metrics"][split] = {
                "wall_RMSE_K": float(np.sqrt(np.mean(error**2))),
                "CPU_bias_K": float(np.mean(error[:, 0])), "GPU_bias_K": float(np.mean(error[:, 1])),
                "max_abs_error_K": float(np.max(np.abs(error))), "n_wall_values": error.size}
        diagnostics.update(pars)
        diagnostics["holdout_runs"] = [r.run_id for r in subset if r.split == "holdout"]
        diagnostics["fill"] = subset[0].cfg.fill
        diagnostics["fixture_identifiable_local"] = (diagnostics["identifiable_local"] if case == "water"
                                                    else report["cases"]["water"]["identifiable_local"])
        report["cases"][case] = diagnostics
        row = dict(fixed_rows.get(case, {}))
        notes = [label, f"CSV SHA256 {fingerprint[:12]}", f"fill {subset[0].cfg.fill:g}"]
        if row.get("source"):
            notes.append("fixed inputs: " + row["source"])
        row.update({"case": case, "source": "; ".join(notes), "k_xy": pars["k_xy"]})
        estimated.append(row)
    def write_csv(name, rows):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(rows)
    write_csv("wall_predictions.csv", predictions)
    # Fits and diagnostics are always available for review, including weakly identifiable fits.
    write_csv("effective_k_fit.csv", estimated)
    (out / "fixture_design.json").write_text(
        json.dumps({**(design or {}), "h_side": fixture_h}, indent=2) + "\n", encoding="utf-8")
    (out / "fit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def write_demo(path, shape, dt):
    """Same-operator recovery test with noise. No experimental measurements are used."""
    rng = np.random.default_rng(114)
    times = np.array([1., 5., 15., 35., 60.])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        for case, k in (("water", 2400.), ("PEG600_0.30", 1900.)):
            for split, load, ambient in (("fit", "80W", 23.), ("holdout", "150W", 25.)):
                cfg = build_config(case, load, .5, T_amb=273.15+ambient, k_xy=k, h_side=36000.)
                ref = finite_volume(cfg, times, shape, dt)
                for t, d in zip(times, ref.diagnostics):
                    a, b = rng.normal(0., .08, size=2)
                    writer.writerow(dict(zip(COLUMNS, (f"SYNTHETIC-{case}-{load}", split, case, load, .5,
                                         ambient, t, d["FV_Tcpu_wall_mean_C"] + a,
                                         d["FV_Tgpu_wall_mean_C"] + b, .08))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path)
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=(24, 12, 4), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--dt", type=float, default=.25, help="Maximum FV time step, seconds")
    parser.add_argument("--design", type=Path, help="JSON of fixed geometry/thermal inputs in SI")
    parser.add_argument("--fixed-properties", type=Path, help="Optional config calibration CSV omitting k_xy")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        if args.design or args.fixed_properties:
            parser.error("--demo generates its own fixed physical inputs")
        args.data = args.out / "synthetic_measurements.csv"
        write_demo(args.data, args.shape, args.dt)
    design = json.loads(args.design.read_text(encoding="utf-8-sig")) if args.design else {}
    report = calibrate(args.data, args.out, args.shape, args.dt, design, args.fixed_properties, args.demo)
    print(report["provenance"])
    for case, fit in report["cases"].items():
        print(f"{case}: k_xy={fit['k_xy']:.5g} W/(m K), h_side={fit['h_side']:.5g} W/(m² K), "
              f"holdout wall RMSE={fit['metrics']['holdout']['wall_RMSE_K']:.3g} K, "
              f"locally identifiable={fit['identifiable_local']}")


if __name__ == "__main__":
    main()
