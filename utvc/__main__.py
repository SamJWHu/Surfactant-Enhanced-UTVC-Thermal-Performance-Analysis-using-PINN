"""Run with python -m utvc from the project root."""
import argparse
import csv
import json
import math
import os
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

from .config import Config, build_config, cases

TIMES = (1, 5, 10, 30, 60)


def _config(args, case=None):
    options = {}
    if args.design:
        options = json.loads(Path(args.design).read_text(encoding="utf-8-sig"))
        allowed = {f.name for f in fields(Config)} - {"case", "load", "fill", "source", "measured_fields"}
        if not isinstance(options, dict) or not set(options) <= allowed:
            raise ValueError("Design JSON must be a flat object of Config physical inputs in SI units.")
    return build_config(case or args.case, args.load, args.fill, args.calibration, **options)


def _times(cfg):
    return tuple(sorted({t for t in TIMES if t <= cfg.t_end} | {cfg.t_end}))


def sweep(args):
    """Deterministic baseline for all cases; never rank differences from random seeds."""
    from .reference import finite_volume
    import numpy as np
    rows, cache = [], {}
    for name in cases():
        cfg = _config(args, name)
        key = (cfg.L, cfg.W, cfg.H, cfg.k_xy, cfg.k_z, cfg.rho_cp, cfg.h_side,
               cfg.h_z, cfg.T_amb, cfg.t_end, cfg.cpu, cfg.gpu, cfg.q_cpu, cfg.q_gpu)
        if key not in cache:
            cache[key] = finite_volume(cfg, _times(cfg))
        result = cache[key]
        for stats, rise in zip(result.diagnostics, result.rise_K):
            ambient = cfg.T_amb - 273.15
            rows.append({"case":name, "wt_percent":100*cfg.c0, "fill":cfg.fill, "load": cfg.load,
                         "input_source":cfg.source or "shared water/illustrative device defaults",
                         "specified_fields":",".join(cfg.measured_fields),
                         **stats, "T_cell_peak_C":ambient + float(np.max(rise)),
                         "apparent_Rgpu_wall_to_air_K_W":(stats["FV_Tgpu_wall_mean_C"]-ambient)/cfg.power,
                         "Qout_over_Qin":(stats["Qleft_W"]+stats["Qright_W"]+stats["Qz_W"])/cfg.power})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(cases())} formulations | {len(cache)} distinct thermal input sets | {out}")


def main():
    p = argparse.ArgumentParser(description="3D transient ultra-thin vapor chamber research model")
    commands = p.add_subparsers(dest="mode", required=True)
    setup = argparse.ArgumentParser(add_help=False)
    setup.add_argument("--case", choices=cases(), default="water")
    setup.add_argument("--load", choices=("80W","150W"), default="80W")
    setup.add_argument("--fill", type=float, default=.5)
    setup.add_argument("--calibration", help="property CSV, one row per formulation")
    setup.add_argument("--design", help="JSON object of physical Config inputs, SI units")
    commands.add_parser("doctor", parents=[setup], help="check geometry, powers and installed TensorFlow")
    r = commands.add_parser("reference", parents=[setup], help="run 3D finite-volume reference using SciPy")
    r.add_argument("--out", default="runs/reference")
    s = commands.add_parser("sweep", parents=[setup], help="FV comparison of water and ten blends")
    s.add_argument("--out", default="runs/sweep.csv")
    t = commands.add_parser("train", parents=[setup], help="train the coupled fields and evaluate")
    t.add_argument("--out", default="runs/water")
    for flag, default in (("steps",250),("width",32),("depth",3),("n-int",96),("n-bc",24),
                          ("n-ic",16),("seed",42),("prefit",400),("log-every",25)):
        t.add_argument("--"+flag, type=int, default=default)
    t.add_argument("--lr", type=float, default=.001)
    t.add_argument("--anchor-weight", type=float, default=128.)
    t.add_argument("--volume-only-anchors", action="store_true", help="ablation: omit reconstructed die/fin wall labels")
    t.add_argument("--no-flow-anchors", action="store_true", help="ablation: omit coarse vapor-Darcy supervision")
    t.add_argument("--physics-only", action="store_true", help="disable all FV thermal labels")
    t.add_argument("--warm-start", help="model.weights.h5 from a matching width/depth, reinitializes optimizer")
    t.add_argument("--eager", action="store_true", help="debug automatic differentiation without graph compilation")
    t.add_argument("--no-plots", action="store_true", help="train now, evaluate the checkpoint separately")
    e = commands.add_parser("evaluate", help="evaluate saved weights with the original run parameters")
    e.add_argument("--run", required=True)
    e.add_argument("--out", help="defaults to the evaluation directory in the saved run")
    args = p.parse_args()
    if args.mode == "sweep":
        sweep(args)
        return
    if args.mode == "evaluate":
        from .train import restore
        from .evaluate import evaluate
        cfg, model, history = restore(args.run)
        evaluate(model, cfg, args.out or Path(args.run)/"evaluation", history, times=_times(cfg))
        return
    cfg = _config(args)
    if args.mode == "doctor":
        print(json.dumps(cfg.audit(), ensure_ascii=False, indent=2))
        import tensorflow as tf
        from .model import PhysicsModel
        from .config import sample_points
        model = PhysicsModel(cfg, width=12, depth=2)
        batch = sample_points(cfg,8,4,4,34)
        vals = model.losses_for({k:tf.constant(v) for k,v in batch.items()})
        for k,v in vals.items():
            tf.debugging.assert_all_finite(v, k)
        print("TensorFlow", tf.__version__, "devices", [d.device_type for d in tf.config.list_physical_devices()])
        print("Finite initial losses", {k:round(float(v),5) for k,v in vals.items()})
    elif args.mode == "reference":
        from .evaluate import reference_study
        reference_study(cfg, args.out, times=_times(cfg))
    else:
        if min(args.steps,args.width,args.depth,args.n_int,args.n_bc,args.n_ic,args.log_every) < 1:
            p.error("Step, layer, width, batch and log counts must be positive.")
        if args.prefit < 0 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.anchor_weight) or args.anchor_weight < 0:
            p.error("Require finite lr > 0, prefit >= 0 and anchor-weight >= 0.")
        from .train import train
        model, history = train(cfg, args.out, steps=args.steps, width=args.width, depth=args.depth,
                               n_int=args.n_int, n_bc=args.n_bc, n_ic=args.n_ic, seed=args.seed,
                               lr=args.lr, anchors=not args.physics_only, prefit=args.prefit,
                               anchor_weight=args.anchor_weight, wall_labels=not args.volume_only_anchors, flow_anchors=not args.no_flow_anchors, warm_start=args.warm_start,
                               compiled=not args.eager, log_every=args.log_every)
        if not args.no_plots:
            from .evaluate import evaluate
            evaluate(model, cfg, Path(args.out)/"evaluation", history, times=_times(cfg))


if __name__ == "__main__":
    main()
