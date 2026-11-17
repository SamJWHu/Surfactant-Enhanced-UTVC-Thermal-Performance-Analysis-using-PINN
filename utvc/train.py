"""Small, resampled space-time batches and explicit diagnostic checkpoints."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from .config import sample_points
from .model import PhysicsModel


def _tensors(batch):
    return {name: tf.convert_to_tensor(a, tf.float32) for name, a in batch.items()}


def _write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _weights(model, directory):
    tmp = directory / "saving.weights.h5"
    model.save_weights(tmp)
    os.replace(tmp, directory / "model.weights.h5")


def train(cfg, outdir, *, steps=250, width=32, depth=3, n_int=96, n_bc=24,
          n_ic=16, seed=42, lr=0.001, anchors=True, prefit=400,
          anchor_weight=128., wall_labels=True, flow_anchors=True, warm_start=None, compiled=True, log_every=25):
    """Train one geometry/load/fluid. Training artifacts are case-specific.

    Hybrid mode uses a distinct coarse-grid thermal solution as supervision.
    Physics-only mode uses neither its labels nor its prefit. The model is
    reevaluated on a fixed unseen collocation set at every logging interval.
    """
    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose a new output directory. {out} contains an existing run.")
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = PhysicsModel(cfg, width=width, depth=depth)
    model(tf.zeros((1, 4)))
    if warm_start:
        model.load_weights(warm_start)
    teacher = None
    if anchors and flow_anchors:
        from .flow_anchors import VaporAnchors
        teacher = VaporAnchors(cfg, model)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt.build(model.trainable_variables)
    meta = {"config": asdict(cfg), "seed": seed, "steps": steps, "width": width,
            "depth": depth, "learning_rate": lr, "tensorflow": tf.__version__,
            "n_interior": n_int, "n_per_boundary": n_bc, "n_initial": n_ic,
            "thermal_anchor_source": "coarse finite volume" if anchors else "none",
            "anchor_weight": anchor_weight if anchors else 0.,
            "reconstructed_wall_labels": bool(anchors and wall_labels),
            "vapor_anchor_source": "coarse Neumann-Darcy" if teacher is not None else "none",
            "prefit_steps": prefit if anchors else 0, "compiled": compiled,
            "warm_start": str(Path(warm_start).name) if warm_start else None,
            "devices": [v.device_type for v in tf.config.list_physical_devices()]}
    _write_json(out / "run.json", meta)
    if anchors:
        from .reference import reference_anchors
        X, Y = reference_anchors(cfg, seed=seed + 14, count=6144, wall_labels=wall_labels)
    else:
        X = np.zeros((1, 4), np.float32)
        Y = np.zeros((1, 1), np.float32)

    def anchor_batch(n):
        ids = rng.integers(len(X), size=n)
        return tf.convert_to_tensor(X[ids]), tf.convert_to_tensor(Y[ids])

    def fit_temperature(x, y):
        with tf.GradientTape() as tape:
            error = (model.temperature_C(x) - y) / cfg.dT
            loss = tf.reduce_mean(tf.square(error))
        gs = tape.gradient(loss, model.trainable_variables)
        pairs = [(g, v) for g, v in zip(gs, model.trainable_variables) if g is not None]
        opt.apply_gradients(pairs)
        return loss

    def joint(points, x, y):
        with tf.GradientTape() as tape:
            terms = model.losses_for(points)
            physics = tf.add_n(list(terms.values()))
            total = physics
            if anchors:
                fit = tf.reduce_mean(tf.square((model.temperature_C(x) - y) / cfg.dT))
                total = total + anchor_weight * fit
            else:
                fit = tf.zeros_like(physics)
            if teacher is not None:
                vp, vv = teacher.losses(model, points["interior"])
                total = total + .5*(vp+vv)
                auxiliary = {"vapor_anchor_pressure":vp, "vapor_anchor_velocity":vv}
            else:
                auxiliary = {}
            total = tf.debugging.check_numerics(total, "Non-finite training objective")
        grads = tape.gradient(total, model.trainable_variables)
        if any(g is None for g in grads):
            raise RuntimeError("Disconnected trainable variable in the full PINN objective")
        grads, norm = tf.clip_by_global_norm(grads, 5.)
        tf.debugging.check_numerics(norm, "Non-finite gradient norm")
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return {"loss": total, "physics": physics, "anchor_mse": fit, "grad_norm": norm, **terms, **auxiliary}

    if compiled:
        fit_temperature = tf.function(fit_temperature, reduce_retracing=True)
        joint = tf.function(joint, reduce_retracing=True)
    if anchors and prefit:
        for _ in range(prefit):
            fit_temperature(*anchor_batch(192))
        print(f"Thermal prefit: {prefit} steps ({perf_counter()-start:.1f} s)", flush=True)
    validation = _tensors(sample_points(cfg, n_int*2, n_bc*2, n_ic*2, seed + 500000))
    history = []
    for step in range(1, steps+1):
        batch = _tensors(sample_points(cfg, n_int, n_bc, n_ic, seed + step))
        terms = joint(batch, *anchor_batch(128 if anchors else 1))
        if step == 1 or step % log_every == 0 or step == steps:
            values = {name: float(value.numpy()) for name, value in terms.items()}
            check = model.losses_for(validation)
            row = {"step": step, "seconds": perf_counter() - start, **values,
                   "validation_physics": sum(float(v.numpy()) for v in check.values())}
            history.append(row)
            with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            _weights(model, out)
            print(f"step {step}/{steps}  loss {row['loss']:.5g}  "
                  f"held-out physics {row['validation_physics']:.5g}  {row['seconds']:.1f} s", flush=True)
    meta["seconds_including_prefit_and_logging"] = perf_counter() - start
    meta["completed_steps"] = steps
    _write_json(out / "run.json", meta)
    return model, history


def restore(run):
    from .config import Config
    run = Path(run)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cfg = Config(**meta["config"])
    model = PhysicsModel(cfg, width=meta["width"], depth=meta["depth"])
    model(tf.zeros((1, 4)))
    model.load_weights(run / "model.weights.h5")
    with (run / "history.csv").open(encoding="utf-8") as f:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return cfg, model, history
