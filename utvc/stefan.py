"""Two-domain PINN for planar vapor-front growth at a heated wall.

A separate water benchmark: semi-infinite, subcooled liquid, prescribed absolute
pressure, local equilibrium, zero vapor velocity, low-Mach liquid displacement.
Learns two independent temperature profiles and a free-boundary coefficient.
No closed-chamber geometry, wick, curvature, nucleation or surfactant here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import erf, erfc, erfcx
import tensorflow as tf

from .config import Config, water_psat


class FilmPINN(tf.keras.Model):
    """The two profile corrections are smooth, boundary-exact functions."""
    def __init__(self):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(16, activation="tanh") for _ in range(2)]
        self.head = tf.keras.layers.Dense(2, kernel_initializer="zeros")
        self.log_lambda = self.add_weight(name="log_lambda", shape=(),
            initializer=tf.keras.initializers.Constant(math.log(.03)))

    def call(self, xi):
        features = 2*xi - 1.
        for hidden in self.hidden:
            features = hidden(features)
        vapor, liquid = tf.split(self.head(features), 2, axis=1)
        # The liquid far boundary at 5 diffusion lengths approximates infinity.
        theta_v = 1-xi + xi*(1-xi)*vapor
        theta_l = (1-xi)*tf.exp(-6*xi) + xi*(1-xi)*tf.exp(-2*xi)*liquid
        return theta_v, theta_l


def run(out, superheat=10., subcool=1., pressure_kPa=19.946, steps=220, polish=120,
        seed=42, warm_start=None):
    if min(superheat, subcool, pressure_kPa) <= 0 or steps < 0 or polish < 0:
        raise ValueError("Require positive thermal differences/pressure and nonnegative training counts.")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose an empty benchmark output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)
    cfg = Config()
    sat_K = brentq(lambda temp: water_psat(temp) - 1000*pressure_kPa, 273.16, 647.09)
    rho_l, rho_v, latent = cfg.rho, cfg.rho_v, cfg.h_fg
    kv, kl = .021, cfg.k_liq  # constant illustrative water properties near 60 °C
    av, al = kv/(rho_v*cfg.cp_v), kl/(rho_l*cfg.cp)
    ratio = (rho_v/rho_l)*math.sqrt(av/al)
    domain = 5.
    qscale = kl*subcool/math.sqrt(al)
    model = FilmPINN()
    model(tf.zeros((1, 1)))
    if warm_start:
        model.load_weights(warm_start)
    # Clenshaw-Curtis abscissas resolve the depleted liquid boundary layer.
    coll = (.5-.5*np.cos(np.linspace(0, np.pi, 65))).astype("float32")[:, None]
    x = tf.constant(coll)
    opt = tf.keras.optimizers.Adam(.008)
    start = perf_counter()

    def terms():
        lam = tf.exp(model.log_lambda)
        eta = ratio*lam
        with tf.GradientTape(persistent=True) as g2:
            g2.watch(x)
            with tf.GradientTape(persistent=True) as g1:
                g1.watch(x)
                v, l = model(x)
            vg = g1.gradient(v, x)
            lg = g1.gradient(l, x)
            del g1
        vgg, lgg = g2.gradient(vg, x), g2.gradient(lg, x)
        del g2
        rv = vgg + 2*lam**2*x*vg
        rl = lgg/domain**2 + (2/domain)*(eta+domain*x)*lg
        qv = -kv*superheat*vg[-1,0]/(2*lam*math.sqrt(av))
        ql = -kl*subcool*lg[0,0]/(2*domain*math.sqrt(al))
        jump = (qv-ql-rho_v*latent*lam*math.sqrt(av)) / qscale
        return tf.reduce_mean(rv**2) + tf.reduce_mean(rl**2) + jump**2

    @tf.function
    def value_grad():
        with tf.GradientTape() as g:
            value = terms()
        return value, g.gradient(value, model.trainable_variables)

    @tf.function
    def step():
        value, grad = value_grad()
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return value

    history = []
    for i in range(steps):
        loss = step()
        if i == 0 or (i+1) % 20 == 0 or i+1 == steps:
            history.append([i+1, float(loss)])
    if polish:
        weights = model.trainable_variables
        sizes = [int(np.prod(w.shape)) for w in weights]
        def load_vector(flat):
            for w, part in zip(weights, np.split(flat, np.cumsum(sizes)[:-1])):
                w.assign(part.reshape(w.shape))

        def objective(flat):
            load_vector(flat)
            value, grad = value_grad()
            return float(value), np.concatenate([v.numpy().ravel() for v in grad]).astype(float)

        initial = np.concatenate([w.numpy().ravel() for w in weights]).astype(float)
        res = minimize(objective, initial, method="L-BFGS-B", jac=True,
                       options={"maxiter": polish, "ftol": 1e-12, "gtol": 1e-8, "maxls": 30})
        load_vector(res.x)
        history.append([steps + res.nit, float(res.fun)])
    final_loss = float(terms())
    seconds = perf_counter()-start
    model.save_weights(out / "film.weights.h5")
    # The exact semi-infinite Neumann solution enters only AFTER optimization.
    def stefan_root(lam):
        qv = kv*superheat*np.exp(-lam**2)/(math.sqrt(np.pi*av)*erf(lam))
        ql = kl*subcool/(math.sqrt(np.pi*al)*erfcx(ratio*lam))
        return qv - ql - rho_v*latent*lam*math.sqrt(av)
    exact = brentq(stefan_root, 1e-10, 12.)
    learned = float(tf.exp(model.log_lambda))
    xx = np.linspace(0, 1, 251, dtype="float32")[:,None]
    v, l = [a.numpy().ravel() for a in model(xx)]
    vv = 1-erf(exact*xx.ravel())/erf(exact)
    ll = erfc(ratio*exact + domain*xx.ravel())/erfc(ratio*exact)
    sat_C = sat_K-273.15
    summary = {
        "pressure_kPa_absolute": pressure_kPa, "Tsat_C": sat_C, "superheat_K": superheat,
        "subcool_K": subcool, "lambda_pinn": learned, "lambda_exact": exact,
        "front_relative_error_pct": 100*abs(learned-exact)/exact,
        "vapor_profile_RMSE_K": float(superheat*np.sqrt(np.mean((v-vv)**2))),
        "liquid_profile_RMSE_K": float(subcool*np.sqrt(np.mean((l-ll)**2))),
        "dimensionless_loss": final_loss, "train_s_including_trace": seconds,
        "seed": seed, "Adam_steps": steps, "LBFGS_iterations": int(res.nit) if polish else 0,
        "warm_start": str(Path(warm_start).name) if warm_start else None,
    }
    (out/"comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out/"loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "dimensionless_loss"])
        writer.writerows(history)
    times = np.array([1., 5., 10., 30., 60.])
    front = 2*learned*np.sqrt(av*times)
    front_exact = 2*exact*np.sqrt(av*times)
    speed = learned*np.sqrt(av/times)
    table = np.column_stack([times, front, front_exact, speed, rho_v*speed,
                             (1-rho_v/rho_l)*speed])
    np.savetxt(out/"moving_interface.csv", table, delimiter=",", comments="",
               header="time_s,s_PINN_m,s_exact_m,front_speed_m_s,J_evap_kg_m2_s,liquid_velocity_m_s")
    fig, axes = plt.subplots(1,2,figsize=(9.5,3.8), constrained_layout=True)
    axes[0].plot(times, 1e3*front_exact, color="#444444", label="Analytical")
    axes[0].scatter(times, 1e3*front, marker="x", color="#147e75",label="PINN")
    axes[0].set(xlabel="Time (s)", ylabel="Front position (mm)",title="Planar vapor film")
    t = 5.
    for lam, vprof, lprof, style, lbl in ((exact,vv,ll,"--","Analytical"),(learned,v,l,"-","PINN")):
        interface = 2*lam*math.sqrt(av*t)
        pos = 1e3*np.r_[interface*xx.ravel(), interface + 2*math.sqrt(al*t)*domain*xx.ravel()]
        temp = np.r_[sat_C+superheat*vprof, sat_C-subcool+subcool*lprof]
        axes[1].plot(pos, temp, style, label=lbl)
    axes[1].set(xlabel="Distance from hot wall (mm)", ylabel="Temperature (°C)", title="Both phases at 5 s")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.savefig(out/"moving_interface.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True)
    cli.add_argument("--superheat", type=float, default=10.)
    cli.add_argument("--subcool", type=float, default=1.)
    cli.add_argument("--pressure-kPa", type=float, default=19.946, dest="pressure_kPa")
    cli.add_argument("--steps", type=int, default=220)
    cli.add_argument("--polish", type=int, default=120)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--warm-start")
    run(**vars(cli.parse_args()))
