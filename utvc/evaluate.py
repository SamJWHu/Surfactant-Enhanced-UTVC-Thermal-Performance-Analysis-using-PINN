"""Field exports, held-out residuals and verification of the effective heat PDE."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .reference import finite_volume
from .darcy_reference import solve_vapor_darcy, vapor_darcy_check


TIMES = (1, 5, 10, 30, 60)


def _csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def _fieldmap(fn, X, batch=1024):
    collected = {}
    for i in range(0, len(X), batch):
        part = fn(np.asarray(X[i:i + batch], dtype=np.float32))
        for key, value in part.items():
            values = np.asarray(value).reshape(-1)
            if len(values) != min(batch, len(X) - i) or not np.isfinite(values).all():
                raise FloatingPointError(f"Invalid field diagnostic {key} at row {i}")
            collected.setdefault(key, []).append(values)
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def _with_time(points, time_s, cfg):
    return np.column_stack((points, np.full(len(points), time_s / cfg.t_end)))


def _plotslices(directory, name, labels, data, times, axes, cfg):
    """One image per field with five XY and XZ slices under a shared scale."""
    xs, ys, zs = axes
    iy, iz = np.argmin(abs(ys - .5)), np.argmin(abs(zs - .5))
    xs, ys, zs = xs * cfg.L * 1000, ys * cfg.W * 1000, zs * cfg.H * 1000
    for key, label in labels.items():
        if key not in data:
            continue
        vals = data[key]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            hi, lo = hi + max(abs(hi) * .01, 1.e-12), lo - max(abs(lo) * .01, 1.e-12)
        scale = Normalize(vmin=lo, vmax=hi)
        fig, ax = plt.subplots(2, len(times), figsize=(3.1 * len(times), 5.5),
                               constrained_layout=True, squeeze=False)
        for j, sec in enumerate(times):
            top = ax[0, j].pcolormesh(xs, ys, vals[j, :, :, iz].T, shading="nearest",
                                       cmap="viridis", norm=scale)
            ax[1, j].pcolormesh(xs, zs, vals[j, :, iy, :].T, shading="nearest",
                                cmap="viridis", norm=scale)
            ax[0, j].set_title(f"{sec:g} s")
            ax[1, j].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        ax[1, 0].set_ylabel("z (mm)")
        fig.colorbar(top, ax=ax.ravel().tolist(), label=label, shrink=.8, pad=.018)
        fig.suptitle(f"{name} | {label}\nUpper: z = {zs[iz]:.2f} mm; lower: y = {ys[iy]:.2f} mm")
        fig.savefig(directory / (key + ".png"), dpi=130)
        plt.close(fig)


def _loss_plot(outdir, history):
    if not history:
        return
    if isinstance(history, dict):
        names = list(history)
        rows = [dict(zip(names, vals)) for vals in zip(*(history[k] for k in names))]
    else:
        rows = list(history)
    _csv(outdir / "training_history.csv", rows)
    fig, (main, part) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    x = [r.get("step", r.get("epoch", j + 1)) for j, r in enumerate(rows)]
    series = {}
    for key in rows[0]:
        if key in {"step", "epoch", "seconds", "elapsed", "runtime_s", "lr",
                   "wall_s", "elapsed_s", "grad_norm"}:
            continue
        try:
            y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        except (ValueError, TypeError):
            continue
        if np.any((y > 0) & np.isfinite(y)):
            series[key] = np.where(y > 0, y, np.nan)
    overview = {"loss", "physics", "anchor_mse", "validation_physics"}
    for key in sorted(overview & set(series)):
        main.semilogy(x, series[key], "o-", label=key, lw=1.6, ms=2.5)
    components = sorted(set(series) - overview,
                        key=lambda key: np.nan_to_num(series[key][-1]), reverse=True)
    for key in components[:6]:
        part.semilogy(x, series[key], "o-", label=key, lw=1.4, ms=2)
    for ax, title in ((main, "Training and held-out physics"), (part, "Largest final component losses")):
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
    main.set_ylabel("Dimensionless loss")
    fig.savefig(outdir / "training_loss.png", dpi=140)
    plt.close(fig)


def _quad(lo, hi, order):
    x, w = np.polynomial.legendre.leggauss(order)
    return lo + (hi - lo) * (x + 1) / 2, w * (hi - lo) / 2


def _integration(bounds, order):
    grids = [_quad(lo, hi, order) for lo, hi in bounds]
    xx = np.meshgrid(*(q[0] for q in grids), indexing="ij")
    ww = np.meshgrid(*(q[1] for q in grids), indexing="ij")
    return np.stack(xx, axis=-1).reshape(-1, len(grids)), np.prod(ww, axis=0).ravel()


def _faces(cfg):
    # Each component is a linear functional on a common prediction array.
    coords, weights, names = [], [], []

    def add(axis, value, bounds, geom_scale, name, order):
        planar, ws = _integration(bounds, order)
        coords.append(np.insert(planar, axis, value, axis=1))
        weights.append(ws * geom_scale)
        names.extend([name] * len(ws))

    bounds = [(0., 1.), (0., 1.)]
    for x, name in ((0., "Qleft_W"), (1., "Qright_W")):
        add(0, x, bounds, cfg.h_side * cfg.W * cfg.H, name, 10)
    # Split at every source edge to integrate heater cut-outs and footprint means.
    xx = sorted({0., 1., *cfg.cpu[:2], *cfg.gpu[:2]})
    yy = sorted({0., 1., *cfg.cpu[2:], *cfg.gpu[2:]})
    for x0, x1 in zip(xx[:-1], xx[1:]):
        for y0, y1 in zip(yy[:-1], yy[1:]):
            xm, ym = (x0+x1)/2, (y0+y1)/2
            bounds = [(x0, x1), (y0, y1)]
            add(2, 1., bounds, cfg.h_z * cfg.L * cfg.W, "Qz_W", 5)
            for die, rect in (("cpu", cfg.cpu), ("gpu", cfg.gpu)):
                rx0, rx1, ry0, ry1 = rect
                if rx0 < xm < rx1 and ry0 < ym < ry1:
                    add(2, 0., bounds, 1 / ((rx1-rx0)*(ry1-ry0)), die, 5)
                    break
            else:
                add(2, 0., bounds, cfg.h_z * cfg.L * cfg.W, "Qz_W", 5)
    return np.concatenate(coords), np.concatenate(weights), np.array(names)


def _balance(model, cfg, t, faces, box):
    faces_xyz, ws, names = faces
    box_xyz, vol_w = box
    pred = _fieldmap(model.predict_fields, _with_time(faces_xyz, t, cfg))["T_C"]
    ambient_c = cfg.T_amb - 273.15
    rise = pred.astype(float) - ambient_c
    terms = {key: float(ws[names == key] @ rise[names == key])
             for key in ("Qleft_W", "Qright_W", "Qz_W", "cpu", "gpu")}
    qout = sum(terms[k] for k in ("Qleft_W", "Qright_W", "Qz_W"))
    # Quadrature estimates dU/dt independently of model PDE residuals. At t_end
    # use a backward difference; all other reported times use a centered one.
    delta = min(.04, cfg.t_end * .001, t / 2)
    left, right = t - delta, min(t + delta, cfg.t_end)
    temp = []
    for sample in (left, right):
        X = _with_time(box_xyz, sample, cfg)
        temp.append(_fieldmap(model.predict_fields, X)["T_C"].astype(float))
    storage = cfg.rho_cp * cfg.volume * (vol_w @ (temp[1] - temp[0])) / (right - left)
    qmiss = cfg.power - qout - storage
    return {
        "time_s": t, "Qin_W": cfg.power, "Qleft_W": terms["Qleft_W"],
        "Qright_W": terms["Qright_W"], "Qz_W": terms["Qz_W"],
        "Qout_W": qout, "Qstorage_W": storage, "balance_error_W": qmiss,
        "abs_balance_pct_input": 100 * abs(qmiss) / cfg.power,
        "out_over_input": qout / cfg.power,
        "Tcpu_wall_mean_C": ambient_c + terms["cpu"],
        "Tgpu_wall_mean_C": ambient_c + terms["gpu"],
        "Rhotter_die_mean_to_air_K_W": max(terms["cpu"], terms["gpu"]) / cfg.power,
    }


def reference_study(cfg, outdir, times=TIMES, shape=(32, 16, 5), dt=.25,
                    refinement=True):
    """Independent effective heat check; also useful without TensorFlow."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fv = finite_volume(cfg, times, shape, dt)
    fields = {"T_C": cfg.T_amb - 273.15 + fv.rise_K}
    _csv(outdir / "fv_energy.csv", fv.diagnostics)
    np.savez_compressed(outdir / "fv_temperature.npz", times_s=fv.times_s,
                        x_m=fv.centers[0]*cfg.L, y_m=fv.centers[1]*cfg.W,
                        z_m=fv.centers[2]*cfg.H, **fields)
    _plotslices(outdir, "Finite-volume effective slab", {"T_C": "Effective T (deg C)"},
                fields, fv.times_s, fv.centers, cfg)
    if refinement:
        fine = finite_volume(cfg, times, tuple(2*n for n in shape), dt / 2)
        rows = []
        for a, b in zip(fv.diagnostics, fine.diagnostics):
            row = {"time_s": a["time_s"]}
            for key in ("Qleft_W", "Qright_W", "Qz_W", "Qstorage_W",
                        "FV_Tcpu_wall_mean_C", "FV_Tgpu_wall_mean_C"):
                row["base_" + key], row["refined_" + key] = a[key], b[key]
                row["refined_minus_base_" + key] = b[key] - a[key]
            rows.append(row)
        _csv(outdir / "grid_refinement.csv", rows)
    return fv


def _boundary_audit(model, cfg, t):
    """Surface quadrature with AD and an independent five-point one-sided stencil."""
    sites, walls, rows = [], [], []
    for axis in range(3):
        for wall in (0., 1.):
            cuts_x, cuts_y = [0., 1.], [0., 1.]
            if axis == 2 and wall == 0:
                cuts_x = sorted({0., 1., *cfg.cpu[:2], *cfg.gpu[:2]})
                cuts_y = sorted({0., 1., *cfg.cpu[2:], *cfg.gpu[2:]})
            for x0, x1 in zip(cuts_x[:-1], cuts_x[1:]):
                for y0, y1 in zip(cuts_y[:-1], cuts_y[1:]):
                    xy, weight = _integration([(x0, x1), (y0, y1)], 5)
                    xyz = np.insert(xy, axis, wall, axis=1)
                    name = "xyz"[axis] + str(int(wall))
                    if name == "z0":
                        for die, rect in (("cpu", cfg.cpu), ("gpu", cfg.gpu)):
                            a, b, c, d = rect
                            if a < (x0 + x1)/2 < b and c < (y0 + y1)/2 < d:
                                name = die
                                break
                    frac_step = .003
                    direction = np.zeros(3)
                    direction[axis] = (1 - 2*wall) * frac_step
                    walls.append(xyz)
                    sites.extend(xyz + m*direction for m in range(5))
                    rows.append((len(weight), name, axis, wall, weight, frac_step))
    test_points = _with_time(np.concatenate(sites), t, cfg)
    if hasattr(model, "temperature_C"):
        import tensorflow as tf
        values = np.asarray(model.temperature_C(test_points), dtype=float).ravel()
        face_points = tf.constant(_with_time(np.concatenate(walls), t, cfg), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(face_points)
            temperatures = model.temperature_C(face_points)
        grads = tape.gradient(temperatures, face_points).numpy().astype(float)
    else:
        values = _fieldmap(model.predict_fields, test_points)["T_C"].astype(float)
        grads = None
    results, offset, face_offset = {}, 0, 0
    lengths = (cfg.L, cfg.W, cfg.H)
    conductivity = (cfg.k_xy, cfg.k_xy, cfg.k_z)
    areas = (cfg.W * cfg.H, cfg.L * cfg.H, cfg.L * cfg.W)
    for n, name, axis, wall, ws, step in rows:
        block = values[offset:offset+5*n].reshape(5, n)
        wall_t = block[0]
        offset += 5*n
        inward = np.array([-25., 48., -36., 16., -3.]) @ block / (12*step)
        flux_fd = inward * conductivity[axis] / lengths[axis]
        if grads is not None:
            g = grads[face_offset:face_offset+n, axis]
            flux = (1-2*wall) * g * conductivity[axis] / lengths[axis]
        else:
            flux = flux_fd
        face_offset += n
        if name in {"cpu", "gpu"}:
            goal = -getattr(cfg, "q_" + name)
        else:
            h = (cfg.h_side, 0., cfg.h_z)[axis]
            goal = h * (wall_t - (cfg.T_amb - 273.15))
        absolute_w = ws * areas[axis]
        cell = np.array([absolute_w.sum(), absolute_w @ (flux - goal)**2,
                         absolute_w @ flux, np.sum(absolute_w * goal), absolute_w @ flux_fd,
                         absolute_w @ (flux_fd-flux)**2])
        results[name] = results.get(name, 0.) + cell
    return [{"time_s": t, "patch": name, "area_m2": v[0],
             "flux_mismatch_RMS_W_m2": np.sqrt(v[1]/v[0]),
             "outward_conductive_W": v[2], "outward_prescribed_W": v[3],
             "FD_outward_conductive_W": v[4], "FD_vs_primary_RMS_W_m2": np.sqrt(v[5]/v[0]),
             "primary_derivative": "AD" if grads is not None else "finite difference"}
            for name, v in results.items()]


def _comparison_plot(outdir, metrics, fv):
    fig, (temp, heat) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    sec = [r["time_s"] for r in metrics]
    for component, color in (("cpu", "#087e75"), ("gpu", "#b85418")):
        temp.plot(sec, [r["T"+component+"_wall_mean_C"] for r in metrics], "o-",
                  color=color, label=component.upper()+" PINN")
        temp.plot(sec, [r["FV_T"+component+"_wall_mean_C"] for r in fv.diagnostics], "x--",
                  color=color, label=component.upper()+" FV")
    temp.set(ylabel="Area-mean wall temperature (°C)", title="Heater contact faces")
    for key, color in (("Qin_W","#202020"), ("Qout_W","#127172"),
                       ("Qstorage_W","#bd521c"), ("balance_error_W","#61458f")):
        heat.plot(sec, [r[key] for r in metrics], "o-", color=color, label=key)
    heat.axhline(0, color="gray", linestyle=":", lw=1)
    heat.set(ylabel="Rate (W)", title="PINN transient energy balance")
    for ax in (temp, heat):
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(outdir / "temperature_and_energy.png", dpi=140)
    plt.close(fig)


def evaluate(model, cfg, outdir, history=None, times=TIMES):
    """Evaluate a trained PINN; all diagnostics use a separate spatial grid.

    Assumes cfg.t_end seconds covers all output times. Saves dimensional full
    volumes, crossing planes, component residual RMS, signed boundary flux and
    transient energy audits, and finite-volume temperature differences. Returns
    one dictionary per requested time. Values for effective slab walls describe
    surface averages; CPU/GPU junctions require separate contact resistance.
    """
    times = np.asarray(times, dtype=float)
    if not len(times) or times[-1] > cfg.t_end or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Output times must be positive, increasing and inside the simulated horizon")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fv = finite_volume(cfg, times=times)
    axes = fv.centers
    shape = tuple(len(a) for a in axes)
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    face_points = _faces(cfg)
    vol_quad = _integration([(0., 1.)]*3, 11)
    fields, residualmaps, metrics, componentrows, boundaryrows = {}, [], [], [], []
    for j, sec in enumerate(times):
        X = _with_time(xyz, sec, cfg)
        values = _fieldmap(model.predict_fields, X)
        values["speed_m_s"] = np.sqrt(sum(values[k]**2 for k in ("u_m_s", "v_m_s", "w_m_s")))
        for key, val in values.items():
            fields.setdefault(key, []).append(val.reshape(shape))
        # Separate deterministic grid, distinct from optimizer batches/anchors.
        residual = _fieldmap(model.residuals, X, batch=256)
        norms = np.array([v.astype(float)**2 for v in residual.values()])
        residualmaps.append(np.sqrt(norms.mean(axis=0)).reshape(shape))
        componentrows.extend({"time_s": sec, "term": k,
                              "heldout_rms": float(np.sqrt(np.mean(v.astype(float)**2))),
                              "heldout_abs_max": float(np.max(abs(v)))}
                             for k, v in residual.items())
        row = _balance(model, cfg, sec, face_points, vol_quad)
        boundaryrows.extend(_boundary_audit(model, cfg, sec))
        temp = values["T_C"].astype(float)
        delta = temp - (cfg.T_amb - 273.15) - fv.rise_K[j].ravel()
        row.update({
            "FV_volume_RMSE_K": np.sqrt(np.mean(delta**2)),
            "FV_volume_max_abs_error_K": np.max(abs(delta)),
            "FV_CPUmean_error_K": row["Tcpu_wall_mean_C"] - fv.diagnostics[j]["FV_Tcpu_wall_mean_C"],
            "FV_GPUmean_error_K": row["Tgpu_wall_mean_C"] - fv.diagnostics[j]["FV_Tgpu_wall_mean_C"],
            "sampled_Tmin_C": temp.min(), "sampled_Tmax_C": temp.max(),
            "liquid_speed_max_m_s": values["speed_m_s"].max(),
            "gauge_pressure_min_Pa": values["p_Pa"].min(), "gauge_pressure_max_Pa": values["p_Pa"].max(),
            "min_wt_pct": values["c_wt_pct"].min(), "max_wt_pct": values["c_wt_pct"].max(),
        })
        if "phase_rate_kg_m3_s" in values:
            # Explicit diagnostics for the fixed-volume two-phase closure.
            mdot = values["phase_rate_kg_m3_s"].astype(float)
            row["evap_proxy_kg_s"] = cfg.volume * np.maximum(mdot, 0).mean()
            row["cond_proxy_kg_s"] = cfg.volume * np.maximum(-mdot, 0).mean()
            row["net_phase_proxy_kg_s"] = cfg.volume * mdot.mean()
            row["latent_evap_proxy_W"] = cfg.h_fg * row["evap_proxy_kg_s"]
            row["latent_cond_proxy_W"] = cfg.h_fg * row["cond_proxy_kg_s"]
        if "p_vapor_gauge_Pa" in values:
            vapor_velocity = np.stack([values[key].reshape(shape) for key in
                                       ("u_vapor_m_s", "v_vapor_m_s", "w_vapor_m_s")], axis=-1)
            rate = values["phase_rate_kg_m3_s"].reshape(shape)
            darcy = solve_vapor_darcy(cfg, rate)
            row.update(vapor_darcy_check(cfg, rate, values["p_vapor_gauge_Pa"].reshape(shape),
                                        vapor_velocity, reference=darcy))
            fields.setdefault("darcy_reference_vapor_mean_gauge_Pa", []).append(darcy[0])
            for axis, v_ref in zip("uvw", np.moveaxis(darcy[1], -1, 0)):
                fields.setdefault("darcy_reference_"+axis+"_vapor_m_s", []).append(v_ref)
            span = np.ptp(values["p_Pa"]) + np.ptp(values["p_vapor_gauge_Pa"])
            row["sum_pressure_spans_Pa"] = span
            row["capillary_budget_minus_spans_Pa"] = cfg.capillary_pressure - span
        row["mean_wt_pct_minus_charge"] = values["c_wt_pct"].mean() - 100*cfg.c0
        metrics.append({k: float(v) for k, v in row.items()})
    fields = {k: np.stack(v) for k, v in fields.items()}
    fields["residual_rms"] = np.stack(residualmaps)
    np.savez_compressed(outdir / "fields_3d.npz", times_s=times, x_m=axes[0]*cfg.L,
                        y_m=axes[1]*cfg.W, z_m=axes[2]*cfg.H, **fields)
    _csv(outdir / "pinn_metrics.csv", metrics)
    _csv(outdir / "boundary_audit.csv", boundaryrows)
    _csv(outdir / "heldout_residuals.csv", componentrows)
    _csv(outdir / "fv_energy.csv", fv.diagnostics)
    _plotslices(outdir, "3D PINN", {"T_C": "Effective temperature (deg C)",
                 "speed_m_s": "Intrinsic liquid speed (m/s)", "p_Pa": "Liquid gauge pressure (Pa)",
                 "c_wt_pct": "Liquid solute concentration (wt %)",
                 "residual_rms": "RMS across normalized PDE components"},
                fields, times, axes, cfg)
    _loss_plot(outdir, history)
    _comparison_plot(outdir, metrics, fv)
    return metrics
