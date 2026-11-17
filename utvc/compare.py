"""Compare PINN checkpoints on physically matching conditions.

With shared water proxies, the macroscopic PDEs and normalized tracer PDE are
identical for all ten blends. Reuse that same solution and restore each case's
mass fraction. Changes to physical inputs require a matching trained model.
"""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import build_config, cases
from .train import restore


def compatible(actual, target):
    # Even uncalibrated data can be compared as scenarios. Matching is physical;
    # nominal case labels, measurement IDs and passive tracer amplitudes differ.
    skip = {"case", "source", "measured_fields"}
    a, b = asdict(actual), asdict(target)
    return all(a[k] == b[k] for k in a if k not in skip) and (actual.c0 > 0 or target.c0 == 0)


def compare(runs, out, load="80W", fill=.5, calibration=None, design=None):
    from .__main__ import _times
    from .evaluate import _balance, _faces, _integration
    cfgs = [build_config(case, load, fill, calibration, **(design or {})) for case in cases()]
    models = [(*restore(p)[:2],Path(p).name) for p in runs]
    chosen = []
    for cfg in cfgs:
        matches = [i for i,(c,_,_) in enumerate(models) if compatible(c, cfg)]
        if not matches:
            raise ValueError(f"No physically matching checkpoint for {cfg.case}. "
                             "Train that case at this design/load/fill and add --run PATH. "
                             "Solute cases require a solute-trained checkpoint.")
        # Prefer the very same representative when conditions are equivalent.
        chosen.append(matches[0])
    axis = [(np.arange(n)+.5)/n for n in (24,14,5)]
    xyz = np.stack(np.meshgrid(*axis,indexing="ij"),axis=-1).reshape(-1,3)
    cache, rows = {}, []
    for cfg,i in zip(cfgs,chosen):
        old, model, label = models[i]
        if i not in cache:
            faces, box = _faces(old), _integration([(0., 1.)]*3, 11)
            stats = []
            for sec in _times(cfg):
                p = np.column_stack((xyz,np.full(len(xyz),sec/cfg.t_end))).astype("float32")
                vals = {k:np.asarray(v) for k,v in model.fields(p).items()}
                speed = np.sqrt(sum(vals[n]**2 for n in ("u_m_s","v_m_s","w_m_s")))
                b = _balance(model,old,sec,faces,box)
                rel = vals["c_wt_pct"]/(100*old.c0) if old.c0 else np.zeros_like(speed)
                stats.append((b, float(vals["T_C"].max()), float(speed.max()),
                              float(vals["p_Pa"].max()-vals["p_Pa"].min()),
                              float(rel.min()),float(rel.max())))
            cache[i] = stats
        for b, peak, speed, pressure, cmin, cmax in cache[i]:
            rows.append(dict(case=cfg.case, wt_pct=100*cfg.c0, fill=cfg.fill, load=cfg.load,
                             checkpoint=label, calibration=cfg.source or "common illustrative closures",
                             **b, peak_sampled_T_C=peak, peak_liquid_speed_m_s=speed,
                             liquid_gauge_span_Pa=pressure, min_c_wt_pct=cmin*100*cfg.c0,
                             max_c_wt_pct=cmax*100*cfg.c0))
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(cfgs)} formulations, {len(cache)} distinct solved conditions, saved {dest}")


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append",required=True,help="repeat for distinct physical conditions")
    p.add_argument("--out",default="runs/pinn_comparison.csv")
    p.add_argument("--calibration")
    p.add_argument("--design",help="flat physical configuration JSON, SI")
    p.add_argument("--load",choices=["80W","150W"],default="80W")
    p.add_argument("--fill",type=float,default=.5)
    a=p.parse_args()
    design=json.loads(Path(a.design).read_text("utf-8-sig")) if a.design else None
    compare(a.run,a.out,a.load,a.fill,a.calibration,design)


if __name__=="__main__":
    main()
