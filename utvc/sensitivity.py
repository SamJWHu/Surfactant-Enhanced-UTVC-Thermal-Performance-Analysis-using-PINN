"""Change plate thickness and charge while holding unknown device closures fixed."""
import argparse
import csv
from pathlib import Path

from .config import build_config
from .reference import finite_volume


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/thickness_fill.csv")
    p.add_argument("--case", default="water")
    p.add_argument("--calibration")
    a = p.parse_args()
    rows = []
    for load in ("80W","150W"):
        for mm in (1.,1.5,2.):
            for f in (.3,.4,.5):
                cfg = build_config(a.case,load,fill=f,H=mm/1000,calibration=a.calibration)
                ref = finite_volume(cfg)
                for d in ref.diagnostics:
                    rows.append({"load":load,"case":a.case,"thickness_mm":mm,"fill_void_fraction":f,
                                 "k_xy_W_mK":cfg.k_xy,"k_z_W_mK":cfg.k_z,
                                 "vol_heat_capacity_J_m3K":cfg.rho_cp,"vapor_permeability_m2":cfg.vapor_permeability,
                                 "model":"conditional effective-plate FV",**d})
    path=Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as dest:
        w=csv.DictWriter(dest,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} snapshots (18 conditions) to {path}")


if __name__ == "__main__":
    main()
