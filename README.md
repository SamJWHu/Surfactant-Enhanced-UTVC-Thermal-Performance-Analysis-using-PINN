# Surfactant-enhanced ultra-thin vapor chamber | 3D transient PINN

This TensorFlow 2.20 research implementation predicts the effective temperature field of a rectangular ultra-thin vapor chamber under two spatially separate CPU/GPU heat loads. A reduced liquid/vapor circulation and passive-solute model accompanies the thermal solve. The version includes two real CPU training runs, five checkpoints in **physical seconds**, and an independently discretized thermal reference. The three original Python scripts remain available at the root for historical reproduction.

**Windows 11:** extract the project and run `Test-Windows.cmd` or follow the [繁體中文操作說明](docs/windows_zh.md). [Verified numerical results](docs/results_zh.md) · [Equation/units guide](docs/model_zh.md) · [Diagnosis of the original scripts](docs/legacy_audit_zh.md) · [Experimental evidence, measured capillary screening and OEM development plan](docs/research_zh.md).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m utvc doctor
.\.venv\Scripts\python.exe -m utvc evaluate --run examples/80W --out runs/replay
.\.venv\Scripts\python.exe -m utvc train --case PEG600_0.30 --load 80W --out runs/peg80
```

| Command/module | Output |
|---|---|
| `python -m utvc reference` | 3D transient finite-volume temperature and heat rejection under the same physical inputs |
| `python -m utvc train` / `evaluate` | TensorFlow weights, PDE and boundary residuals, training curves, area-average heater temperatures, volume fields, and surface power audit |
| `python -m utvc.compare --run examples/80W` | Eleven-case PINN comparison using a checkpoint matching the specified physical coefficients |
| `python -m utvc.interface --csv runs/interface.csv` | Liquid/vapor mass, momentum, Stefan, shear and adsorption jump checks on a manufactured moving interface |
| `python -m utvc.sensitivity` | Two loads × three thicknesses × three filling ratios, with the effective coefficients held fixed |
| `python -m utvc sweep` | Independent finite-volume comparison of eleven formulations |
| `python -m utvc.fit --demo --out runs/fit-demo` | Thermal inverse-fit pipeline with a held-out hardware run; demo measurements are synthetic. [Data schema](docs/inverse_fit_zh.md) |
| `python -m utvc.screen data/screen_template.csv runs/screen.csv` | Measured capillary pressure budget and incremental additive cost, with source and missing-value tracking |
| `python -m utvc.stefan --out runs/stefan` | Train a planar two-phase moving-front benchmark with analytical reference |
| `python -m unittest discover -s tests -v` | Geometry, automatic differentiation, manufactured solution, heat boundary, interface and regression tests |

The two delivered loads use common placeholder device properties and water-like liquid properties for all ten additives. Differences in mixture performance require experimental information. The reported chip-contact wall temperatures, gauge pressures, liquid velocities, effective plate temperature and 1D phase-interface benchmark each describe a different physical observable. The validation table and measurement instructions distinguish these quantities.

For the 80 W example, run `python -m utvc evaluate --run examples/80W --out runs/replay80` to regenerate the full 3D field archive and training-loss figure from the published weights. Both are included in the downloadable project package.
