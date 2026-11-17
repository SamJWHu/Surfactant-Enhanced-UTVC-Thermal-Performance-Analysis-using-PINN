# Windows 11 本機測試

## 1．開啟專案

安裝 python.org 的 **64 位元 Python 3.12**，保留 Python Launcher。下載 ZIP，先「全部解壓縮」，進入內含 `requirements.txt`、`utvc` 和 `Test-Windows.cmd` 的資料夾。可直接雙擊 `Test-Windows.cmd`，它會建立專案專用環境、安裝相依套件、執行單元測試，再從已訓練的 80 W 權重重畫五個時間點。第一次安裝 TensorFlow 需下載套件。[Python 下載](https://www.python.org/downloads/windows/) · [TensorFlow 官方安裝](https://www.tensorflow.org/install/pip)

若已開啟該資料夾的 PowerShell，可手動執行。這種寫法不需要啟動 PowerShell 腳本或修改執行原則。

```powershell
py -3.12 -m venv .venv
$p = ".\.venv\Scripts\python.exe"
& $p -m pip install -r requirements.txt
& $p -m utvc doctor
& $p -m unittest discover -s tests -v
& $p -m utvc evaluate --run examples/80W --out runs/replay80
```

`doctor` 應列出 30 與 50 W 的積分功率、CPU 裝置及有限的各項初值 loss。純水的濃度場必為零，0.30 wt% 的初始質量分率為 0.003；圖上使用 wt%。測試若在 DLL 載入時中止，依 TensorFlow 官方 Windows 說明確認已安裝 x64 Microsoft Visual C++ Redistributable。診斷時保留第一個 traceback。

## 2．看哪些結果

開啟 `runs/replay80`。在內建案例，所有熱源位於 z=0。

| 檔案 | 對照方式 |
|---|---|
| `T_C.png`, `speed_m_s.png`, `p_Pa.png`, `c_wt_pct.png` | 各圖含 1、5、10、30、60 s。看平面切片與厚度剖面。濃度與壓力的小變化亦須核對色階單位。 |
| `fields_3d.npz` | NumPy 完整三維場、座標及時間。可用 `& $p -c "import numpy as n; d=n.load('runs/replay80/fields_3d.npz'); print(d.files, d['T_C'].shape)"` 列出變數。 |
| `training_loss.png`, `heldout_residuals.csv`, `residual_rms.png` | 查哪些方程還有誤差；逐式保留的殘差比單一總 loss 更容易定位。 |
| `pinn_metrics.csv`, `fv_energy.csv` | 核對接觸面平均溫度、相對 FV 溫差、冷端排熱與暫態蓄熱。`Qout_W` 在升溫初期可小於輸入功率。 |
| `boundary_audit.csv` | CPU/GPU 的 outward 熱流符號為負；數值應分別接近 −30／−50 W 或 −35／−115 W。其餘表面的流出符號為正。 |

## 3．改幾何、功率、配方並重新訓練

```powershell
& $p -m utvc train --case PEG600_0.30 --load 80W --fill 0.50 --out runs/peg80
& $p -m utvc train --case APG_C12C13_0.15 --load 150W --fill 0.40 --design data/design_80W.json --out runs/apg150
& $p -m utvc reference --load 150W --out runs/reference150
& $p -m utvc.compare --load 80W --run examples/80W --out runs/pinn_11_cases_80W.csv
& $p -m utvc sweep --load 80W --out runs/fv_11_cases_80W.csv
```

在 `data/design_80W.json` 修改以公尺表示的 L/W/H、晶片尺寸、位置、冷端面積換算後的 `h_side`。負載由 `--load` 決定。充填率分母是腔內可用孔隙。提供自己的 `--calibration data/my_measurements.csv` 時，請從 `data/calibration_template.csv` 填上量測值與來源。若某組實測參數不同，先單獨訓練它，再將相符的 checkpoint 透過第二個 `--run` 傳給 `utvc.compare`。從 [.fit 實測反算說明](inverse_fit_zh.md)可取得實驗資料格式。

預設混合訓練先從粗網格 FV 取得溫度監督，再聯合訓練 PDE 與邊界條件。需要純 PINN 對照時執行：

```powershell
& $p -m utvc train --case PEG600_0.30 --physics-only --steps 400 --out runs/physics-only
& $p -m utvc train --case PEG600_0.40 --warm-start examples/80W/model.weights.h5 --out runs/transfer
```

轉移權重的 `--width` 和 `--depth` 必須與原模型一致。每項訓練使用一個新目錄；重新畫圖可重用已訓練權重。若想逐行檢查計算圖，用 `--eager --steps 3 --prefit 5 --n-int 12 --n-bc 6 --n-ic 4`。多個種子可用 `--seed 41`、`42`、`43` 分別執行並保留各自輸出。

## 4．NVIDIA GPU

原生 Windows 可使用本版 TensorFlow CPU。TensorFlow 自 2.11 起，Windows 的官方 NVIDIA CUDA 路徑採 WSL2。已設定 WSL2 與相容驅動程式時，在 Linux 環境建立獨立 Python 3.12 venv，安裝相同 requirements，再執行以下命令。不要共用 Windows 的 .venv。[官方 Windows／WSL2 安裝條件](https://www.tensorflow.org/install/pip)

```bash
python -m pip install 'tensorflow[and-cuda]==2.20.0'
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m utvc doctor
```

## 5．直接用 GitHub 分支測試

若已安裝 Git for Windows，可從 GitHub 取得相同版本。下載 ZIP 的使用者可直接執行前述步驟。

```powershell
git clone --branch feature/transient-utvc-pinn https://github.com/SamJWHu/Surfactant-Enhanced-UTVC-Thermal-Performance-Analysis-using-PINN.git utvc-pinn
cd utvc-pinn
.\Test-Windows.cmd
```
