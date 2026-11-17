# 如何檢查蒸氣速度與壓力

`evaluate` 會在每個輸出時間把 PINN 的蒸發／凝結質量源 `j` 代入另一個求解器。這可檢查神經網路能否在選定的質量源下求出流場。質量源是否符合真實腔體則需要量測或兩相 CFD 檢驗。

## 固定來源的蒸氣子問題

取蒸氣占整塊板體積的比例 `alpha_v = void_fraction × (1 − fill)`。蒸氣連續及達西式給出

$$
\nabla\!\cdot\mathbf u_v=\frac{j}{\rho_v\alpha_v},\qquad
\mathbf u_v=-\frac{K_v}{\mu_v}\nabla p_v,\qquad
-\nabla^2 p_v=\frac{\mu_v}{K_v\rho_v\alpha_v}j.
$$

六面皆無法向穿透。有限體積網格的離散餘弦變換直接解出此常係數 Poisson 方程。封閉問題只決定壓差，程式扣除各自網格平均後比較壓力。由於求值在有限個單元中心，空間平均質量源可能有微小誤差。求解器將此平均項扣除，使離散源與六面零通量相容，並獨立列出扣除量。解析餘弦製造解測試涵蓋三個方向、壓力任意常數與由面通量內插到中心的速度。

| `pinn_metrics.csv` 欄位 | 讀法 |
|---|---|
| `darcy_reference_pressure_span_Pa`、`darcy_pinn_pressure_span_Pa` | 比較整個蒸氣區最高與最低壓的差值（Pa）。兩者皆為流動引起的壓差。 |
| `darcy_pressure_gauge_aligned_RMSE_Pa` | 扣除兩者各自的平均壓力後，全域壓力差的均方根。 |
| `darcy_velocity_vector_RMSE_m_s`、`darcy_reference_speed_RMS_m_s` | 速度向量誤差的均方根除以基準速度均方根，可評估相對誤差。參考流量接近零時，請看絕對數值。 |
| `darcy_removed_net_source_kg_s` | 扣除的淨質量源，單位 kg/s。應比蒸發及凝結循環量小，亦可加密網格核對。 |

`fields_3d.npz` 另外提供 `darcy_reference_vapor_mean_gauge_Pa` 與 `darcy_reference_u_vapor_m_s`、`darcy_reference_v_vapor_m_s`、`darcy_reference_w_vapor_m_s`，可直接取出同一來源下的離散壓力與速度場。

速度及壓力誤差大時，可從 `heldout_residuals.csv` 分別檢查 `continuity_vapor` 及三個 `darcy_vapor_*`，搭配蒸發區與兩端的取樣密度判讀。這一組比較驗證常密度、固定相體積分率的蒸氣達西子問題。液相含 Brinkman 黏性、慣性與非揮發溶質傳輸，其速度、壓力與濃度仍需要額外參考。

## 使用蒸氣場訓練資料

預設混合訓練使用一個 **16 × 10 × 4** 的粗網格。程式對單位化的蒸發／凝結空間分布求一次蒸氣 Darcy 解。由於固定物性下方程線性、相變來源可分成固定空間形狀乘上 `mdot(t)`，每個時間點的目標壓力與速度可用該流率縮放。目標壓力在幾何中心歸零。訓練時切斷目標流率的梯度，避免網路為了降低流場誤差而直接調整目標。單獨推論只執行神經模型。`evaluate` 在獨立的 **32 × 16 × 5** 網格重解，兩種網格共用相變假設。

以相同案例、亂數種子、步數與初值執行兩次 `train`，第二次加上 `--no-flow-anchors`，可量化這項輔助資料對蒸氣流場的影響。`--physics-only` 會同時關閉熱與蒸氣兩種粗網格資料。

## 80 W、PEG600 0.30 wt% 的實跑比較

三次訓練皆使用 seed 62、兩層各 24 個神經元、每次 48 個內點／每區 12 個邊界點／8 個初始點、400 次聯立更新。使用熱資料的兩組各先執行 200 次溫度預訓練。訓練物性為共同的示範參數。下表以各模型自己預測的相變來源，在獨立 **32 × 16 × 5** 網格重解，顯示速度向量 RMSE 除以基準速率 RMS。

| 時間 | 純物理訓練 | 加熱場資料 | 加熱場及蒸氣資料 |
|---|---:|---:|---:|
| 1 s | 106.0% | 106.5% | 105.8% |
| 5 s | 64.4% | 69.3% | 36.0% |
| 10 s | 59.7% | 66.5% | 35.5% |
| 30 s | 108.3% | 115.6% | 72.6% |
| 60 s | 94.5% | 101.0% | 50.3% |

60 秒的對齊壓力 RMSE 依序為 0.395、0.365、0.184 Pa。早期流場仍有很大的誤差，這組設定下應優先採用條件相同的獨立達西解評估蒸氣壓降。三次訓練只比較同一亂數種子，無法外推配方優劣。逐時間的 SI 誤差存於 `examples/vapor_flow_ablation.csv`。

重跑基線使用 `python -m utvc train --case PEG600_0.30 --load 80W --steps 400 --width 24 --depth 2 --n-int 48 --n-bc 12 --n-ic 8 --prefit 200 --seed 62 --log-every 100 --out runs/both`。另兩次分別加上 `--no-flow-anchors` 與 `--physics-only`，並各自指定新的 `--out` 資料夾。
