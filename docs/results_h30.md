### Horizon 3.0 s (10 past -> 30 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 9,415 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | ADE 95% CI | FDE (m) |
|---|---|---|---|
| Stationary | 4.006 | [3.549, 4.556] | 7.675 |
| Constant velocity | 1.248 | [1.088, 1.447] | 3.092 |
| Kalman (q/r=31.6228) | 1.165 | [1.016, 1.345] | 2.954 |
| Linear (ridge) | 1.074 | [0.959, 1.213] | 2.682 |
| LSTM | 1.309 ± 0.001 | [1.162, 1.482] | 3.177 ± 0.006 |
| Transformer | 1.306 ± 0.005 | [1.166, 1.474] | 3.139 ± 0.013 |
| Social | 1.266 ± 0.009 | [1.130, 1.426] | 3.025 ± 0.035 |
| Transformer + ridge prior | 1.106 ± 0.002 | [0.991, 1.246] | 2.758 ± 0.008 |
| Social + ridge prior | 1.085 ± 0.009 | [0.974, 1.223] | 2.684 ± 0.016 |

**Conflict detection** — 15,135 not-yet-close co-occurring pairs (3-15 m apart now), 930 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 70.7 / 71.5 / 71.1 | 46.3 / 31.2 / 37.3 |
| Kalman (q/r=31.6228) | 68.6 / 71.4 / 70.0 | 44.0 / 32.5 / 37.4 |
| Linear (ridge) | 66.7 / 67.6 / 67.2 | 43.7 / 31.9 / 36.9 |
| LSTM | 70.0 ± 0.4 / 64.2 ± 0.3 / 67.0 ± 0.3 | 51.9 ± 1.3 / 28.4 ± 0.6 / 36.7 ± 0.8 |
| Transformer | 70.2 ± 0.9 / 60.0 ± 0.4 / 64.7 ± 0.1 | 59.0 ± 2.6 / 26.0 ± 1.0 / 36.1 ± 1.4 |
| Social | 70.9 ± 0.4 / 69.0 ± 1.2 / 69.9 ± 0.8 | 53.1 ± 3.5 / 31.0 ± 1.4 / 39.1 ± 2.0 |
| Transformer + ridge prior | 66.9 ± 0.8 / 64.8 ± 0.6 / 65.9 ± 0.6 | 42.7 ± 0.6 / 29.8 ± 0.6 / 35.1 ± 0.6 |
| Social + ridge prior | 66.2 ± 0.4 / 68.7 ± 1.1 / 67.4 ± 0.6 | 41.3 ± 1.0 / 31.0 ± 1.0 / 35.4 ± 1.0 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| LSTM | 98.7 ± 0.8 | 2.00 ± 0.01 | 0.00 ± 0.00 |
| Transformer | 98.6 ± 0.4 | 2.07 ± 0.01 | 0.03 ± 0.02 |
| Social | 90.5 ± 0.7 | 2.07 ± 0.01 | 0.72 ± 0.04 |
| Transformer + ridge prior | 99.4 ± 0.1 | 2.00 ± 0.01 | 0.03 ± 0.01 |
| Social + ridge prior | 98.6 ± 0.5 | 1.98 ± 0.01 | 0.18 ± 0.05 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| Social | 1.266 ± 0.009 | 1.231 ± 0.011 | 3.025 ± 0.035 | 2.923 ± 0.012 |
| Social + ridge prior | 1.085 ± 0.009 | 1.065 ± 0.001 | 2.684 ± 0.016 | 2.655 ± 0.004 |

**Neighbor reaction** - shift of B's predicted final position (m) when A's past speed is doubled; should shrink with distance:

| Model | Mean | 95th pct | Pairs < 5 m | Pairs > 10 m |
|---|---|---|---|---|
| Social | 0.224 ± 0.017 | 0.892 ± 0.091 | 0.108 ± 0.018 | 0.314 ± 0.005 |
| Social + ridge prior | 0.103 ± 0.024 | 0.417 ± 0.110 | 0.053 ± 0.014 | 0.154 ± 0.033 |

**Risk-tier calibration** - fraction of pairs followed by a real conflict (true future gap < 3 m), per predicted tier:

| Model | SAFE | LOW | MEDIUM | HIGH | CRITICAL | F1 (MEDIUM+) | best TTC cut-off |
|---|---|---|---|---|---|---|---|
| LSTM | 0.2 ± 0.0% | 17.9 ± 0.1% | 36.3 ± 2.5% | 78.2 ± 0.9% | 100.0 ± 0.0% | 36.7 ± 0.8% | 5.2 ± 0.2 s |
| Transformer | 0.1 ± 0.0% | 18.4 ± 0.2% | 45.4 ± 4.4% | 78.6 ± 0.6% | 100.0 ± 0.0% | 36.1 ± 1.4% | 6.0 ± 0.0 s |
| Social | 0.1 ± 0.0% | 17.1 ± 0.3% | 37.3 ± 4.7% | 81.3 ± 1.5% | 100.0 ± 0.0% | 39.1 ± 2.0% | 4.8 ± 0.2 s |
| Transformer + ridge prior | 0.1 ± 0.0% | 17.7 ± 0.1% | 22.3 ± 0.3% | 76.7 ± 1.2% | 100.0 ± 0.0% | 35.1 ± 0.6% | 4.5 ± 0.0 s |
| Social + ridge prior | 0.1 ± 0.0% | 17.1 ± 0.1% | 19.4 ± 0.8% | 79.3 ± 0.9% | 100.0 ± 0.0% | 35.4 ± 1.0% | 4.2 ± 0.2 s |

**Intervention plausibility** - share of real test histories that stay under 10 m/s^2 after the intervention:

| Intervention | Before | After |
|---|---|---|
| speed = 2.0 | 84.5% | 54.6% |
| speed = 0.5 | 84.5% | 98.2% |
| brake = 0.8 | 84.5% | 89.8% |
| lateral = 3.0 | 84.5% | 84.5% |
