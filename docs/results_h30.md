### Horizon 3.0 s (10 past -> 30 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 9,415 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | ADE 95% CI | FDE (m) |
|---|---|---|---|
| Stationary | 4.006 | [3.549, 4.556] | 7.675 |
| Constant velocity | 1.248 | [1.088, 1.447] | 3.092 |
| Kalman (q/r=31.6228) | 1.165 | [1.016, 1.345] | 2.954 |
| Linear (ridge) | 1.074 | [0.959, 1.213] | 2.682 |
| Transformer | 1.306 ± 0.005 | [1.166, 1.474] | 3.141 ± 0.013 |
| Social | 1.266 ± 0.010 | [1.131, 1.426] | 3.025 ± 0.038 |
| Transformer + ridge prior | 1.107 ± 0.009 | [0.992, 1.247] | 2.762 ± 0.025 |
| Social + ridge prior | 1.087 ± 0.011 | [0.975, 1.225] | 2.688 ± 0.020 |

**Conflict detection** — 15,135 not-yet-close co-occurring pairs (3-15 m apart now), 930 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 70.7 / 71.5 / 71.1 | 46.3 / 31.2 / 37.3 |
| Kalman (q/r=31.6228) | 68.6 / 71.4 / 70.0 | 44.0 / 32.5 / 37.4 |
| Linear (ridge) | 66.7 / 67.6 / 67.2 | 43.7 / 31.9 / 36.9 |
| Transformer | 70.2 ± 0.8 / 59.9 ± 0.4 / 64.6 ± 0.2 | 58.9 ± 2.3 / 26.0 ± 1.0 / 36.1 ± 1.3 |
| Social | 70.9 ± 0.4 / 69.0 ± 1.1 / 69.9 ± 0.8 | 52.9 ± 3.6 / 31.0 ± 1.3 / 39.1 ± 2.0 |
| Transformer + ridge prior | 66.5 ± 0.4 / 64.8 ± 0.3 / 65.7 ± 0.4 | 42.5 ± 0.4 / 28.9 ± 1.0 / 34.4 ± 0.9 |
| Social + ridge prior | 66.1 ± 0.6 / 68.6 ± 1.3 / 67.3 ± 0.8 | 41.4 ± 1.0 / 31.0 ± 1.0 / 35.4 ± 1.0 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| Transformer | 98.6 ± 0.4 | 2.07 ± 0.01 | 0.03 ± 0.02 |
| Social | 90.4 ± 0.8 | 2.06 ± 0.01 | 0.72 ± 0.04 |
| Transformer + ridge prior | 99.3 ± 0.3 | 2.01 ± 0.02 | 0.01 ± 0.01 |
| Social + ridge prior | 98.6 ± 0.5 | 1.98 ± 0.01 | 0.18 ± 0.05 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| Social | 1.266 ± 0.010 | 1.231 ± 0.011 | 3.025 ± 0.038 | 2.922 ± 0.012 |
| Social + ridge prior | 1.087 ± 0.011 | 1.066 ± 0.001 | 2.688 ± 0.020 | 2.656 ± 0.002 |

**Neighbor reaction** - shift of B's predicted final position (m) when A's past speed is doubled; should shrink with distance:

| Model | Mean | 95th pct | Pairs < 5 m | Pairs > 10 m |
|---|---|---|---|---|
| Social | 0.225 ± 0.017 | 0.877 ± 0.106 | 0.109 ± 0.017 | 0.315 ± 0.005 |
| Social + ridge prior | 0.105 ± 0.024 | 0.423 ± 0.109 | 0.054 ± 0.014 | 0.156 ± 0.033 |

**Risk-tier calibration** - fraction of pairs followed by a real conflict (true future gap < 3 m), per predicted tier:

| Model | SAFE | LOW | MEDIUM | HIGH | CRITICAL | F1 (MEDIUM+) | best TTC cut-off |
|---|---|---|---|---|---|---|---|
| Transformer | 0.1 ± 0.0% | 18.4 ± 0.2% | 45.2 ± 4.3% | 78.4 ± 0.5% | 100.0 ± 0.0% | 36.1 ± 1.3% | 6.0 ± 0.4 s |
| Social | 0.1 ± 0.0% | 17.1 ± 0.3% | 37.3 ± 4.7% | 81.1 ± 1.6% | 100.0 ± 0.0% | 39.1 ± 2.0% | 4.8 ± 0.2 s |
| Transformer + ridge prior | 0.1 ± 0.0% | 17.9 ± 0.2% | 22.0 ± 0.2% | 76.1 ± 1.4% | 100.0 ± 0.0% | 34.4 ± 0.9% | 4.5 ± 0.0 s |
| Social + ridge prior | 0.1 ± 0.0% | 17.0 ± 0.1% | 19.5 ± 0.9% | 79.2 ± 1.1% | 100.0 ± 0.0% | 35.4 ± 1.0% | 4.2 ± 0.2 s |

**Intervention plausibility** - share of real test histories that stay under 10 m/s^2 after the intervention:

| Intervention | Before | After |
|---|---|---|
| speed = 2.0 | 84.5% | 54.6% |
| speed = 0.5 | 84.5% | 98.2% |
| brake = 0.8 | 84.5% | 89.8% |
| lateral = 3.0 | 84.5% | 84.5% |
