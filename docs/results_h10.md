### Horizon 1.0 s (10 past -> 10 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 12,568 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | ADE 95% CI | FDE (m) |
|---|---|---|---|
| Stationary | 1.725 | [1.545, 1.951] | 3.133 |
| Constant velocity | 0.231 | [0.209, 0.257] | 0.519 |
| Kalman (q/r=17.7828) | 0.195 | [0.177, 0.216] | 0.457 |
| Linear (ridge) | 0.194 | [0.179, 0.212] | 0.438 |
| LSTM | 0.231 ± 0.001 | [0.210, 0.255] | 0.514 ± 0.001 |
| Transformer | 0.220 ± 0.003 | [0.200, 0.241] | 0.489 ± 0.006 |
| Social | 0.229 ± 0.002 | [0.207, 0.252] | 0.505 ± 0.003 |
| Transformer + ridge prior | 0.192 ± 0.000 | [0.177, 0.210] | 0.436 ± 0.001 |
| Social + ridge prior | 0.197 ± 0.001 | [0.180, 0.217] | 0.447 ± 0.004 |

**Conflict detection** — 24,172 not-yet-close co-occurring pairs (3-15 m apart now), 559 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 82.5 / 81.6 / 82.0 | 13.3 / 63.3 / 22.0 |
| Kalman (q/r=17.7828) | 79.0 / 82.8 / 80.9 | 13.2 / 64.6 / 21.9 |
| Linear (ridge) | 77.3 / 77.8 / 77.5 | 12.5 / 61.7 / 20.8 |
| LSTM | 83.6 ± 0.3 / 81.1 ± 0.2 / 82.3 ± 0.0 | 15.0 ± 0.1 / 65.8 ± 0.3 / 24.4 ± 0.2 |
| Transformer | 83.1 ± 0.3 / 80.7 ± 0.3 / 81.9 ± 0.3 | 14.3 ± 0.2 / 63.9 ± 0.4 / 23.3 ± 0.3 |
| Social | 83.1 ± 0.2 / 81.3 ± 0.6 / 82.2 ± 0.3 | 13.3 ± 0.4 / 63.1 ± 0.6 / 22.0 ± 0.5 |
| Transformer + ridge prior | 77.8 ± 0.2 / 79.2 ± 0.1 / 78.5 ± 0.2 | 12.6 ± 0.1 / 62.0 ± 0.4 / 21.0 ± 0.1 |
| Social + ridge prior | 77.8 ± 0.2 / 79.6 ± 0.6 / 78.7 ± 0.4 | 12.4 ± 0.1 / 62.0 ± 0.4 / 20.6 ± 0.1 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| LSTM | 100.0 ± 0.0 | 2.02 ± 0.00 | 0.00 ± 0.00 |
| Transformer | 100.0 ± 0.0 | 2.02 ± 0.00 | 0.01 ± 0.00 |
| Social | 99.8 ± 0.1 | 2.02 ± 0.00 | 0.09 ± 0.01 |
| Transformer + ridge prior | 100.0 ± 0.0 | 2.00 ± 0.00 | 0.00 ± 0.00 |
| Social + ridge prior | 99.9 ± 0.0 | 2.00 ± 0.00 | 0.01 ± 0.00 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| Social | 0.229 ± 0.002 | 0.231 ± 0.002 | 0.505 ± 0.003 | 0.512 ± 0.004 |
| Social + ridge prior | 0.197 ± 0.001 | 0.194 ± 0.000 | 0.447 ± 0.004 | 0.440 ± 0.001 |

**Neighbor reaction** - shift of B's predicted final position (m) when A's past speed is doubled; should shrink with distance:

| Model | Mean | 95th pct | Pairs < 5 m | Pairs > 10 m |
|---|---|---|---|---|
| Social | 0.024 ± 0.003 | 0.112 ± 0.013 | 0.019 ± 0.001 | 0.031 ± 0.005 |
| Social + ridge prior | 0.018 ± 0.002 | 0.088 ± 0.006 | 0.011 ± 0.001 | 0.025 ± 0.002 |

**Risk-tier calibration** - fraction of pairs followed by a real conflict (true future gap < 3 m), per predicted tier:

| Model | SAFE | LOW | MEDIUM | HIGH | CRITICAL | F1 (MEDIUM+) | best TTC cut-off |
|---|---|---|---|---|---|---|---|
| LSTM | 0.0 ± 0.0% | 5.3 ± 0.0% | 4.4 ± 0.0% | 23.2 ± 0.1% | 100.0 ± 0.0% | 24.4 ± 0.2% | 2.0 ± 0.0 s |
| Transformer | 0.0 ± 0.0% | 5.6 ± 0.0% | 4.0 ± 0.2% | 22.6 ± 0.2% | 100.0 ± 0.0% | 23.3 ± 0.3% | 1.5 ± 0.0 s |
| Social | 0.0 ± 0.0% | 5.7 ± 0.1% | 3.9 ± 0.2% | 21.3 ± 0.4% | 100.0 ± 0.0% | 22.0 ± 0.5% | 1.5 ± 0.0 s |
| Transformer + ridge prior | 0.0 ± 0.0% | 6.0 ± 0.1% | 3.1 ± 0.1% | 20.5 ± 0.1% | 100.0 ± 0.0% | 21.0 ± 0.1% | 1.5 ± 0.0 s |
| Social + ridge prior | 0.0 ± 0.0% | 6.0 ± 0.1% | 3.4 ± 0.1% | 20.1 ± 0.1% | 100.0 ± 0.0% | 20.6 ± 0.1% | 1.5 ± 0.0 s |

**Intervention plausibility** - share of real test histories that stay under 10 m/s^2 after the intervention:

| Intervention | Before | After |
|---|---|---|
| speed = 2.0 | 86.6% | 56.7% |
| speed = 0.5 | 86.6% | 97.6% |
| brake = 0.8 | 86.6% | 88.0% |
| lateral = 3.0 | 86.6% | 86.6% |
