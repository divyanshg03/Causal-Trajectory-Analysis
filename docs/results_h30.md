### Horizon 3.0 s (10 past -> 30 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 9,415 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | FDE (m) |
|---|---|---|
| Stationary | 4.006 | 7.675 |
| Constant velocity | 1.248 | 3.092 |
| Kalman (q/r=31.6228) | 1.165 | 2.954 |
| Linear (ridge) | 1.074 | 2.682 |
| social | 1.266 ± 0.009 | 3.025 ± 0.035 |
| transformer | 1.306 ± 0.005 | 3.139 ± 0.013 |
| social+ridge prior | 1.085 ± 0.009 | 2.684 ± 0.016 |
| transformer+ridge prior | 1.106 ± 0.002 | 2.758 ± 0.008 |

**Conflict detection** — 15,135 not-yet-close co-occurring pairs (3-15 m apart now), 930 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 70.7 / 71.5 / 71.1 | 46.3 / 31.2 / 37.3 |
| Kalman (q/r=31.6228) | 68.6 / 71.4 / 70.0 | 44.0 / 32.5 / 37.4 |
| Linear (ridge) | 66.7 / 67.6 / 67.2 | 43.7 / 31.9 / 36.9 |
| social | 70.9 ± 0.4 / 69.0 ± 1.2 / 69.9 ± 0.8 | 53.1 ± 3.5 / 31.0 ± 1.4 / 39.1 ± 2.0 |
| transformer | 70.2 ± 0.9 / 60.0 ± 0.4 / 64.7 ± 0.1 | 59.0 ± 2.6 / 26.0 ± 1.0 / 36.1 ± 1.4 |
| social+ridge prior | 66.2 ± 0.4 / 68.7 ± 1.1 / 67.4 ± 0.6 | 41.3 ± 1.0 / 31.0 ± 1.0 / 35.4 ± 1.0 |
| transformer+ridge prior | 66.9 ± 0.8 / 64.8 ± 0.6 / 65.9 ± 0.6 | 42.7 ± 0.6 / 29.8 ± 0.6 / 35.1 ± 0.6 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| social | 90.5 ± 0.7 | 2.07 ± 0.01 | 0.72 ± 0.04 |
| transformer | 98.6 ± 0.4 | 2.07 ± 0.01 | 0.03 ± 0.02 |
| social+ridge prior | 98.6 ± 0.5 | 1.98 ± 0.01 | 0.18 ± 0.05 |
| transformer+ridge prior | 99.4 ± 0.1 | 2.00 ± 0.01 | 0.03 ± 0.01 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| social | 1.266 ± 0.009 | 1.231 ± 0.011 | 3.025 ± 0.035 | 2.923 ± 0.012 |
| social+ridge prior | 1.085 ± 0.009 | 1.065 ± 0.001 | 2.684 ± 0.016 | 2.655 ± 0.004 |
