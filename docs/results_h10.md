### Horizon 1.0 s (10 past -> 10 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 12,568 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | FDE (m) |
|---|---|---|
| Stationary | 1.725 | 3.133 |
| Constant velocity | 0.231 | 0.519 |
| Kalman (q/r=17.7828) | 0.195 | 0.457 |
| Linear (ridge) | 0.194 | 0.438 |
| social | 0.229 ± 0.002 | 0.505 ± 0.003 |
| transformer | 0.220 ± 0.003 | 0.489 ± 0.006 |
| social+ridge prior | 0.197 ± 0.001 | 0.447 ± 0.004 |
| transformer+ridge prior | 0.192 ± 0.000 | 0.436 ± 0.001 |

**Conflict detection** — 24,172 not-yet-close co-occurring pairs (3-15 m apart now), 559 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 82.5 / 81.6 / 82.0 | 13.3 / 63.3 / 22.0 |
| Kalman (q/r=17.7828) | 79.0 / 82.8 / 80.9 | 13.2 / 64.6 / 21.9 |
| Linear (ridge) | 77.3 / 77.8 / 77.5 | 12.5 / 61.7 / 20.8 |
| social | 83.1 ± 0.2 / 81.3 ± 0.6 / 82.2 ± 0.3 | 13.3 ± 0.4 / 63.1 ± 0.6 / 22.0 ± 0.5 |
| transformer | 83.1 ± 0.3 / 80.7 ± 0.3 / 81.9 ± 0.3 | 14.3 ± 0.2 / 63.9 ± 0.4 / 23.3 ± 0.3 |
| social+ridge prior | 77.8 ± 0.2 / 79.6 ± 0.6 / 78.7 ± 0.4 | 12.4 ± 0.1 / 62.0 ± 0.4 / 20.6 ± 0.1 |
| transformer+ridge prior | 77.8 ± 0.2 / 79.2 ± 0.1 / 78.5 ± 0.2 | 12.6 ± 0.1 / 62.0 ± 0.4 / 21.0 ± 0.1 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| social | 99.8 ± 0.1 | 2.02 ± 0.00 | 0.09 ± 0.01 |
| transformer | 100.0 ± 0.0 | 2.02 ± 0.00 | 0.01 ± 0.00 |
| social+ridge prior | 99.9 ± 0.0 | 2.00 ± 0.00 | 0.01 ± 0.00 |
| transformer+ridge prior | 100.0 ± 0.0 | 2.00 ± 0.00 | 0.00 ± 0.00 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| social | 0.229 ± 0.002 | 0.231 ± 0.002 | 0.505 ± 0.003 | 0.512 ± 0.004 |
| social+ridge prior | 0.197 ± 0.001 | 0.194 ± 0.000 | 0.447 ± 0.004 | 0.440 ± 0.001 |
