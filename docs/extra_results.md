# Extended evaluation (v0.3)

Output of `python -m cftraj evaluate` after the 0.3 additions: bootstrap confidence intervals,
risk-tier calibration, neighbor-reaction and intervention-plausibility checks, and
best-of-K metrics. Models: the shipped seed-0 checkpoints, plus `social-mm5+ridge`, a
5-mode social model trained for this report (`--n-modes 5 --ridge-prior`, 14 epochs, best
epoch 4 by validation minADE; not shipped). The headline multi-seed tables stay in
`results_h10.md` / `results_h30.md`.

Reading guide:

- **ADE 95% CI** resamples whole tracks. The intervals of ridge and the social model overlap
  almost completely, so their ADE difference at 3 s is not significant.
- **minADE** is best-of-5 and is *not* comparable with single-trajectory ADE.
- **Neighbor reaction** is larger for far pairs than near ones: the social model's response to
  another agent does not reflect interaction strength, consistent with the context ablation.
- **Risk-tier calibration** shows the heuristic tiers are monotone only at the extremes
  (CRITICAL/HIGH vs SAFE); LOW and MEDIUM are barely distinguishable, and the F1-optimal TTC
  cut-off (4-4.5 s at the 3 s horizon) is below the 5 s MEDIUM bound.
- **Intervention plausibility**: about 15% of real histories already exceed 10 m/s^2 (label
  jitter), and doubling speed pushes another ~30% over the limit, so `speed = 2` is often out
  of distribution; `analyze` now warns when that happens.

### Horizon 1.0 s (10 past -> 10 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 12,568 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | ADE 95% CI | FDE (m) |
|---|---|---|---|
| Stationary | 1.725 | [1.545, 1.951] | 3.133 |
| Constant velocity | 0.231 | [0.209, 0.257] | 0.519 |
| Kalman (q/r=17.7828) | 0.195 | [0.177, 0.216] | 0.457 |
| Linear (ridge) | 0.194 | [0.179, 0.212] | 0.438 |
| tf+ridge | 0.192 | [0.176, 0.210] | 0.435 |

**Conflict detection** — 24,172 not-yet-close co-occurring pairs (3-15 m apart now), 559 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 82.5 / 81.6 / 82.0 | 13.3 / 63.3 / 22.0 |
| Kalman (q/r=17.7828) | 79.0 / 82.8 / 80.9 | 13.2 / 64.6 / 21.9 |
| Linear (ridge) | 77.3 / 77.8 / 77.5 | 12.5 / 61.7 / 20.8 |
| tf+ridge | 77.7 / 79.1 / 78.4 | 12.7 / 62.4 / 21.1 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| tf+ridge | 100.0 | 2.00 | 0.00 |

**Risk-tier calibration** - fraction of pairs followed by a real conflict (true future gap < 3 m), per predicted tier:

| Model | SAFE | LOW | MEDIUM | HIGH | CRITICAL | F1 (MEDIUM+) | best TTC cut-off |
|---|---|---|---|---|---|---|---|
| tf+ridge | 0.0% | 5.9% | 3.1% | 20.7% | 100.0% | 21.1% | 1.5 s |

**Intervention plausibility** - share of real test histories that stay under 10 m/s^2 after the intervention:

| Intervention | Before | After |
|---|---|---|
| speed = 2.0 | 86.6% | 56.7% |
| speed = 0.5 | 86.6% | 97.6% |
| brake = 0.8 | 86.6% | 88.0% |
| lateral = 3.0 | 86.6% | 86.6% |


### Horizon 3.0 s (10 past -> 30 future frames at 10 Hz)

Test split: KITTI tracking sequences 0018, 0019, 0020, 9,415 windows. Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std over seeds.

| Model | ADE (m) | ADE 95% CI | FDE (m) |
|---|---|---|---|
| Stationary | 4.006 | [3.549, 4.556] | 7.675 |
| Constant velocity | 1.248 | [1.088, 1.447] | 3.092 |
| Kalman (q/r=31.6228) | 1.165 | [1.016, 1.345] | 2.954 |
| Linear (ridge) | 1.074 | [0.959, 1.213] | 2.682 |
| social-mm5+ridge | 1.156 | [1.020, 1.313] | 2.823 |
| social+ridge | 1.074 | [0.960, 1.212] | 2.663 |

**Multi-modal models** (best of K modes per window):

| Model | K | minADE (m) | minFDE (m) |
|---|---|---|---|
| social-mm5+ridge | 5 | 0.684 | 1.666 |

**Conflict detection** — 15,135 not-yet-close co-occurring pairs (3-15 m apart now), 930 real conflicts (true future gap < 3 m). *Distance rule*: predicted gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic).

| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |
|---|---|---|
| Stationary | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Constant velocity | 70.7 / 71.5 / 71.1 | 46.3 / 31.2 / 37.3 |
| Kalman (q/r=31.6228) | 68.6 / 71.4 / 70.0 | 44.0 / 32.5 / 37.4 |
| Linear (ridge) | 66.7 / 67.6 / 67.2 | 43.7 / 31.9 / 36.9 |
| social-mm5+ridge | 65.5 / 63.3 / 64.4 | 40.8 / 30.5 / 34.9 |
| social+ridge | 65.7 / 69.7 / 67.6 | 41.1 / 31.4 / 35.6 |

**Counterfactual validity** — predicted travel when an agent's past speed is scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x).

| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |
|---|---|---|---|
| social-mm5+ridge | 95.3 | 2.04 | 0.01 |
| social+ridge | 99.2 | 1.98 | 0.13 |

**Scene-context ablation** (social model, neighbors masked out):

| Model | ADE with context | ADE without | FDE with | FDE without |
|---|---|---|---|---|
| social-mm5+ridge | 1.156 | 1.193 | 2.823 | 2.918 |
| social+ridge | 1.074 | 1.066 | 2.663 | 2.660 |

**Neighbor reaction** - shift of B's predicted final position (m) when A's past speed is doubled; should shrink with distance:

| Model | Mean | 95th pct | Pairs < 5 m | Pairs > 10 m |
|---|---|---|---|---|
| social-mm5+ridge | 0.018 | 0.010 | 0.005 | 0.021 |
| social+ridge | 0.077 | 0.299 | 0.035 | 0.120 |

**Risk-tier calibration** - fraction of pairs followed by a real conflict (true future gap < 3 m), per predicted tier:

| Model | SAFE | LOW | MEDIUM | HIGH | CRITICAL | F1 (MEDIUM+) | best TTC cut-off |
|---|---|---|---|---|---|---|---|
| social-mm5+ridge | 0.2% | 17.0% | 21.4% | 73.1% | 100.0% | 34.9% | 4.5 s |
| social+ridge | 0.1% | 17.1% | 18.3% | 79.6% | 100.0% | 35.6% | 4.0 s |

**Intervention plausibility** - share of real test histories that stay under 10 m/s^2 after the intervention:

| Intervention | Before | After |
|---|---|---|
| speed = 2.0 | 84.5% | 54.6% |
| speed = 0.5 | 84.5% | 98.2% |
| brake = 0.8 | 84.5% | 89.8% |
| lateral = 3.0 | 84.5% | 84.5% |

