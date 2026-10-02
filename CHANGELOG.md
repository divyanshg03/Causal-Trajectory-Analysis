# Changelog

## 0.3.0

### Added
- Interventions `brake` and `delay`, and a physical-plausibility check (peak acceleration) that
  `analyze` uses to warn about out-of-distribution counterfactuals.
- Multi-modal prediction (`--n-modes K`, winner-takes-all loss, best-of-K validation) with
  minADE/minFDE in `evaluate`.
- Uncertainty: `analyze --samples N` (MC dropout or mode sampling) reports P(overlap) and
  P(MEDIUM+).
- Risk calibration (tier reliability, F1-optimal TTC cut-off), neighbor-reaction and
  intervention-plausibility validity checks; 95% track-level bootstrap CIs on ADE.
- `evaluate` accepts checkpoints of several horizons in one call.
- `crossval` (sequence-level k-fold), `download` (SHA-256-verified checkpoint fetch),
  `--config` (JSON/YAML option defaults), `--cache-dir` (window cache), `--amp`, `--version`.
- Run provenance (version, torch, git commit, args) stored next to each checkpoint.
- `cftraj.synthetic`, `cftraj.data.load_csv_trajectories`, `py.typed`, mypy, coverage gate,
  CI matrix (Linux/Windows, Python 3.9/3.12), Makefile, `scripts/reproduce.sh`.

### Changed
- `find_escalation` predicts all pairs in batched calls instead of one pair at a time
  (same result, much faster).
- Models gained `forward_modes`; `forward` still returns a single `(B, F, 2)` future, and
  existing checkpoints load unchanged.

## 0.2.0
- `cftraj` package with CLI, tests and CI; pair selection and TTC unit fixes.
