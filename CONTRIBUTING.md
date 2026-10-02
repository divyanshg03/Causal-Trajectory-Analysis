# Contributing

```bash
pip install -r requirements-dev.txt
make lint typecheck test      # ruff, mypy, pytest
make cov                      # with coverage (CI requires >= 80%)
```

- The tests need no dataset: `cftraj.synthetic.write_synthetic_kitti` writes KITTI-format
  labels. Tests that need real KITTI data or shipped checkpoints skip themselves when absent.
- Keep claims honest: new metrics go through `evaluate`, results land in `docs/`, and
  `docs/` tables must be regenerated with the command that produced them
  (`scripts/reproduce.sh`).
- New interventions: add the function and an `INTERVENTIONS` entry in
  `cftraj/counterfactual.py` (keep the current position fixed, accept leading batch
  dimensions) plus a test that checks the shape and endpoint.
- Checkpoints are listed with SHA-256 hashes in `checkpoints/MANIFEST.json`
  (`python -c "from cftraj.download import write_manifest; write_manifest('checkpoints')"`).
