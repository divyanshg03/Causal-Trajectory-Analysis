"""Optional experiment tracking (Weights & Biases or MLflow), imported lazily.

``--track wandb`` / ``--track mlflow`` on ``train`` logs the run arguments, per-epoch
train loss and validation errors, and the best epoch. Neither package is a dependency;
a clear error is raised if the chosen backend is not installed.
"""

BACKENDS = ("wandb", "mlflow")


class Tracker:
    """No-op tracker (``backend=None``) with the same interface as the real ones."""

    def __init__(self, backend=None, project="cftraj", run_name=None, params=None):
        self.backend = backend
        self._mod = None
        if backend is None:
            return
        if backend not in BACKENDS:
            raise ValueError(f"Unknown tracking backend '{backend}'. Choose from {BACKENDS}")
        try:
            self._mod = __import__(backend)
        except ImportError as err:
            raise RuntimeError(
                f"--track {backend} needs the package: pip install {backend}") from err
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if backend == "wandb":
            self._run = self._mod.init(project=project, name=run_name, config=params)
        else:
            self._mod.set_experiment(project)
            self._run = self._mod.start_run(run_name=run_name)
            self._mod.log_params({k: str(v)[:250] for k, v in params.items()})

    def log(self, metrics, step=None):
        if self._mod is None:
            return
        if self.backend == "wandb":
            self._mod.log(metrics, step=step)
        else:
            self._mod.log_metrics(metrics, step=step)

    def summary(self, values):
        if self._mod is None:
            return
        if self.backend == "wandb":
            self._run.summary.update(values)
        else:
            self._mod.log_metrics(values)

    def finish(self):
        if self._mod is None:
            return
        if self.backend == "wandb":
            self._run.finish()
        else:
            self._mod.end_run()
