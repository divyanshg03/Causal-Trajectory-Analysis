import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__
from .analyze import analyze_pair, find_escalation, plot_result, print_report
from .counterfactual import INTERVENTIONS
from .data import load_split, load_windows
from .evaluate import parse_ckpt_specs, run_evaluation
from .model import ARCHS, load_model
from .tracking import Tracker
from .train import pick_device, train_model

DEFAULT_DIR = "data/training/label_02"
DEFAULT_MODEL = "checkpoints/social_h30.pt"


def seq_from_path(path):
    """KITTI label files are named ``NNNN.txt``; fall back to 0."""
    m = re.fullmatch(r"\d+", Path(path).stem)
    return int(m.group()) if m else 0


def load_config(path):
    """Read run options from a ``.json`` (or ``.yaml``/``.yml``, needs PyYAML) file."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as err:
            raise ValueError("YAML configs need PyYAML (pip install pyyaml)") from err
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a mapping of option names to values")
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def build_parser():
    p = argparse.ArgumentParser(prog="cftraj")
    p.add_argument("--version", action="version", version=f"cftraj {__version__}")
    p.add_argument("--config", default=None,
                   help="JSON/YAML file of option defaults (command-line flags win)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a trajectory predictor")
    t.add_argument("--arch", choices=sorted(ARCHS), default="transformer")
    t.add_argument("--label-dir", default=DEFAULT_DIR, help="KITTI label_02 directory")
    t.add_argument("--out", default=None, help="checkpoint path (default outputs/<arch>.pt)")
    t.add_argument("--past-len", type=int, default=10)
    t.add_argument("--future-len", type=int, default=10, help="frames to predict (10 = 1 s)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--ridge-prior", action="store_true",
                   help="start from a ridge-regression linear path and learn a correction")
    t.add_argument("--speed-aug", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="augment by replaying targets at a random speed factor in [LO, HI]")
    t.add_argument("--n-modes", type=int, default=1,
                   help="predict K trajectories (winner-takes-all training); 1 = single future")
    t.add_argument("--track", choices=("wandb", "mlflow"), default=None,
                   help="log the run to Weights & Biases or MLflow (must be installed)")
    t.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    t.add_argument("--cache-dir", default=None, help="cache preprocessed windows here")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")

    e = sub.add_parser("evaluate", help="ADE/FDE, conflict detection and validity on test split")
    e.add_argument("--label-dir", default=DEFAULT_DIR)
    e.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH[,PATH...]",
                   help="learned model(s); repeat per model, comma-separate seeds")
    e.add_argument("--out-dir", default="docs")
    e.add_argument("--cache-dir", default=None, help="cache preprocessed windows here")
    e.add_argument("--device", default="auto")

    c = sub.add_parser("crossval", help="sequence-level k-fold benchmark (baselines, +1 model)")
    c.add_argument("--label-dir", default=DEFAULT_DIR)
    c.add_argument("--folds", type=int, default=5)
    c.add_argument("--future-len", type=int, default=10)
    c.add_argument("--arch", choices=sorted(ARCHS), default=None,
                   help="also train and score this architecture in every fold")
    c.add_argument("--epochs", type=int, default=30)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--out", default=None, help="write the markdown table here")
    c.add_argument("--device", default="cpu")

    eu = sub.add_parser("ethucy", help="leave-one-scene-out benchmark on ETH/UCY pedestrians")
    eu.add_argument("--root", default="data/ethucy", help="<root>/<scene>/{train,val,test}/*.txt")
    eu.add_argument("--scenes", nargs="+", default=["eth", "hotel", "univ", "zara1", "zara2"])
    eu.add_argument("--seeds", type=int, nargs="+", default=[0])
    eu.add_argument("--epochs", type=int, default=60)
    eu.add_argument("--out", default=None, help="write the markdown table here")
    eu.add_argument("--device", default="auto")

    sv = sub.add_parser("serve", help="web explorer (FastAPI backend + browser UI)")
    sv.add_argument("--label-dir", default=DEFAULT_DIR)
    sv.add_argument("--ckpt-dir", default="checkpoints")
    sv.add_argument("--image-dir", default=None, help="KITTI image_02 folder (default: next to label_02)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--device", default="cpu")

    dl = sub.add_parser("download", help="fetch the shipped checkpoints listed in MANIFEST.json")
    dl.add_argument("--dir", default="checkpoints")
    dl.add_argument("--base-url", default=None, help="release URL holding the .pt files")
    dl.add_argument("--force", action="store_true")

    def scene_args(sp):
        sp.add_argument("--labels", default=f"{DEFAULT_DIR}/0018.txt", help="one KITTI label file")
        sp.add_argument("--seq", type=int, default=None,
                        help="sequence id (default: parsed from the label file name)")
        sp.add_argument("--model", default=DEFAULT_MODEL)

    a = sub.add_parser("analyze", help="counterfactual collision analysis on one sequence")
    scene_args(a)
    a.add_argument("--intervention", choices=INTERVENTIONS, default="speed",
                   help="speed: scale A's past speed; lateral: drift A sideways (m); "
                        "brake: A decelerated by value in [0,1]; delay: A started value frames ago")
    a.add_argument("--value", type=float, default=2.0,
                   help="speed factor, lateral offset (m), brake strength or delay (frames)")
    a.add_argument("--max-distance", type=float, default=None,
                   help="reject pairs whose last observed points are farther apart (m)")
    a.add_argument("--find-escalation", action="store_true",
                   help="scan all nearby pairs for the one whose risk the intervention raises most")
    a.add_argument("--samples", type=int, default=0,
                   help="draw this many sampled futures to report P(overlap) / P(MEDIUM+)")
    a.add_argument("--no-context", action="store_true",
                   help="social models: ignore neighboring agents")
    a.add_argument("--save-fig", default=None, help="write the plot to this path")
    a.add_argument("--no-show", action="store_true", help="do not open a plot window")

    d = sub.add_parser("demo", help="render a camera + BEV risk GIF for a sequence")
    scene_args(d)
    d.add_argument("--image-dir", default=None,
                   help="KITTI image_02/NNNN folder (default: next to label_02)")
    d.add_argument("--first-frame", type=int, default=20)
    d.add_argument("--n-frames", type=int, default=50)
    d.add_argument("--step", type=int, default=2)
    d.add_argument("--out", default="docs/demo.gif")

    at = sub.add_parser("attention", help="plot a social model's attention over neighbors")
    scene_args(at)
    at.add_argument("--index", type=int, default=None, help="window index (default: busiest)")
    at.add_argument("--out", default="docs/attention.png")
    return p


def _load_scene(args):
    model = load_model(args.model, pick_device("cpu"))
    seq = args.seq if args.seq is not None else seq_from_path(args.labels)
    windows = load_windows(args.labels, model.past_len, model.future_len, seq=seq)
    return model, windows, seq


def parse_args(argv=None):
    """Parse ``argv``; options from ``--config`` become defaults the command line overrides."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        cfg = load_config(args.config)
        unknown = sorted(set(cfg) - set(vars(args)))
        if unknown:
            raise ValueError(f"Unknown option(s) in {args.config}: {', '.join(unknown)}")
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        sub.choices[args.command].set_defaults(**cfg)
        args = parser.parse_args(argv)
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        sys.exit(str(err))
    try:
        if args.command == "train":
            kw = {"past_len": args.past_len, "future_len": args.future_len,
                  "cache_dir": args.cache_dir}
            train_w = load_split(args.label_dir, "train", **kw)
            val_w = load_split(args.label_dir, "val", **kw)
            out = args.out or f"outputs/{args.arch}.pt"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            tracker = Tracker(args.track, run_name=Path(out).stem, params=vars(args))
            try:
                train_model(args.arch, train_w, val_w, out, args.epochs, args.batch_size,
                            args.lr, patience=args.patience, seed=args.seed, device=args.device,
                            tracker=tracker,
                            speed_aug=tuple(args.speed_aug) if args.speed_aug else None,
                            amp=args.amp, run_args=vars(args),
                            **({"n_modes": args.n_modes} if args.n_modes > 1 else {}),
                            **({"linear_skip": True, "residual": False} if args.ridge_prior else {}))
            finally:
                tracker.finish()
        elif args.command == "evaluate":
            run_evaluation(args.label_dir, parse_ckpt_specs(args.ckpt), args.out_dir, args.device,
                           cache_dir=args.cache_dir)
        elif args.command == "crossval":
            from .crossval import cross_validate, render

            cv = cross_validate(args.label_dir, args.folds, future_len=args.future_len,
                                arch=args.arch, epochs=args.epochs, seed=args.seed,
                                device=args.device, cache_dir=args.cache_dir)
            text = render(cv)
            print("\n" + text)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(text + "\n", encoding="utf-8")
        elif args.command == "ethucy":
            from .ethucy import render as render_eu
            from .ethucy import run_ethucy

            res = run_ethucy(args.root, args.scenes, seeds=tuple(args.seeds),
                             epochs=args.epochs, device=pick_device(args.device))
            text = render_eu(res)
            print("\n" + text)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(text + "\n", encoding="utf-8")
        elif args.command == "serve":
            try:
                from .server import serve
            except ImportError as err:
                raise RuntimeError("The web explorer needs: pip install 'cftraj[web]'") from err
            print(f"Open http://{args.host}:{args.port}")
            serve(args.host, args.port, label_dir=args.label_dir, ckpt_dir=args.ckpt_dir,
                  image_dir=args.image_dir, device=args.device)
        elif args.command == "download":
            from .download import DEFAULT_BASE_URL, download_checkpoints

            got = download_checkpoints(args.dir, args.base_url or DEFAULT_BASE_URL, args.force)
            print(f"Downloaded {len(got)} file(s): {', '.join(got) or 'nothing to do'}")
        elif args.command == "analyze":
            model, windows, _ = _load_scene(args)
            if args.find_escalation:
                res = find_escalation(model, windows, args.intervention, args.value,
                                      mask_neighbors=args.no_context, n_samples=args.samples)
                if res is None:
                    sys.exit("This intervention does not raise the risk of any pair here.")
            else:
                res = analyze_pair(model, windows, args.intervention, args.value,
                                   args.max_distance, mask_neighbors=args.no_context,
                                   n_samples=args.samples)
                if res is None:
                    sys.exit("No co-occurring pair of agents found in this sequence.")
            print_report(res)
            if args.save_fig or not args.no_show:
                plot_result(res, args.save_fig, show=not args.no_show)
        elif args.command == "demo":
            from .viz import make_demo

            model, windows, seq = _load_scene(args)
            image_dir = args.image_dir or str(
                Path(args.labels).parent.parent / "image_02" / f"{seq:04d}")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            n = make_demo(model, windows, args.labels, image_dir, args.out,
                          args.first_frame, args.n_frames, args.step)
            print(f"Wrote {n} frames to {args.out}")
        else:
            from .viz import plot_attention

            model, windows, _ = _load_scene(args)
            idx = args.index
            if idx is None:
                idx = int(windows.nmask.sum(1).argmax())
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            plot_attention(model, windows, idx, args.out)
            print(f"Wrote attention plot for window {idx} to {args.out}")
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
