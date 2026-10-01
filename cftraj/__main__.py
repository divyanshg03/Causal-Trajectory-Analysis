import argparse
import re
import sys
from pathlib import Path

from .analyze import analyze_pair, find_escalation, plot_result, print_report
from .counterfactual import INTERVENTIONS
from .data import load_split, load_windows
from .evaluate import parse_ckpt_specs, run_evaluation
from .model import ARCHS, load_model
from .train import pick_device, train_model

DEFAULT_DIR = "data/training/label_02"
DEFAULT_MODEL = "checkpoints/social_h30.pt"


def seq_from_path(path):
    """KITTI label files are named ``NNNN.txt``; fall back to 0."""
    m = re.fullmatch(r"\d+", Path(path).stem)
    return int(m.group()) if m else 0


def build_parser():
    p = argparse.ArgumentParser(prog="cftraj")
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
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")

    e = sub.add_parser("evaluate", help="ADE/FDE, conflict detection and validity on test split")
    e.add_argument("--label-dir", default=DEFAULT_DIR)
    e.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH[,PATH...]",
                   help="learned model(s); repeat per model, comma-separate seeds")
    e.add_argument("--out-dir", default="docs")
    e.add_argument("--device", default="auto")

    def scene_args(sp):
        sp.add_argument("--labels", default=f"{DEFAULT_DIR}/0018.txt", help="one KITTI label file")
        sp.add_argument("--seq", type=int, default=None,
                        help="sequence id (default: parsed from the label file name)")
        sp.add_argument("--model", default=DEFAULT_MODEL)

    a = sub.add_parser("analyze", help="counterfactual collision analysis on one sequence")
    scene_args(a)
    a.add_argument("--intervention", choices=INTERVENTIONS, default="speed",
                   help="speed: scale A's past speed; lateral: drift A sideways (m)")
    a.add_argument("--value", type=float, default=2.0,
                   help="speed factor, or lateral offset in meters")
    a.add_argument("--max-distance", type=float, default=None,
                   help="reject pairs whose last observed points are farther apart (m)")
    a.add_argument("--find-escalation", action="store_true",
                   help="scan all nearby pairs for the one whose risk the intervention raises most")
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            kw = {"past_len": args.past_len, "future_len": args.future_len}
            train_w = load_split(args.label_dir, "train", **kw)
            val_w = load_split(args.label_dir, "val", **kw)
            out = args.out or f"outputs/{args.arch}.pt"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            train_model(args.arch, train_w, val_w, out, args.epochs, args.batch_size,
                        args.lr, patience=args.patience, seed=args.seed, device=args.device,
                        speed_aug=tuple(args.speed_aug) if args.speed_aug else None,
                        **({"linear_skip": True, "residual": False} if args.ridge_prior else {}))
        elif args.command == "evaluate":
            run_evaluation(args.label_dir, parse_ckpt_specs(args.ckpt), args.out_dir, args.device)
        elif args.command == "analyze":
            model, windows, _ = _load_scene(args)
            if args.find_escalation:
                res = find_escalation(model, windows, args.intervention, args.value,
                                      mask_neighbors=args.no_context)
                if res is None:
                    sys.exit("This intervention does not raise the risk of any pair here.")
            else:
                res = analyze_pair(model, windows, args.intervention, args.value,
                                   args.max_distance, mask_neighbors=args.no_context)
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
