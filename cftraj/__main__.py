import argparse
import sys

from .analyze import analyze_pair, plot_result, print_report
from .data import load_windows
from .model import load_model
from .train import run_training

DEFAULT_LABELS = "data/training/label_02/0000.txt"


def build_parser():
    p = argparse.ArgumentParser(prog="cftraj", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--labels", default=DEFAULT_LABELS, help="KITTI label_02/XXXX.txt")
        sp.add_argument("--model", default="model.pth", help="weights path")

    t = sub.add_parser("train", help="train the trajectory transformer")
    common(t)
    t.add_argument("--epochs", type=int, default=250)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="run counterfactual collision analysis")
    common(a)
    a.add_argument("--factor", type=float, default=2.0, help="speed multiplier for agent A")
    a.add_argument("--max-distance", type=float, default=None,
                   help="reject pairs whose last observed points are farther apart")
    a.add_argument("--save-fig", default=None, help="write the plot to this path")
    a.add_argument("--no-show", action="store_true", help="do not open a plot window")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        windows = load_windows(args.labels)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.command == "train":
        run_training(windows, args.epochs, args.lr, args.model, args.seed)
        return

    res = analyze_pair(load_model(args.model), windows, args.factor, args.max_distance)
    if res is None:
        sys.exit("No co-occurring pair of agents found in this sequence.")
    print_report(res)
    if args.save_fig or not args.no_show:
        plot_result(res, args.save_fig, show=not args.no_show)


if __name__ == "__main__":
    main()
