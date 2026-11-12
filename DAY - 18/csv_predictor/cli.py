"""Command-line interface: `predict-csv input.csv -o predictions.csv`."""

import argparse
import sys

from .predict import predict_csv


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predict-csv",
        description="Run the bundled ML pipeline on a CSV and write predictions.",
    )
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        default="predictions.csv",
        help="Path for the output CSV (default: predictions.csv)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Optional path to a custom .joblib pipeline (default: bundled model)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = predict_csv(args.input_csv, args.output, args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(result)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
