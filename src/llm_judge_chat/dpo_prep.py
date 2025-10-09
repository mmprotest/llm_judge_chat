"""Command-line tool to export DPO preference pairs."""

from __future__ import annotations

import argparse

from .logging_io import export_pairs_for_dpo


def main() -> None:
    """Parse arguments and run export."""

    parser = argparse.ArgumentParser(description="Export DPO preference pairs from logs.")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory containing session logs")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--min_gap", type=float, default=0.5, help="Minimum score gap between chosen and rejected")
    args = parser.parse_args()

    export_pairs_for_dpo(args.logs_dir, args.out, min_gap=args.min_gap)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
