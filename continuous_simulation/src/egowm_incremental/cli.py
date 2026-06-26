from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_project_config
from .orchestrator import format_command, prepare_quicktest_run, run_quicktest, validate_project_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EgoWM incremental inference subcommand.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate paths and environment resolution.")
    validate_parser.add_argument("--config", required=True, help="Path to project YAML config.")
    validate_parser.add_argument("--mode", default="full", help="Mode used for validation.")

    print_parser = subparsers.add_parser("print-command", help="Print the resolved backend command.")
    print_parser.add_argument("--config", required=True, help="Path to project YAML config.")
    print_parser.add_argument("--mode", default="recon_visualize", help="Incremental run mode.")
    print_parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    print_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    run_parser = subparsers.add_parser(
        "run-incremental",
        aliases=["quicktest"],
        help="Run the incremental inference pipeline.",
    )
    run_parser.add_argument("--config", required=True, help="Path to project YAML config.")
    run_parser.add_argument("--mode", default="recon_visualize", help="Incremental run mode.")
    run_parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate and print the command without executing it.")
    run_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    return parser


def _strip_remainder(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_project_config(args.config)

    if args.command == "validate-config":
        validate_project_config(config, args.mode)
        print(json.dumps({"status": "ok", "mode": args.mode}, indent=2))
        return 0

    extra_args = _strip_remainder(list(getattr(args, "extra_args", [])))
    output_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", "") else None
    run = prepare_quicktest_run(
        config,
        mode=args.mode,
        output_dir=output_dir,
        extra_args=extra_args,
    )

    if args.command == "print-command":
        print(format_command(run.command))
        return 0

    print(format_command(run.command))
    return run_quicktest(run, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
