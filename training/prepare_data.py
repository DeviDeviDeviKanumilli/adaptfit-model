"""CLI for converting configured public sources into training windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .src.config import load_config, resolve_project_root
from .src.data.prepare import prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Move an existing prepared dataset to a timestamped backup before replacement.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    summary = prepare_dataset(
        config,
        resolve_project_root(args.config, args.project_root),
        replace_existing=args.replace_existing,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
