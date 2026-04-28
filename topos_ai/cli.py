"""CLI entrypoints for running application and benchmark modules."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
APPLICATIONS_DIR = REPO_ROOT / "applications"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _available_modules(directory: Path) -> list[str]:
    return sorted(
        path.stem
        for path in directory.glob("*.py")
        if path.is_file() and not path.name.startswith("__")
    )


def _build_parser(kind: str, modules: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"topos-{kind}",
        description=f"Run ToposAI {kind} modules via package-friendly entrypoints.",
    )
    parser.add_argument(
        "module",
        choices=modules,
        help=f"{kind.capitalize()} module name (without .py)",
    )
    return parser


def _run_module(kind: str, directory: Path) -> None:
    modules = _available_modules(directory)
    parser = _build_parser(kind, modules)
    args = parser.parse_args()
    runpy.run_module(f"{kind}s.{args.module}", run_name="__main__")


def run_application() -> None:
    _run_module("application", APPLICATIONS_DIR)


def run_benchmark() -> None:
    _run_module("benchmark", BENCHMARKS_DIR)
