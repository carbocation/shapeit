#!/usr/bin/env python3
"""Verify that phasing is invariant to the worker-thread count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run import (
    CASES,
    DEFAULT_SEED,
    REPOSITORY,
    TEMP_ROOT,
    Case,
    ensure_fixtures,
    run_case,
    with_argument,
)
from vcf_compare import compare_paths


DEFAULT_THREADS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Require exact canonical GT output at every requested thread count."
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=REPOSITORY,
        help="binary directory or SHAPEIT5 repository root (default: this repository)",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="case to run; repeat to select multiple cases (default: all)",
    )
    parser.add_argument(
        "--thread",
        action="append",
        dest="threads",
        type=int,
        help="worker-thread count; repeat to select several (default: 1, 2, 4, 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEMP_ROOT / "thread-determinism",
        help="directory for outputs, logs, and results.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"phasing seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--skip-fixture-check",
        action="store_true",
        help="skip canonical GT verification of generated fixtures",
    )
    return parser.parse_args()


def with_thread_count(case: Case, threads: int) -> Case:
    return with_argument(case, "--thread", threads)


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    threads = args.threads or list(DEFAULT_THREADS)
    if any(count <= 0 for count in threads):
        raise ValueError("thread counts must be positive")
    if len(set(threads)) != len(threads):
        raise ValueError("thread counts must be unique")
    if len(threads) < 2:
        raise ValueError("at least two thread counts are required")

    ensure_fixtures(verify=not args.skip_fixture_check)
    selected = args.case or list(CASES)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"seed": args.seed, "threads": threads, "cases": {}}

    for name in selected:
        reference: dict[str, object] | None = None
        case_results: dict[str, object] = {}
        for count in threads:
            result = run_case(
                with_thread_count(with_argument(CASES[name], "--seed", args.seed), count),
                args.bin_dir,
                output_dir,
                f"threads-{count}",
            )
            case_results[str(count)] = result
            if reference is None:
                reference = result
                continue
            if result["exact_gt_sha256"] != reference["exact_gt_sha256"]:
                comparison = compare_paths(str(reference["output"]), str(result["output"]))
                raise RuntimeError(
                    f"{name} changed at --thread {count}: {comparison.as_dict()}"
                )
        results["cases"][name] = case_results  # type: ignore[index]
        print(f"threads   {name:19} exact at {', '.join(map(str, threads))}")

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results   {results_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1)
