#!/usr/bin/env python3
"""Alternating repeated wall-time comparison of two SHAPEIT5 builds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from run import CASES, DEFAULT_SEED, REPOSITORY, TEMP_ROOT, run_case, with_argument
from vcf_compare import compare_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin-dir", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=REPOSITORY)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--thread", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir", type=Path, default=TEMP_ROOT / "performance-comparison"
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "runs_seconds": values,
    }


def main() -> int:
    args = parse_args()
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2")
    if args.thread < 1 or args.seed < 0:
        raise ValueError("--thread must be positive and --seed must be non-negative")

    selected = args.case or list(CASES)
    output_dir = args.output_dir.resolve()
    results: dict[str, object] = {
        "seed": args.seed,
        "thread": args.thread,
        "repetitions": args.repeat,
        "cases": {},
    }

    for name in selected:
        case = with_argument(
            with_argument(CASES[name], "--seed", args.seed), "--thread", args.thread
        )
        timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
        exact = True
        scientific = True
        for repetition in range(args.repeat):
            order = (
                ("baseline", "candidate")
                if repetition % 2 == 0
                else ("candidate", "baseline")
            )
            run_results: dict[str, dict[str, object]] = {}
            for label in order:
                binary_root = args.baseline_bin_dir if label == "baseline" else args.bin_dir
                current = run_case(
                    case,
                    binary_root,
                    output_dir,
                    f"repetition-{repetition}/{label}",
                )
                run_results[label] = current
                timings[label].append(float(current["wall_seconds"]))

            comparison = compare_paths(
                str(run_results["baseline"]["output"]),
                str(run_results["candidate"]["output"]),
            )
            exact &= comparison.exact_gt
            scientific &= comparison.scientifically_equivalent
            if not comparison.scientifically_equivalent:
                raise RuntimeError(
                    f"candidate changed {name} in repetition {repetition}: {comparison.as_dict()}"
                )

        baseline = summarize(timings["baseline"])
        candidate = summarize(timings["candidate"])
        ratio = float(candidate["median_seconds"]) / float(baseline["median_seconds"])
        results["cases"][name] = {  # type: ignore[index]
            "baseline": baseline,
            "candidate": candidate,
            "median_runtime_ratio": ratio,
            "exact": exact,
            "scientifically_equivalent": scientific,
        }
        equality = "exact" if exact else "scientifically equivalent"
        print(
            f"performance {name:19} {equality}; "
            f"baseline={float(baseline['median_seconds']):.3f}s "
            f"candidate={float(candidate['median_seconds']):.3f}s "
            f"ratio={ratio:.3f}"
        )

    results_path = output_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results     {results_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1)
