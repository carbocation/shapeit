#!/usr/bin/env python3
"""Exercise deterministic PBWT fallback and rare-state boundary cases."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from run import CASES, DEFAULT_SEED, REPOSITORY, TEMP_ROOT, Case, ensure_fixtures, run_case, with_argument
from vcf_compare import compare_paths


DEFAULT_THREADS = (1, 4)


def with_extra_arguments(case: Case, name: str, *arguments: str) -> Case:
    return replace(case, name=name, arguments=(*case.arguments, *arguments))


SCENARIOS = {
    "common-sparse-fallback": with_extra_arguments(
        CASES["common-unrelated"],
        "common-sparse-fallback",
        "--pbwt-mac",
        "100000",
    ),
    "rare-odd-depth": with_extra_arguments(
        CASES["rare-scaffolded"],
        "rare-odd-depth",
        "--pbwt-depth-common",
        "3",
    ),
    "rare-empty-pbwt": with_extra_arguments(
        CASES["rare-scaffolded"],
        "rare-empty-pbwt",
        "--pbwt-depth-common",
        "3",
        "--pbwt-depth-rare",
        "0",
        "--pbwt-mac",
        "100000",
    ),
    "rare-map-tail": with_extra_arguments(
        CASES["rare-scaffolded"],
        "rare-map-tail",
        "--map",
        str(REPOSITORY / "test/info/chr1.truncated.gmap"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=REPOSITORY)
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    parser.add_argument("--thread", action="append", dest="threads", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir", type=Path, default=TEMP_ROOT / "edge-regressions"
    )
    parser.add_argument("--skip-fixture-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threads = args.threads or list(DEFAULT_THREADS)
    if args.seed < 0 or any(count < 1 for count in threads):
        raise ValueError("seed must be non-negative and thread counts must be positive")
    if len(set(threads)) != len(threads) or len(threads) < 2:
        raise ValueError("provide at least two unique thread counts")

    ensure_fixtures(verify=not args.skip_fixture_check)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "seed": args.seed,
        "threads": threads,
        "scenarios": {},
    }

    for name in args.scenario or list(SCENARIOS):
        case = with_argument(SCENARIOS[name], "--seed", args.seed)
        reference: dict[str, object] | None = None
        scenario_results: dict[str, object] = {}
        for count in threads:
            current = run_case(
                with_argument(case, "--thread", count),
                args.bin_dir,
                output_dir,
                f"threads-{count}",
            )
            scenario_results[str(count)] = current
            if reference is None:
                reference = current
            elif current["exact_gt_sha256"] != reference["exact_gt_sha256"]:
                comparison = compare_paths(str(reference["output"]), str(current["output"]))
                raise RuntimeError(
                    f"{name} changed at --thread {count}: {comparison.as_dict()}"
                )
        results["scenarios"][name] = scenario_results  # type: ignore[index]
        print(f"edge      {name:23} exact at {', '.join(map(str, threads))}")

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
