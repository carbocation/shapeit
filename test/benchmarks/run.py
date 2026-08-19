#!/usr/bin/env python3
"""Run deterministic local SHAPEIT5 regression benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from vcf_compare import (
    Comparison,
    canonical_gt_digest,
    compare_paths,
    load_dataset,
    scientific_gt_digest,
    validate_allele_count_metadata,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY = BENCHMARK_DIR.parents[1]
TEMP_ROOT = Path(os.environ.get("SHAPEIT5_BENCHMARK_TMP", f"/tmp/shapeit5-benchmarks-{os.getuid()}"))
FIXTURES = TEMP_ROOT / "fixtures"
DEFAULT_SEED = 15052011


@dataclass(frozen=True)
class Case:
    name: str
    binary: str
    truth: Path
    scaffold: Path | None
    arguments: tuple[str, ...]


CASES = {
    "common-unrelated": Case(
        "common-unrelated",
        "phase_common",
        FIXTURES / "common.truth.bcf",
        None,
        (
            "--input",
            str(FIXTURES / "common.truth.bcf"),
            "--region",
            "1",
            "--map",
            str(REPOSITORY / "test/info/chr1.gmap.gz"),
            "--seed",
            str(DEFAULT_SEED),
            "--thread",
            "1",
            "--mcmc-iterations",
            "1b,1p,1m",
        ),
    ),
    "common-scaffolded": Case(
        "common-scaffolded",
        "phase_common",
        FIXTURES / "common.truth.bcf",
        FIXTURES / "common.scaffold.bcf",
        (
            "--input",
            str(FIXTURES / "common.truth.bcf"),
            "--scaffold",
            str(FIXTURES / "common.scaffold.bcf"),
            "--region",
            "1",
            "--map",
            str(REPOSITORY / "test/info/chr1.gmap.gz"),
            "--seed",
            str(DEFAULT_SEED),
            "--thread",
            "1",
            "--mcmc-iterations",
            "1b,1p,1m",
        ),
    ),
    "rare-scaffolded": Case(
        "rare-scaffolded",
        "phase_rare",
        FIXTURES / "rare.truth.bcf",
        FIXTURES / "rare.scaffold.bcf",
        (
            "--input",
            str(FIXTURES / "rare.truth.bcf"),
            "--scaffold",
            str(FIXTURES / "rare.scaffold.bcf"),
            "--input-region",
            "1:1-500000",
            "--scaffold-region",
            "1:1-500000",
            "--seed",
            str(DEFAULT_SEED),
            "--thread",
            "1",
            "--output-format",
            "vcf",
            "--output-noPP",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one SHAPEIT5 build, or compare a candidate build with a baseline. "
            "Exact GT output is preferred; global per-sample haplotype flips are "
            "accepted as scientifically equivalent."
        )
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=REPOSITORY,
        help="candidate binary directory or SHAPEIT5 repository root (default: this repository)",
    )
    parser.add_argument(
        "--baseline-bin-dir",
        type=Path,
        help="optional baseline binary directory/repository root for differential regression",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="case to run; repeat the option to select multiple cases (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEMP_ROOT / "out",
        help="directory for outputs, logs, and results.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"phasing seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-runtime-ratio",
        type=float,
        default=None,
        help="fail if candidate/baseline wall time exceeds this ratio",
    )
    parser.add_argument(
        "--skip-fixture-check",
        action="store_true",
        help="skip canonical GT verification of generated fixtures",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="generate and verify fixtures without running SHAPEIT5 binaries",
    )
    return parser.parse_args()


def ensure_fixtures(*, verify: bool) -> None:
    specification = json.loads((BENCHMARK_DIR / "fixture_specs.json").read_text())
    required = [FIXTURES / name for name in specification]
    required.extend([path.with_suffix(".bcf.csi") for path in required])
    if any(not path.is_file() for path in required):
        try:
            subprocess.run([str(BENCHMARK_DIR / "regenerate_fixtures.sh")], check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("failed to generate benchmark fixtures") from error
    if not verify:
        return

    failures = []
    for name, expected in specification.items():
        dataset = load_dataset(FIXTURES / name)
        actual = {
            "samples": len(dataset.samples),
            "variants": len(dataset.variants),
            "canonical_gt_sha256": canonical_gt_digest(dataset),
        }
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("fixture integrity check failed:\n  " + "\n  ".join(failures))


def resolve_binary(location: Path, name: str) -> Path:
    candidates = (
        location / name,
        location / "bin" / name,
        location / name / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cannot find executable {name}; searched {searched}")


def with_argument(case: Case, option: str, value: object) -> Case:
    arguments = list(case.arguments)
    try:
        index = arguments.index(option)
    except ValueError as error:
        raise RuntimeError(f"benchmark case {case.name} has no {option} argument") from error
    arguments[index + 1] = str(value)
    return replace(case, arguments=tuple(arguments))


def run_case(
    case: Case,
    binary_root: Path,
    output_root: Path,
    label: str,
    recorded_baseline: dict[str, str] | None = None,
) -> dict[str, object]:
    binary = resolve_binary(binary_root, case.binary)
    case_dir = output_root / label / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "phased.bcf"
    log = case_dir / "run.log"
    for old_path in (output, output.with_suffix(".bcf.csi"), log):
        if old_path.exists():
            old_path.unlink()

    command = [str(binary), *case.arguments, "--output", str(output)]
    started = time.perf_counter()
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    log.write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"{label}/{case.name} failed ({process.returncode}); see {log}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{label}/{case.name} produced no output; see {log}")

    allele_count_records = validate_allele_count_metadata(output)

    truth = compare_paths(case.truth, output)
    if not truth.samples_equal or not truth.variants_equal or truth.genotype_errors:
        raise RuntimeError(
            f"{label}/{case.name} did not preserve variants/genotypes: {truth.as_dict()}"
        )
    if truth.unphased_heterozygotes:
        raise RuntimeError(f"{label}/{case.name} emitted unphased heterozygotes")

    output_dataset = load_dataset(output)
    exact_digest = canonical_gt_digest(output_dataset)
    scientific_digest = scientific_gt_digest(output_dataset)
    recorded_status: str | None = None
    if recorded_baseline:
        if exact_digest == recorded_baseline["exact_gt_sha256"]:
            recorded_status = "exact"
        elif scientific_digest == recorded_baseline["scientific_gt_sha256"]:
            recorded_status = "scientifically equivalent"
        else:
            raise RuntimeError(
                f"{label}/{case.name} differs from the recorded baseline: "
                f"exact={exact_digest}, scientific={scientific_digest}"
            )

    scaffold: Comparison | None = None
    if case.scaffold:
        scaffold = compare_paths(case.scaffold, output, allow_right_superset=True)
        if not scaffold.scientifically_equivalent:
            raise RuntimeError(
                f"{label}/{case.name} did not preserve scaffold phase: {scaffold.as_dict()}"
            )

    result: dict[str, object] = {
        "binary": str(binary),
        "command": command,
        "wall_seconds": elapsed,
        "output_bytes": output.stat().st_size,
        "output": str(output),
        "exact_gt_sha256": exact_digest,
        "scientific_gt_sha256": scientific_digest,
        "allele_count_records": allele_count_records,
        "truth": truth.as_dict(),
    }
    if recorded_status:
        result["recorded_baseline"] = recorded_status
    if scaffold:
        result["scaffold"] = scaffold.as_dict()
    print(
        f"{label:9} {case.name:19} {elapsed:8.3f}s  "
        f"truth switches={truth.switch_errors}/{truth.phase_transitions} "
        f"({truth.switch_rate:.6g})"
        + (f"  recorded={recorded_status}" if recorded_status else "")
    )
    return result


def compare_builds(
    case: Case,
    baseline: dict[str, object],
    candidate: dict[str, object],
    max_runtime_ratio: float | None,
) -> dict[str, object]:
    comparison = compare_paths(str(baseline["output"]), str(candidate["output"]))
    if not comparison.exact_gt and not comparison.scientifically_equivalent:
        raise RuntimeError(f"candidate changed {case.name} phasing: {comparison.as_dict()}")

    baseline_seconds = float(baseline["wall_seconds"])
    candidate_seconds = float(candidate["wall_seconds"])
    runtime_ratio = candidate_seconds / baseline_seconds if baseline_seconds else 0.0
    if max_runtime_ratio is not None and runtime_ratio > max_runtime_ratio:
        raise RuntimeError(
            f"candidate {case.name} runtime ratio {runtime_ratio:.3f} exceeds "
            f"{max_runtime_ratio:.3f}"
        )
    status = "exact" if comparison.exact_gt else "scientifically equivalent"
    print(f"compare   {case.name:19} {status}; runtime ratio={runtime_ratio:.3f}")
    return {**comparison.as_dict(), "runtime_ratio": runtime_ratio}


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_runtime_ratio is not None and args.max_runtime_ratio <= 0:
        raise ValueError("--max-runtime-ratio must be positive")
    ensure_fixtures(verify=not args.skip_fixture_check)
    if args.prepare_only:
        print(f"fixtures  {FIXTURES}")
        return 0

    selected = args.case or list(CASES)
    recorded_cases = json.loads((BENCHMARK_DIR / "baseline_specs.json").read_text())["cases"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "seed": args.seed,
        "candidate": {},
        "baseline": {},
        "comparison": {},
    }

    for name in selected:
        case = with_argument(CASES[name], "--seed", args.seed)
        if args.baseline_bin_dir:
            baseline = run_case(case, args.baseline_bin_dir, output_dir, "baseline")
            results["baseline"][name] = baseline  # type: ignore[index]
        recorded_baseline = (
            recorded_cases[name]
            if not args.baseline_bin_dir and args.seed == DEFAULT_SEED
            else None
        )
        candidate = run_case(
            case, args.bin_dir, output_dir, "candidate", recorded_baseline
        )
        results["candidate"][name] = candidate  # type: ignore[index]
        if args.baseline_bin_dir:
            results["comparison"][name] = compare_builds(  # type: ignore[index]
                case, baseline, candidate, args.max_runtime_ratio
            )

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
