# Local regression benchmarks

These benchmarks are deliberately small and deterministic. They run offline
from a normal checkout and are intended to answer two questions after a code
change:

1. Are the phased genotypes exactly the same as a baseline build?
2. If not byte-for-byte identical, are they scientifically equivalent?

The suite runs three cases:

| Case | Coverage | Fixture |
|---|---|---|
| `common-unrelated` | `phase_common` without external phase information | 128 samples, 1,200 common sites across 10 Mb |
| `common-scaffolded` | withheld-scaffold-site regression for `phase_common` | the same target plus 300 truth-derived scaffold sites |
| `rare-scaffolded` | `phase_rare` with a common-site scaffold | 128 samples, 1,039 polymorphic sites in a 500 kb WGS slice, including 120 scaffold sites |

All target BCFs come from the repository's simulated test data. Their phased
input haplotypes are retained as independent simulator truth. The phasing code
reads target GTs as unordered genotypes; only the separately supplied scaffold
contributes input phase. The derived fixtures total less than 100 KiB of BCF
data and are generated under `/tmp/shapeit5-benchmarks-<uid>/`, never in the
checkout. They do not require GCS, S3, or participant data.

The source datasets are completely public and pinned to repository commit
`c34d4db3e99a2f7e23deb727671ae260901a5886`. If they are not present in the
checkout, the fixture script fetches them and their indexes over HTTPS:

- [array simulation](https://raw.githubusercontent.com/carbocation/shapeit/c34d4db3e99a2f7e23deb727671ae260901a5886/test/array/target.unrelated.bcf)
- [WGS simulation](https://raw.githubusercontent.com/carbocation/shapeit/c34d4db3e99a2f7e23deb727671ae260901a5886/test/wgs/target.unrelated.bcf)

## Running one build

Build `phase_common` and `phase_rare`, then run:

```sh
make benchmark
# equivalent to: python3 test/benchmarks/run.py --bin-dir .
```

`--bin-dir` may be a SHAPEIT5 repository root, a `bin/` directory, or a
directory containing the executables. A single-build run checks that every
variant and genotype is preserved, verifies scaffold phase, and reports switch
error against simulator truth. It also compares exact and orientation-invariant
canonical GT hashes with `baseline_specs.json`; a global haplotype-label flip
per sample is accepted, while a local phase change fails. Only hashes and code
are checked in—no derived BCF data. Results and full command logs default to
the same user-specific `/tmp/` directory. Set `SHAPEIT5_BENCHMARK_TMP` or pass
`--output-dir` to choose another temporary workspace.

Use `--prepare-only` to generate and verify the `/tmp/` fixtures without
requiring built SHAPEIT5 executables.

The phasing binaries still require the platform documented by SHAPEIT5 (in
particular AVX2 for the current `phase_common` implementation). Python 3 and
either HTSlib's `htsfile` command or `bcftools` are also required to inspect BCF
output.

## Comparing a candidate with a baseline

To compare against a baseline other than the recorded hashes, run both builds
with the same fixed seed and one thread:

```sh
python3 test/benchmarks/run.py \
  --baseline-bin-dir /path/to/baseline/shapeit5 \
  --bin-dir /path/to/candidate/shapeit5
```

The comparison first hashes the canonical variant/sample/GT stream, ignoring
BCF compression and provenance headers. If that differs, it accepts only a
scientifically equivalent result: identical variants and genotype dosages, no
unphased heterozygotes, and zero phase switches after allowing one global
haplotype-label flip per sample. Local phase changes fail.

Wall time is always reported but is not gated by default because these cases
are short. A deliberately controlled performance job can add, for example,
`--max-runtime-ratio 1.20`. Use `--case NAME` to run one case.

## Comparator tests

The standard-library-only tests exercise exact matches, global haplotype flips,
local switches, dosage changes, and scaffold subsets:

```sh
python3 -m unittest discover -s test/benchmarks/tests -v
```

## Rebuilding fixtures

Fixtures should rarely change. To regenerate them from `test/`, install HTSlib
headers/libraries and run:

```sh
test/benchmarks/regenerate_fixtures.sh
```

Set `HTSLIB_PREFIX` if HTSlib is not under `/usr/local` (Homebrew is detected
automatically). The runner invokes the script automatically when the `/tmp/`
fixtures are absent. The script builds `test/benchmarks/tools/subset_bcf.cpp`
and creates all derived BCFs and indexes. `fixture_specs.json` records canonical
GT hashes so fixture verification is independent of BCF compression.
