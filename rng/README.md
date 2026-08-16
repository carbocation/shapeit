# SHAPEIT Rust ABI

The `shapeit-rng` Rust static library provides versioned low-level kernels for
SHAPEIT. It does not allocate, retain mutable global state, or own any phasing
data. Its first ABIs are the counter-based random-number kernel and checked
bitmatrix transpose operations used by the common-variant HMM.

## Bitmatrix transpose

`shapeit_bitmatrix_subset_transpose_v1` selects an arbitrary list of rows from
a row-major bitmatrix and transposes a contiguous byte span into the layout
consumed by the HMM. `shapeit_bitmatrix_transpose_v1` transposes an aligned
rectangle between the persistent horizontal and vertical haplotype layouts,
without changing bytes outside that rectangle. Rust validates dimensions,
index bounds, and integer arithmetic before writing caller-owned output. On
x86-64 both operations perform runtime BMI2 detection and use tiled `PEXT`
kernels; other targets use portable byte-exact implementations.

`shapeit_bitmatrix_het_overlap_v1` computes the matching-heterozygote score
used for common-phasing IBD2 protection. It preserves the established inclusive
whole-byte interval and floating-point formula, while processing eight bytes at
a time with hardware POPCNT when available.

## Common HMM window

`shapeit_hmm_run_segment_double_v1` owns the complete double-precision fallback.
It covers a whole common-phasing HMM window: forward and backward recurrences,
transition contraction, and missing-genotype probabilities. The ABI accepts the
genotype graph, an already subset-transposed conditioning panel, model
parameters, and window coordinates. Rust validates the complete graph-derived
layout before writing any output, while all variable-size workspace remains
caller-owned.

`shapeit_hmm_run_segment_single_v1` owns the normal single-precision HMM with
exact 1/2/4/8-lane state compression and mixed-precision underflow recovery.
The common phaser uses the constant-time prevalidated entry point after its C++
adapter constructs a valid window layout; the fully validating entry point
remains available to other ABI callers.

The portable implementation is used on non-x86 targets. SHAPEIT's existing
x86-64 build contract requires AVX2 and FMA; the Rust build uses the same
features. The single-precision implementation preserves the existing AVX2/FMA
arithmetic and reduction order. The C++ adapters make one Rust call per HMM
window rather than crossing the ABI within a locus loop.

## Genotype graph construction

`shapeit_genotype_graph_sizes_v1` and `shapeit_genotype_graph_build_v1` own the
per-sample common-phasing graph builder. They preserve segment splitting,
scaffold ordering, 8-lane ambiguity codes, diplotype masks, missing counts, and
transition counts while reducing repeated scans of the packed variants. Output
storage remains caller-owned.

## Random-number generation

ABI version 1 uses Philox4x32-10. A random block is a pure function of:

```text
(master seed, domain, iteration, item, block counter) -> four uint32 words
```

The 64-bit Philox key is the master seed XOR a bijective permutation of the
packed `(domain, iteration)` pair. The 128-bit counter packs the 64-bit block
counter followed by the 64-bit item identifier. For a fixed master seed, every
logical stream therefore has a distinct key or counter prefix.

The C++ adapter in `xcftools/common/src/utils/random_number.h` pins conversion
of these words into doubles, bounded integers, samples, and permutations. OS
threads are deliberately absent from the ABI: callers must identify work by a
stable logical item such as a sample or PBWT group.

## Stable domains

Domain numbers are part of ABI version 1 and must not be renumbered:

| Value | Domain |
|---:|---|
| 0 | Serial/default operations |
| 1 | `phase_common` PBWT site selection |
| 2 | `phase_common` window splitting |
| 3 | `phase_common` MCMC sampling and missing-genotype imputation |
| 4 | `phase_rare` PBWT initial ordering |
| 5 | `phase_rare` PBWT site selection |
| 6 | `phase_rare` fallback-state selection |
| 7 | `phase_rare` PBWT solve ordering |
| 8 | `phase_common` sparse-PBWT fallback-state selection |

Beginning with ABI version 1, any incompatible change requires a new exported
ABI function and a new RNG version. Existing versioned mappings and conversion
behavior must remain available for reproducing results.

## Provenance

Philox was described by Salmon et al., *Parallel Random Numbers: As Easy as
1, 2, 3* (SC11). Constants and known-answer vectors are checked against the
permissively licensed Random123 reference implementation. The Rust code here
is an independent implementation released under SHAPEIT's MIT license.
