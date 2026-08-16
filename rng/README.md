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

## Genotype graph construction and sampling

`shapeit_genotype_graph_sizes_v1` and `shapeit_genotype_graph_build_v1` own the
per-sample common-phasing graph builder. They preserve segment splitting,
scaffold ordering, 8-lane ambiguity codes, diplotype masks, missing counts, and
transition counts while reducing repeated scans of the packed variants. Output
storage remains caller-owned.

`shapeit_genotype_sample_v1` samples a complete graph in one call, including
forward/backward transition sampling, missing-genotype imputation, and applying
ambiguity codes to the packed haplotypes. It reconstructs the caller's fresh
logical Philox stream from its stable coordinates and preserves the established
draw sequence exactly; scheduling and thread identity never enter the ABI.

`shapeit_genotype_solve_v1` performs final maximum-probability path decoding and
applies stored missing-genotype consensus probabilities.

The opaque `shapeit_genotype_storage_v1` now owns that persistent main-iteration
state: the thresholded transition mask, ordered active indexes, accumulated
transition and missing probabilities, and storage-event count. Final solving
borrows it directly, while the graph writer receives a read-only view using the
legacy LSB-first mask encoding.

Pedigree scaffolding and the established haploid-heterozygote reset operate on
whole packed samples through checked Rust calls. `shapeit_genotype_prune_v1`
owns a complete graph-pruning round: it ranks transition mass, selects
non-adjacent merges, rebuilds ambiguity and diplotype encodings, and recomputes
the transition count. Equal probabilities and entropies use original indexes as
stable tie-breakers, so pruning does not inherit implementation-defined sort
ordering.

The whole target-haplotype refresh is likewise a single checked Rust call. It
decodes every sample's packed genotype variants directly into the haplotype-
major bitmatrix while preserving fixed homozygous and scaffolded loci between
iterations; C++ supplies only a borrowed array of packed-variant views.

`shapeit_genotype_windows_v1` maps an entire graph into HMM windows and owns the
recursive randomized splitter. It reconstructs the fresh logical Philox stream
and preserves depth-first draw order, overlapping split boundaries, and all
graph-coordinate conventions in one checked call.

## Common conditioning jobs

`shapeit_conditioning_job_build_v1` owns a worker's complete per-sample window
and conditioning-state assembly. It collects and deduplicates the full PBWT
neighbour set, applies Rust heterozygote-overlap IBD2 protection, and reproduces
the logical fallback shuffle when a window has fewer than two states. The
opaque Rust job retains the nested state vectors; C++ borrows immutable spans
for the HMM and never copies or reallocates them. Each worker rebuilds the same
opaque job in place so its scratch storage and vector capacities are reused.

## IBD2 registry

The opaque `shapeit_ibd2_tracks_v1` registry owns accumulated common-phasing
IBD2 exclusions. Rust expands new tracks by the established 4 cM rule, sorts
and collapses overlapping intervals, and exposes the live registry directly to
PBWT neighbour selection. C++ no longer stores or flattens nested track vectors.

## PBWT initialization sweep

`shapeit_pbwt_solve_chunk_v1` owns one complete PBWT initialization chunk,
including prefix replay, heterozygote and missing-genotype resolution, and PBWT
ordering/divergence updates. C++ retains only chunk scheduling and progress
reporting. Parallel chunks consume immutable prefix snapshots and write disjoint
locus rows, so logical results do not depend on worker assignment.

`shapeit_pbwt_select_sites_v1` first chooses one evaluated locus per PBWT group
from stable group-specific Philox streams. `shapeit_pbwt_select_chunk_v1` then
owns the iterative common-phasing PBWT ordering/divergence scan and IBD2-aware
neighbour search. C++ flattens its collapsed IBD2 tracks once per sweep and
schedules disjoint chunks; Rust transposes the completed neighbour slabs
directly into the haplotype-major layout borrowed by the conditioning jobs.

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
