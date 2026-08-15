# SHAPEIT Rust ABI

The `shapeit-rng` Rust static library provides versioned low-level kernels for
SHAPEIT. It does not allocate, retain mutable global state, or own any phasing
data. The first two ABIs are the counter-based random-number kernel and the
checked subset-transpose operation used by the common-variant HMM.

## Bitmatrix subset transpose

`shapeit_bitmatrix_subset_transpose_v1` selects an arbitrary list of rows from
a row-major bitmatrix and transposes a contiguous byte span into the layout
consumed by the HMM. Rust validates dimensions, index bounds, and integer
arithmetic before writing the caller-owned output buffer. On x86-64 it performs
runtime BMI2 detection and uses a tiled `PEXT` kernel; other targets use the
portable byte-exact implementation.

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
