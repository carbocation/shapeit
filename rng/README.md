# SHAPEIT RNG ABI

The `shapeit-rng` Rust static library provides SHAPEIT's versioned,
counter-based random-number kernel. It does not allocate, retain mutable global
state, or own any phasing data.

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

Beginning with ABI version 1, any incompatible change requires a new exported
ABI function and a new RNG version. Existing versioned mappings and conversion
behavior must remain available for reproducing results.

## Provenance

Philox was described by Salmon et al., *Parallel Random Numbers: As Easy as
1, 2, 3* (SC11). Constants and known-answer vectors are checked against the
permissively licensed Random123 reference implementation. The Rust code here
is an independent implementation released under SHAPEIT's MIT license.
