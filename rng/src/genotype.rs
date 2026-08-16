use core::slice;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;

const MAX_AMBIGUOUS_PER_SEGMENT: usize = 22;
const MASK_INIT: u64 = u64::MAX;
const MASK_SCAFFOLD: u64 = 0x00aa_00aa_00aa_00aa;
const MASK_UNFOLD0: u64 = 0x55aa_55aa_55aa_55aa;
const MASK_UNFOLD1: u64 = 0x3333_cccc_3333_cccc;
const MASK_UNFOLD2: u64 = 0x0f0f_0f0f_f0f0_f0f0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphSizes {
    segments: usize,
    ambiguous: usize,
    missing: usize,
}

#[inline]
fn variant_nibble(variants: &[u8], locus: usize) -> u8 {
    (variants[locus >> 1] >> ((locus & 1) << 2)) & 0x0f
}

#[inline]
fn graph_code(variant: u8) -> u8 {
    variant & 3
}

fn required_variant_bytes(variant_count: usize) -> Result<usize, u32> {
    variant_count
        .checked_add(1)
        .map(|value| value >> 1)
        .ok_or(STATUS_INTEGER_OVERFLOW)
}

fn graph_sizes(variants: &[u8], variant_count: usize) -> GraphSizes {
    let mut relative_unfolded = 0usize;
    let mut relative_variants = 0usize;
    let mut relative_scaffolded = false;
    let mut relative_ambiguous = 0usize;
    let mut segments = 1usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    let mut locus = 0usize;

    while locus < variant_count {
        let code = graph_code(variant_nibble(variants, locus));
        let heterozygous = code == 2;
        let scaffolded = code == 3;
        let predicted_unfolded = relative_unfolded
            + usize::from(heterozygous)
            + usize::from(relative_scaffolded || scaffolded);
        if predicted_unfolded == 4
            || relative_variants == u16::MAX as usize
            || relative_ambiguous == MAX_AMBIGUOUS_PER_SEGMENT
        {
            relative_unfolded = 0;
            relative_variants = 0;
            relative_scaffolded = false;
            relative_ambiguous = 0;
            segments += 1;
            continue;
        }
        relative_unfolded += usize::from(heterozygous);
        relative_scaffolded |= scaffolded;
        relative_ambiguous += usize::from(heterozygous || scaffolded);
        ambiguous += usize::from(heterozygous || scaffolded);
        missing += usize::from(code == 1);
        relative_variants += 1;
        locus += 1;
    }
    GraphSizes {
        segments,
        ambiguous,
        missing,
    }
}

fn build_graph(
    variants: &[u8],
    variant_count: usize,
    segment_lengths: &mut [u16],
    ambiguous: &mut [u8],
    diplotypes: &mut [u64],
) -> u32 {
    let mut ordered_segments = vec![false; segment_lengths.len()];
    let mut relative_unfolded = 0usize;
    let mut relative_variants = 0usize;
    let mut relative_scaffolded = false;
    let mut relative_ambiguous = 0usize;
    let mut segment = 0usize;
    let mut locus = 0usize;

    while locus < variant_count {
        let code = graph_code(variant_nibble(variants, locus));
        let heterozygous = code == 2;
        let scaffolded = code == 3;
        let predicted_unfolded = relative_unfolded
            + usize::from(heterozygous)
            + usize::from(relative_scaffolded || scaffolded);
        if predicted_unfolded == 4
            || relative_variants == u16::MAX as usize
            || relative_ambiguous == MAX_AMBIGUOUS_PER_SEGMENT
        {
            segment_lengths[segment] = relative_variants as u16;
            ordered_segments[segment] = relative_scaffolded;
            relative_unfolded = 0;
            relative_variants = 0;
            relative_scaffolded = false;
            relative_ambiguous = 0;
            segment += 1;
            continue;
        }
        relative_unfolded += usize::from(heterozygous);
        relative_scaffolded |= scaffolded;
        relative_ambiguous += usize::from(heterozygous || scaffolded);
        relative_variants += 1;
        locus += 1;
    }
    segment_lengths[segment] = relative_variants as u16;
    ordered_segments[segment] = relative_scaffolded;

    let mut absolute_locus = 0usize;
    let mut absolute_ambiguous = 0usize;
    let mut transitions = 0u32;
    let mut previous_diplotypes = 1u32;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let mut unfolded = usize::from(ordered_segments[segment]);
        let mut diplotype = if ordered_segments[segment] {
            MASK_SCAFFOLD
        } else {
            MASK_INIT
        };
        for relative_locus in 0..usize::from(length) {
            let variant = variant_nibble(variants, absolute_locus + relative_locus);
            match graph_code(variant) {
                2 => {
                    ambiguous[absolute_ambiguous] = match unfolded {
                        0 => 0xaa,
                        1 => 0xcc,
                        2 => 0xf0,
                        _ => unreachable!("segment contains too many unfolded genotypes"),
                    };
                    diplotype &= match unfolded {
                        0 => MASK_UNFOLD0,
                        1 => MASK_UNFOLD1,
                        2 => MASK_UNFOLD2,
                        _ => unreachable!("segment contains too many unfolded genotypes"),
                    };
                    unfolded += 1;
                    absolute_ambiguous += 1;
                }
                3 => {
                    let haplotype0 = (variant & 4) != 0;
                    let haplotype1 = (variant & 8) != 0;
                    let mut code = 0u8;
                    for haplotype in 0..8 {
                        let allele = if haplotype & 1 == 0 {
                            haplotype0
                        } else {
                            haplotype1
                        };
                        code |= u8::from(allele) << haplotype;
                    }
                    ambiguous[absolute_ambiguous] = code;
                    absolute_ambiguous += 1;
                }
                _ => {}
            }
        }
        diplotypes[segment] = diplotype;
        let current_diplotypes = diplotype.count_ones();
        transitions =
            transitions.wrapping_add(previous_diplotypes.wrapping_mul(current_diplotypes));
        previous_diplotypes = current_diplotypes;
        absolute_locus += usize::from(length);
    }
    transitions
}

#[no_mangle]
pub extern "C" fn shapeit_genotype_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Return the exact output sizes for one packed genotype graph.
///
/// # Safety
///
/// `variants` must be readable for `variants_length` bytes when `variant_count`
/// is nonzero. All three output pointers must be writable.
pub unsafe extern "C" fn shapeit_genotype_graph_sizes_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    segment_count: *mut usize,
    ambiguous_count: *mut usize,
    missing_count: *mut usize,
) -> u32 {
    if segment_count.is_null() || ambiguous_count.is_null() || missing_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if required == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, required)
    };
    let sizes = graph_sizes(variants, variant_count);
    *segment_count = sizes.segments;
    *ambiguous_count = sizes.ambiguous;
    *missing_count = sizes.missing;
    STATUS_OK
}

#[no_mangle]
/// Build one complete common-phasing genotype graph.
///
/// # Safety
///
/// Input and output buffers must be valid for their stated lengths and must not
/// overlap. Output sizes must match `shapeit_genotype_graph_sizes_v1`.
pub unsafe extern "C" fn shapeit_genotype_graph_build_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    segment_lengths: *mut u16,
    segment_lengths_length: usize,
    ambiguous: *mut u8,
    ambiguous_length: usize,
    diplotypes: *mut u64,
    diplotypes_length: usize,
    transition_count: *mut u32,
) -> u32 {
    if transition_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if required == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, required)
    };
    let sizes = graph_sizes(variants, variant_count);
    if segment_lengths_length != sizes.segments
        || ambiguous_length != sizes.ambiguous
        || diplotypes_length != sizes.segments
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if segment_lengths.is_null()
        || diplotypes.is_null()
        || (ambiguous_length != 0 && ambiguous.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let segment_lengths = slice::from_raw_parts_mut(segment_lengths, segment_lengths_length);
    let ambiguous = if ambiguous_length == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts_mut(diplotypes, diplotypes_length);
    *transition_count = build_graph(
        variants,
        variant_count,
        segment_lengths,
        ambiguous,
        diplotypes,
    );
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(variants: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; (variants.len() + 1) >> 1];
        for (locus, &variant) in variants.iter().enumerate() {
            packed[locus >> 1] |= variant << ((locus & 1) << 2);
        }
        packed
    }

    fn build(variants: &[u8]) -> (GraphSizes, Vec<u16>, Vec<u8>, Vec<u64>, u32) {
        let packed = pack(variants);
        let sizes = graph_sizes(&packed, variants.len());
        let mut lengths = vec![0u16; sizes.segments];
        let mut ambiguous = vec![0u8; sizes.ambiguous];
        let mut diplotypes = vec![0u64; sizes.segments];
        let transitions = build_graph(
            &packed,
            variants.len(),
            &mut lengths,
            &mut ambiguous,
            &mut diplotypes,
        );
        (sizes, lengths, ambiguous, diplotypes, transitions)
    }

    #[test]
    fn graph_builder_preserves_unfolding_and_segment_limits() {
        let variants = [2, 2, 2, 2, 1, 0];
        let (sizes, lengths, ambiguous, diplotypes, transitions) = build(&variants);
        assert_eq!(
            sizes,
            GraphSizes {
                segments: 2,
                ambiguous: 4,
                missing: 1
            }
        );
        assert_eq!(lengths, [3, 3]);
        assert_eq!(ambiguous, [0xaa, 0xcc, 0xf0, 0xaa]);
        let expected0 = MASK_INIT & MASK_UNFOLD0 & MASK_UNFOLD1 & MASK_UNFOLD2;
        let expected1 = MASK_INIT & MASK_UNFOLD0;
        assert_eq!(diplotypes, [expected0, expected1]);
        assert_eq!(
            transitions,
            expected0.count_ones() + expected0.count_ones() * expected1.count_ones()
        );
    }

    #[test]
    fn graph_builder_preserves_scaffold_order_and_alleles() {
        let variants = [3 | 4, 2, 2, 2];
        let (sizes, lengths, ambiguous, diplotypes, _) = build(&variants);
        assert_eq!(
            sizes,
            GraphSizes {
                segments: 2,
                ambiguous: 4,
                missing: 0
            }
        );
        assert_eq!(lengths, [3, 1]);
        assert_eq!(ambiguous, [0x55, 0xcc, 0xf0, 0xaa]);
        assert_eq!(
            diplotypes,
            [
                MASK_SCAFFOLD & MASK_UNFOLD1 & MASK_UNFOLD2,
                MASK_INIT & MASK_UNFOLD0
            ]
        );
    }

    #[test]
    fn graph_abi_rejects_wrong_output_sizes_without_writing() {
        let variants = pack(&[2, 0]);
        let mut lengths = [0xa5a5u16; 2];
        let mut ambiguous = [0xa5u8; 1];
        let mut diplotypes = [0xa5a5_a5a5_a5a5_a5a5u64; 2];
        let mut transitions = 0xa5a5_a5a5u32;
        let status = unsafe {
            shapeit_genotype_graph_build_v1(
                variants.as_ptr(),
                variants.len(),
                2,
                lengths.as_mut_ptr(),
                lengths.len(),
                ambiguous.as_mut_ptr(),
                ambiguous.len(),
                diplotypes.as_mut_ptr(),
                diplotypes.len(),
                &mut transitions,
            )
        };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(lengths, [0xa5a5; 2]);
        assert_eq!(ambiguous, [0xa5]);
        assert_eq!(diplotypes, [0xa5a5_a5a5_a5a5_a5a5; 2]);
        assert_eq!(transitions, 0xa5a5_a5a5);
    }
}
