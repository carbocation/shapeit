use core::slice;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SolveLayout {
    first_current: usize,
    last_current: usize,
    variant_bytes: usize,
}

struct SolveValidation<'a> {
    haplotypes_length: usize,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    individual_count: usize,
    genotype_variants_length: usize,
    site_chunks: &'a [i32],
    chunk: usize,
    buffer_start: usize,
    buffer_length: usize,
    scores_length: usize,
}

fn validate_solve_layout(parameters: SolveValidation<'_>) -> Result<SolveLayout, u32> {
    let SolveValidation {
        haplotypes_length,
        haplotype_stride,
        site_count,
        haplotype_count,
        individual_count,
        genotype_variants_length,
        site_chunks,
        chunk,
        buffer_start,
        buffer_length,
        scores_length,
    } = parameters;
    if site_count == 0
        || haplotype_stride == 0
        || haplotype_count == 0
        || individual_count == 0
        || site_chunks.len() != site_count
        || buffer_start > site_count
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let chunk = i32::try_from(chunk).map_err(|_| STATUS_INTEGER_OVERFLOW)?;
    let target_haplotypes = individual_count
        .checked_mul(2)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let padded_haplotypes = haplotype_stride
        .checked_mul(8)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if target_haplotypes > haplotype_count || haplotype_count > padded_haplotypes {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let required_haplotypes = site_count
        .checked_mul(haplotype_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_haplotypes > haplotypes_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let variant_bytes = site_count
        .checked_add(1)
        .map(|value| value >> 1)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if genotype_variants_length < variant_bytes || scores_length <= site_count {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    let mut first_current = None;
    let mut last_current = 0usize;
    let mut previous = -1i32;
    for (locus, &site_chunk) in site_chunks.iter().enumerate() {
        if site_chunk < previous || site_chunk < 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        previous = site_chunk;
        if site_chunk == chunk {
            first_current.get_or_insert(locus);
            last_current = locus;
        }
    }
    let first_current = first_current.ok_or(STATUS_INVALID_DIMENSIONS)?;
    if buffer_start > first_current {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let required_buffer = first_current
        .checked_sub(buffer_start)
        .and_then(|count| count.checked_mul(haplotype_stride))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_buffer > buffer_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(SolveLayout {
        first_current,
        last_current,
        variant_bytes,
    })
}

#[inline]
fn variant_code(variants: &[u8], locus: usize) -> u8 {
    (variants[locus >> 1] >> ((locus & 1) << 2)) & 3
}

#[inline]
unsafe fn matrix_get(haplotypes: *const u8, stride: usize, locus: usize, haplotype: usize) -> bool {
    ((*haplotypes.add(locus * stride + (haplotype >> 3)) >> (7 - (haplotype & 7))) & 1) != 0
}

#[inline]
unsafe fn matrix_set(
    haplotypes: *mut u8,
    stride: usize,
    locus: usize,
    haplotype: usize,
    allele: bool,
) {
    let target = haplotypes.add(locus * stride + (haplotype >> 3));
    let shift = 7 - (haplotype & 7);
    let mask = 1u8 << shift;
    *target = (*target & !mask) | (u8::from(allele) << shift);
}

#[inline]
fn weighted_score(value: i32, score: f32) -> f64 {
    f64::from((value as f32) * score)
}

struct SolveParameters<'a> {
    haplotypes: *mut u8,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    genotype_variants: &'a [&'a [u8]],
    site_chunks: &'a [i32],
    chunk: i32,
    buffer_start: usize,
    buffer: &'a [u8],
    scores: &'a [f32],
}

unsafe fn solve_chunk(parameters: SolveParameters<'_>) {
    let SolveParameters {
        haplotypes,
        haplotype_stride,
        site_count,
        haplotype_count,
        genotype_variants,
        site_chunks,
        chunk,
        buffer_start,
        buffer,
        scores,
    } = parameters;
    let individual_count = genotype_variants.len();
    let mut ordering: Vec<usize> = (0..haplotype_count).collect();
    let mut alternate = vec![0usize; haplotype_count];
    let mut divergence = vec![0usize; haplotype_count];
    let mut alternate_divergence = vec![0usize; haplotype_count];
    let mut ranks = vec![0usize; haplotype_count];
    let mut alleles = vec![0i32; haplotype_count];
    let mut heterozygous = vec![false; individual_count];
    let mut missing = vec![false; individual_count];
    let mut ambiguous = vec![false; individual_count];

    for locus in 0..site_count {
        let current = site_chunks[locus] == chunk;
        let buffered = site_chunks[locus] < chunk && locus >= buffer_start;

        if current && locus != 0 {
            for (haplotype, allele) in alleles.iter_mut().enumerate() {
                *allele = if matrix_get(haplotypes, haplotype_stride, locus, haplotype) {
                    1
                } else {
                    -1
                };
            }
            let mut unresolved_hets = 0usize;
            for individual in 0..individual_count {
                let code = variant_code(genotype_variants[individual], locus);
                missing[individual] = code == 1;
                heterozygous[individual] = code == 2;
                ambiguous[individual] = missing[individual] || heterozygous[individual];
                if ambiguous[individual] {
                    alleles[2 * individual] = 0;
                    alleles[2 * individual + 1] = 0;
                }
                unresolved_hets += usize::from(heterozygous[individual]);
            }

            let mut unresolved_missing = 0usize;
            let mut threshold = 2.5f64;
            while unresolved_hets != 0 && threshold > 1.0 {
                let old_unresolved_hets = unresolved_hets;
                unresolved_hets = 0;
                unresolved_missing = 0;
                for individual in 0..individual_count {
                    if !ambiguous[individual] {
                        continue;
                    }
                    let haplotype0 = 2 * individual;
                    let haplotype1 = haplotype0 + 1;
                    if heterozygous[individual] {
                        let mut score = 0.0f64;
                        if ranks[haplotype0] > 0 {
                            score += f64::from(alleles[ordering[ranks[haplotype0] - 1]]);
                        }
                        if ranks[haplotype0] + 1 < haplotype_count {
                            score += f64::from(alleles[ordering[ranks[haplotype0] + 1]]);
                        }
                        if ranks[haplotype1] > 0 {
                            score -= f64::from(alleles[ordering[ranks[haplotype1] - 1]]);
                        }
                        if ranks[haplotype1] + 1 < haplotype_count {
                            score -= f64::from(alleles[ordering[ranks[haplotype1] + 1]]);
                        }
                        if score > threshold {
                            alleles[haplotype0] = 1;
                            alleles[haplotype1] = -1;
                            ambiguous[individual] = false;
                        } else if score < -threshold {
                            alleles[haplotype0] = -1;
                            alleles[haplotype1] = 1;
                            ambiguous[individual] = false;
                        } else {
                            unresolved_hets += 1;
                        }
                    }
                    if missing[individual] {
                        let mut score0 = 0.0f64;
                        let mut score1 = 0.0f64;
                        if ranks[haplotype0] > 0 {
                            score0 = f64::from(alleles[ordering[ranks[haplotype0] - 1]]);
                        }
                        if ranks[haplotype0] + 1 < haplotype_count {
                            score0 += f64::from(alleles[ordering[ranks[haplotype0] + 1]]);
                        }
                        if ranks[haplotype1] > 0 {
                            score1 = f64::from(alleles[ordering[ranks[haplotype1] - 1]]);
                        }
                        if ranks[haplotype1] + 1 < haplotype_count {
                            score1 += f64::from(alleles[ordering[ranks[haplotype1] + 1]]);
                        }
                        match (score0 as i32, score1 as i32) {
                            (-2, -2) => {
                                alleles[haplotype0] = -1;
                                alleles[haplotype1] = -1;
                                ambiguous[individual] = false;
                            }
                            (-2, 2) => {
                                alleles[haplotype0] = -1;
                                alleles[haplotype1] = 1;
                                ambiguous[individual] = false;
                            }
                            (2, -2) => {
                                alleles[haplotype0] = 1;
                                alleles[haplotype1] = -1;
                                ambiguous[individual] = false;
                            }
                            (2, 2) => {
                                alleles[haplotype0] = 1;
                                alleles[haplotype1] = 1;
                                ambiguous[individual] = false;
                            }
                            _ => unresolved_missing += 1,
                        }
                    }
                }
                if unresolved_hets == old_unresolved_hets {
                    threshold -= 1.0;
                }
            }

            if unresolved_hets != 0 || unresolved_missing != 0 {
                for individual in 0..individual_count {
                    if !ambiguous[individual] {
                        continue;
                    }
                    let haplotype0 = 2 * individual;
                    let haplotype1 = haplotype0 + 1;
                    if heterozygous[individual] {
                        let mut score = 0.0f64;
                        if ranks[haplotype0] > 0 {
                            score += weighted_score(
                                alleles[ordering[ranks[haplotype0] - 1]],
                                scores[locus - divergence[ranks[haplotype0]] + 1],
                            );
                        }
                        if ranks[haplotype0] + 1 < haplotype_count {
                            score += weighted_score(
                                alleles[ordering[ranks[haplotype0] + 1]],
                                scores[locus - divergence[ranks[haplotype0] + 1] + 1],
                            );
                        }
                        if ranks[haplotype1] > 0 {
                            score -= weighted_score(
                                alleles[ordering[ranks[haplotype1] - 1]],
                                scores[locus - divergence[ranks[haplotype1]] + 1],
                            );
                        }
                        if ranks[haplotype1] + 1 < haplotype_count {
                            score -= weighted_score(
                                alleles[ordering[ranks[haplotype1] + 1]],
                                scores[locus - divergence[ranks[haplotype1] + 1] + 1],
                            );
                        }
                        if score > 0.0 {
                            alleles[haplotype0] = 1;
                            alleles[haplotype1] = -1;
                        } else {
                            alleles[haplotype0] = -1;
                            alleles[haplotype1] = 1;
                        }
                    }
                    if missing[individual] {
                        let mut score0 = 0.0f64;
                        let mut score1 = 0.0f64;
                        if ranks[haplotype0] > 0 {
                            score0 = weighted_score(
                                alleles[ordering[ranks[haplotype0] - 1]],
                                scores[locus - divergence[ranks[haplotype0]] + 1],
                            );
                        }
                        if ranks[haplotype0] + 1 < haplotype_count {
                            score0 += weighted_score(
                                alleles[ordering[ranks[haplotype0] + 1]],
                                scores[locus - divergence[ranks[haplotype0] + 1] + 1],
                            );
                        }
                        if ranks[haplotype1] > 0 {
                            score1 = weighted_score(
                                alleles[ordering[ranks[haplotype1] - 1]],
                                scores[locus - divergence[ranks[haplotype1]] + 1],
                            );
                        }
                        if ranks[haplotype1] + 1 < haplotype_count {
                            score1 += weighted_score(
                                alleles[ordering[ranks[haplotype1] + 1]],
                                scores[locus - divergence[ranks[haplotype1] + 1] + 1],
                            );
                        }
                        alleles[haplotype0] = if score0 > 0.0 { 1 } else { -1 };
                        alleles[haplotype1] = if score1 > 0.0 { 1 } else { -1 };
                    }
                }
            }

            for individual in 0..individual_count {
                if heterozygous[individual] || missing[individual] {
                    matrix_set(
                        haplotypes,
                        haplotype_stride,
                        locus,
                        2 * individual,
                        alleles[2 * individual] > 0,
                    );
                    matrix_set(
                        haplotypes,
                        haplotype_stride,
                        locus,
                        2 * individual + 1,
                        alleles[2 * individual + 1] > 0,
                    );
                }
            }
        }

        if current || buffered {
            let mut zero_count = 0usize;
            let mut one_count = 0usize;
            let mut zero_divergence = locus;
            let mut one_divergence = locus;
            for haplotype in 0..haplotype_count {
                let ordered_haplotype = ordering[haplotype];
                let prior_divergence = divergence[haplotype];
                zero_divergence = zero_divergence.max(prior_divergence);
                one_divergence = one_divergence.max(prior_divergence);
                let allele = if buffered {
                    let address =
                        (locus - buffer_start) * haplotype_stride + (ordered_haplotype >> 3);
                    ((buffer[address] >> (7 - (ordered_haplotype & 7))) & 1) != 0
                } else {
                    matrix_get(haplotypes, haplotype_stride, locus, ordered_haplotype)
                };
                if allele {
                    alternate[one_count] = ordered_haplotype;
                    alternate_divergence[one_count] = one_divergence;
                    one_divergence = 0;
                    one_count += 1;
                } else {
                    ordering[zero_count] = ordered_haplotype;
                    divergence[zero_count] = zero_divergence;
                    zero_divergence = 0;
                    zero_count += 1;
                }
            }
            ordering[zero_count..zero_count + one_count].copy_from_slice(&alternate[..one_count]);
            divergence[zero_count..zero_count + one_count]
                .copy_from_slice(&alternate_divergence[..one_count]);
            for (rank, &haplotype) in ordering.iter().enumerate() {
                ranks[haplotype] = rank;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn shapeit_pbwt_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Replay and solve one independently writable PBWT initialization chunk.
///
/// Concurrent calls may share inputs and `haplotypes` only when their chunk
/// mappings write disjoint locus rows and each call has its own immutable prefix
/// buffer.
///
/// # Safety
///
/// Every buffer and pointer array must be valid for its stated length. Genotype,
/// chunk, prefix, and score inputs must remain immutable for the duration of the
/// call. Invalid layouts are rejected before the haplotype matrix is modified.
pub unsafe extern "C" fn shapeit_pbwt_solve_chunk_v1(
    haplotypes: *mut u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    genotype_variants: *const *const u8,
    individual_count: usize,
    genotype_variants_length: usize,
    site_chunks: *const i32,
    site_chunks_length: usize,
    chunk: usize,
    buffer_start: usize,
    buffer: *const u8,
    buffer_length: usize,
    scores: *const f32,
    scores_length: usize,
) -> u32 {
    if haplotypes.is_null()
        || genotype_variants.is_null()
        || site_chunks.is_null()
        || scores.is_null()
        || (buffer_length != 0 && buffer.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let site_chunks = slice::from_raw_parts(site_chunks, site_chunks_length);
    let layout = match validate_solve_layout(SolveValidation {
        haplotypes_length,
        haplotype_stride,
        site_count,
        haplotype_count,
        individual_count,
        genotype_variants_length,
        site_chunks,
        chunk,
        buffer_start,
        buffer_length,
        scores_length,
    }) {
        Ok(value) => value,
        Err(status) => return status,
    };
    debug_assert!(layout.first_current <= layout.last_current);
    let pointers = slice::from_raw_parts(genotype_variants, individual_count);
    if pointers.iter().any(|pointer| pointer.is_null()) {
        return STATUS_NULL_POINTER;
    }
    let genotype_variants: Vec<&[u8]> = pointers
        .iter()
        .map(|&pointer| slice::from_raw_parts(pointer, layout.variant_bytes))
        .collect();
    let buffer = if buffer_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(buffer, buffer_length)
    };
    let scores = slice::from_raw_parts(scores, scores_length);
    solve_chunk(SolveParameters {
        haplotypes,
        haplotype_stride,
        site_count,
        haplotype_count,
        genotype_variants: &genotype_variants,
        site_chunks,
        chunk: chunk as i32,
        buffer_start,
        buffer,
        scores,
    });
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_layout_rejects_non_monotone_chunks() {
        assert_eq!(
            validate_solve_layout(SolveValidation {
                haplotypes_length: 4,
                haplotype_stride: 1,
                site_count: 4,
                haplotype_count: 2,
                individual_count: 1,
                genotype_variants_length: 2,
                site_chunks: &[0, 1, 0, 1],
                chunk: 1,
                buffer_start: 0,
                buffer_length: 1,
                scores_length: 5,
            }),
            Err(STATUS_INVALID_DIMENSIONS)
        );
    }

    #[test]
    fn solve_chunk_preserves_simple_heterozygote_and_clears_unresolved_missing() {
        let mut haplotypes = [0b0100_0000u8, 0, 0];
        let genotype = [0x20u8, 0x01];
        let genotypes: [&[u8]; 1] = [&genotype];
        let chunks = [0, 0, 0];
        let scores = [0.0, 0.693_147_2, 1.098_612_3, 1.386_294_4];
        unsafe {
            solve_chunk(SolveParameters {
                haplotypes: haplotypes.as_mut_ptr(),
                haplotype_stride: 1,
                site_count: 3,
                haplotype_count: 2,
                genotype_variants: &genotypes,
                site_chunks: &chunks,
                chunk: 0,
                buffer_start: 0,
                buffer: &[],
                scores: &scores,
            });
        }
        assert_eq!(haplotypes[1] & 0xc0, 0x40);
        assert_eq!(haplotypes[2] & 0xc0, 0x00);
    }
}
