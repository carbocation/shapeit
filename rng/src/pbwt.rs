use core::slice;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

use crate::bitmatrix::shapeit_bitmatrix_transpose_v1;
use crate::genotype::{GenotypeGraphV1, LogicalRng};
use crate::ibd2::Ibd2TracksV1;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;
const STATUS_INSUFFICIENT_STATES: u32 = 5;
const STATUS_THREAD_FAILURE: u32 = 6;

pub type PbwtProgressV1 = unsafe extern "C" fn(usize, usize, *mut c_void);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PbwtBatchResultV1 {
    completed: usize,
    failed_chunk: usize,
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectLayout {
    target_haplotype_count: usize,
    last_current: usize,
    neighbor_slab: usize,
}

struct SelectValidation<'a> {
    haplotypes_length: usize,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    target_individual_count: usize,
    evaluated_sites: &'a [u8],
    selected_sites: &'a [u8],
    site_groups: &'a [i32],
    group_count: usize,
    site_chunks: &'a [i32],
    chunk: usize,
    buffer_start: usize,
    depth: usize,
    ibd2: &'a Ibd2TracksV1,
    neighbors_length: usize,
}

fn validate_select_layout(parameters: SelectValidation<'_>) -> Result<SelectLayout, u32> {
    let SelectValidation {
        haplotypes_length,
        haplotype_stride,
        site_count,
        haplotype_count,
        target_individual_count,
        evaluated_sites,
        selected_sites,
        site_groups,
        group_count,
        site_chunks,
        chunk,
        buffer_start,
        depth,
        ibd2,
        neighbors_length,
    } = parameters;
    if site_count == 0
        || haplotype_count == 0
        || target_individual_count == 0
        || haplotype_stride == 0
        || group_count == 0
        || depth == 0
        || evaluated_sites.len() != site_count
        || selected_sites.len() != site_count
        || site_groups.len() != site_count
        || site_chunks.len() != site_count
        || ibd2.individual_count() != target_individual_count
        || !ibd2.is_collapsed()
        || buffer_start >= site_count
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    if evaluated_sites.iter().any(|&value| value > 1)
        || selected_sites.iter().any(|&value| value > 1)
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let target_haplotype_count = target_individual_count
        .checked_mul(2)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let padded_haplotypes = haplotype_stride
        .checked_mul(8)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if target_haplotype_count > haplotype_count
        || haplotype_count > padded_haplotypes
        || haplotype_count > i32::MAX as usize
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let required_haplotypes = site_count
        .checked_mul(haplotype_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_haplotypes > haplotypes_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    for &group in site_groups {
        if group < 0 || group as usize >= group_count {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
    }

    let chunk = i32::try_from(chunk).map_err(|_| STATUS_INTEGER_OVERFLOW)?;
    let mut first_current = None;
    let mut last_current = 0usize;
    let mut previous_chunk = -1i32;
    for (locus, &site_chunk) in site_chunks.iter().enumerate() {
        if site_chunk < 0 || site_chunk < previous_chunk {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        previous_chunk = site_chunk;
        if site_chunk == chunk {
            first_current.get_or_insert(locus);
            last_current = locus;
        }
    }
    let first_current = first_current.ok_or(STATUS_INVALID_DIMENSIONS)?;
    if buffer_start > first_current {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let neighbor_slab = group_count
        .checked_mul(target_haplotype_count)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let required_neighbors = depth
        .checked_add(1)
        .and_then(|slabs| slabs.checked_mul(neighbor_slab))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_neighbors > neighbors_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(SelectLayout {
        target_haplotype_count,
        last_current,
        neighbor_slab,
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

struct SelectParameters<'a> {
    haplotypes: *const u8,
    haplotype_stride: usize,
    haplotype_count: usize,
    target_haplotype_count: usize,
    evaluated_sites: &'a [u8],
    selected_sites: &'a [u8],
    site_groups: &'a [i32],
    site_chunks: &'a [i32],
    chunk: i32,
    buffer_start: usize,
    last_current: usize,
    depth: usize,
    ibd2: &'a Ibd2TracksV1,
    neighbors: *mut i32,
    neighbor_slab: usize,
}

unsafe fn select_chunk(parameters: SelectParameters<'_>) -> Result<(), u32> {
    let SelectParameters {
        haplotypes,
        haplotype_stride,
        haplotype_count,
        target_haplotype_count,
        evaluated_sites,
        selected_sites,
        site_groups,
        site_chunks,
        chunk,
        buffer_start,
        last_current,
        depth,
        ibd2,
        neighbors,
        neighbor_slab,
    } = parameters;
    let mut ordering: Vec<usize> = (0..haplotype_count).collect();
    let mut alternate = vec![0usize; haplotype_count];
    let mut divergence = vec![0usize; haplotype_count];
    let mut alternate_divergence = vec![0usize; haplotype_count];

    for locus in buffer_start..=last_current {
        let current = site_chunks[locus] == chunk;
        let buffered = site_chunks[locus] < chunk;
        if evaluated_sites[locus] == 0 || (!current && !buffered) {
            continue;
        }

        let mut zero_count = 0usize;
        let mut one_count = 0usize;
        let mut zero_divergence = locus;
        let mut one_divergence = locus;
        for rank in 0..haplotype_count {
            let ordered_haplotype = ordering[rank];
            let prior_divergence = divergence[rank];
            zero_divergence = zero_divergence.max(prior_divergence);
            one_divergence = one_divergence.max(prior_divergence);
            if matrix_get(haplotypes, haplotype_stride, locus, ordered_haplotype) {
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

        if selected_sites[locus] == 0 || !current {
            continue;
        }
        let group = site_groups[locus] as usize;
        for rank in 0..haplotype_count {
            let target_haplotype = ordering[rank];
            if target_haplotype >= target_haplotype_count {
                continue;
            }
            let mut left_offset = 1usize;
            let mut right_offset = 1usize;
            let mut left_divergence = None;
            let mut right_divergence = None;
            let mut added = 0usize;
            while added < depth {
                let left = rank.checked_sub(left_offset).map(|left_rank| {
                    left_divergence =
                        Some(left_divergence.unwrap_or(0).max(divergence[left_rank + 1]));
                    (ordering[left_rank], left_divergence.unwrap())
                });
                let right_rank = rank.checked_add(right_offset);
                let right = right_rank
                    .filter(|&value| value < haplotype_count)
                    .map(|value| {
                        right_divergence =
                            Some(right_divergence.unwrap_or(0).max(divergence[value]));
                        (ordering[value], right_divergence.unwrap())
                    });
                let allowed_left =
                    left.filter(|&(candidate, _)| ibd2.allows(target_haplotype, candidate, locus));
                let allowed_right =
                    right.filter(|&(candidate, _)| ibd2.allows(target_haplotype, candidate, locus));

                let candidate = match (allowed_left, allowed_right) {
                    (Some(left), Some(right)) => {
                        if left.1 < right.1 {
                            left_offset += 1;
                            left.0
                        } else {
                            right_offset += 1;
                            right.0
                        }
                    }
                    (Some(left), None) => {
                        left_offset += 1;
                        left.0
                    }
                    (None, Some(right)) => {
                        right_offset += 1;
                        right.0
                    }
                    (None, None) => {
                        left_offset = left_offset.checked_add(1).ok_or(STATUS_INTEGER_OVERFLOW)?;
                        right_offset =
                            right_offset.checked_add(1).ok_or(STATUS_INTEGER_OVERFLOW)?;
                        if left.is_none() && right.is_none() {
                            return Err(STATUS_INSUFFICIENT_STATES);
                        }
                        continue;
                    }
                };
                let output =
                    added * neighbor_slab + group * target_haplotype_count + target_haplotype;
                *neighbors.add(output) = candidate as i32;
                added += 1;
            }
        }
    }
    Ok(())
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
/// Select exactly one evaluated locus per PBWT group from logical RNG streams.
///
/// Each group uses `(seed, domain, iteration, group)` as its complete stream
/// coordinate, so selection is independent of thread scheduling.
///
/// # Safety
///
/// Input and output buffers must be valid for their stated lengths and must not
/// overlap. Invalid layouts are rejected before `selected_sites` is modified.
pub unsafe extern "C" fn shapeit_pbwt_select_sites_v1(
    evaluated_sites: *const u8,
    evaluated_sites_length: usize,
    site_groups: *const i32,
    site_groups_length: usize,
    group_count: usize,
    seed: u64,
    domain: u32,
    iteration: u32,
    selected_sites: *mut u8,
    selected_sites_length: usize,
) -> u32 {
    if evaluated_sites.is_null() || site_groups.is_null() || selected_sites.is_null() {
        return STATUS_NULL_POINTER;
    }
    if evaluated_sites_length == 0
        || group_count == 0
        || site_groups_length != evaluated_sites_length
        || selected_sites_length != evaluated_sites_length
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let evaluated_sites = slice::from_raw_parts(evaluated_sites, evaluated_sites_length);
    let site_groups = slice::from_raw_parts(site_groups, site_groups_length);
    if evaluated_sites.iter().any(|&value| value > 1) {
        return STATUS_INVALID_DIMENSIONS;
    }
    let mut counts = vec![0u32; group_count];
    for (locus, &group) in site_groups.iter().enumerate() {
        if group < 0 || group as usize >= group_count {
            return STATUS_OUT_OF_BOUNDS;
        }
        if evaluated_sites[locus] != 0 {
            counts[group as usize] = match counts[group as usize].checked_add(1) {
                Some(value) => value,
                None => return STATUS_INTEGER_OVERFLOW,
            };
        }
    }
    let mut choices = vec![0u32; group_count];
    for (group, (&count, choice)) in counts.iter().zip(choices.iter_mut()).enumerate() {
        if count != 0 {
            let item = match u64::try_from(group) {
                Ok(value) => value,
                Err(_) => return STATUS_INTEGER_OVERFLOW,
            };
            *choice = LogicalRng::new(seed, domain, iteration, item).next_bounded(count);
        }
    }
    let selected_sites = slice::from_raw_parts_mut(selected_sites, selected_sites_length);
    selected_sites.fill(0);
    let mut observed = vec![0u32; group_count];
    for (locus, &group) in site_groups.iter().enumerate() {
        if evaluated_sites[locus] == 0 {
            continue;
        }
        let group = group as usize;
        if observed[group] == choices[group] {
            selected_sites[locus] = 1;
        }
        observed[group] += 1;
    }
    STATUS_OK
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

struct SyncVariantViews(Vec<*const u8>);

// Graph variant allocations remain live and immutable throughout the initial
// PBWT sweep.
unsafe impl Sync for SyncVariantViews {}

struct SolveAllShared<'a> {
    variant_major_address: usize,
    variant_major_length: usize,
    variant_major_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    variant_views: &'a SyncVariantViews,
    variants_length: usize,
    site_chunks: &'a [i32],
    chunk_starts: &'a [i32],
    buffers: &'a [Vec<u8>],
    scores: &'a [f32],
    progress: Option<PbwtProgressV1>,
    progress_context_address: usize,
    next_chunk: AtomicUsize,
    completed: AtomicUsize,
    status: AtomicU32,
    failed_chunk: AtomicUsize,
    serialized_progress: Mutex<()>,
}

fn record_solve_all_failure(shared: &SolveAllShared<'_>, status: u32, chunk: usize) {
    if shared
        .status
        .compare_exchange(STATUS_OK, status, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        shared.failed_chunk.store(chunk, Ordering::SeqCst);
    }
}

fn run_solve_all_worker(shared: &SolveAllShared<'_>) {
    loop {
        if shared.status.load(Ordering::SeqCst) != STATUS_OK {
            break;
        }
        let chunk = shared.next_chunk.fetch_add(1, Ordering::SeqCst);
        if chunk >= shared.chunk_starts.len() {
            break;
        }
        let buffer_start = match usize::try_from(shared.chunk_starts[chunk]) {
            Ok(value) => value,
            Err(_) => {
                record_solve_all_failure(shared, STATUS_OUT_OF_BOUNDS, chunk);
                break;
            }
        };
        let buffer = &shared.buffers[chunk];
        let status = unsafe {
            shapeit_pbwt_solve_chunk_v1(
                shared.variant_major_address as *mut u8,
                shared.variant_major_length,
                shared.variant_major_stride,
                shared.site_count,
                shared.haplotype_count,
                shared.variant_views.0.as_ptr(),
                shared.variant_views.0.len(),
                shared.variants_length,
                shared.site_chunks.as_ptr(),
                shared.site_chunks.len(),
                chunk,
                buffer_start,
                buffer.as_ptr(),
                buffer.len(),
                shared.scores.as_ptr(),
                shared.scores.len(),
            )
        };
        if status != STATUS_OK {
            record_solve_all_failure(shared, status, chunk);
            break;
        }
        let _guard = match shared.serialized_progress.lock() {
            Ok(value) => value,
            Err(_) => {
                record_solve_all_failure(shared, STATUS_THREAD_FAILURE, chunk);
                break;
            }
        };
        let completed = shared.completed.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(progress) = shared.progress {
            unsafe {
                progress(
                    completed,
                    shared.chunk_starts.len(),
                    shared.progress_context_address as *mut c_void,
                );
            }
        }
    }
}

#[no_mangle]
/// Run and transpose the complete initial PBWT phasing sweep on scoped Rust
/// worker threads. Immutable prefix snapshots are created before any chunk
/// writes begin.
///
/// # Safety
///
/// Graph pointers and all input buffers must remain live and immutable until
/// return. Graphs must be distinct. Variant-major and haplotype-major matrices
/// must be writable, non-overlapping, and valid for their stated layouts.
pub unsafe extern "C" fn shapeit_pbwt_solve_all_v1(
    worker_count: usize,
    variant_major: *mut u8,
    variant_major_length: usize,
    variant_major_rows: usize,
    variant_major_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    graphs: *const *mut GenotypeGraphV1,
    graph_count: usize,
    site_chunks: *const i32,
    site_chunks_length: usize,
    chunk_starts: *const i32,
    chunk_count: usize,
    scores: *const f32,
    scores_length: usize,
    haplotype_major: *mut u8,
    haplotype_major_length: usize,
    haplotype_major_rows: usize,
    haplotype_major_stride: usize,
    progress: Option<PbwtProgressV1>,
    progress_context: *mut c_void,
    result: *mut PbwtBatchResultV1,
) -> u32 {
    if result.is_null() {
        return STATUS_NULL_POINTER;
    }
    if worker_count == 0 || graph_count == 0 || chunk_count == 0 {
        return STATUS_INVALID_DIMENSIONS;
    }
    if variant_major.is_null()
        || graphs.is_null()
        || site_chunks.is_null()
        || chunk_starts.is_null()
        || scores.is_null()
        || haplotype_major.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let graphs = slice::from_raw_parts(graphs, graph_count);
    let site_chunks = slice::from_raw_parts(site_chunks, site_chunks_length);
    let chunk_starts = slice::from_raw_parts(chunk_starts, chunk_count);
    let scores = slice::from_raw_parts(scores, scores_length);
    if site_chunks.len() != site_count
        || chunk_starts.iter().any(|&start| start < 0)
        || graphs.iter().any(|graph| graph.is_null())
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let target_haplotype_count = match graph_count.checked_mul(2) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if target_haplotype_count > haplotype_count {
        return STATUS_INVALID_DIMENSIONS;
    }
    let mut variants_length = None;
    let mut variant_views = Vec::with_capacity(graph_count);
    for &graph in graphs {
        let graph = &*graph;
        if graph.hmm_dimensions().0 != site_count {
            return STATUS_INVALID_DIMENSIONS;
        }
        let variants = graph.packed_variants();
        if variants_length.is_some_and(|expected| expected != variants.len()) {
            return STATUS_INVALID_DIMENSIONS;
        }
        variants_length = Some(variants.len());
        variant_views.push(variants.as_ptr());
    }
    let variant_views = SyncVariantViews(variant_views);

    let variant_major_bytes = match variant_major_rows.checked_mul(variant_major_stride) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if variant_major_bytes > variant_major_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variant_major_slice = slice::from_raw_parts(variant_major, variant_major_length);
    let mut buffers = Vec::with_capacity(chunk_count);
    for (chunk, &chunk_start) in chunk_starts.iter().enumerate() {
        let first_current = match site_chunks.iter().position(|&value| value == chunk as i32) {
            Some(value) => value,
            None => return STATUS_INVALID_DIMENSIONS,
        };
        let buffer_start = chunk_start as usize;
        if buffer_start > first_current {
            return STATUS_INVALID_DIMENSIONS;
        }
        let byte_start = match buffer_start.checked_mul(variant_major_stride) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        let byte_stop = match first_current.checked_mul(variant_major_stride) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        if byte_stop > variant_major_slice.len() {
            return STATUS_OUT_OF_BOUNDS;
        }
        buffers.push(variant_major_slice[byte_start..byte_stop].to_vec());
    }

    let shared = SolveAllShared {
        variant_major_address: variant_major as usize,
        variant_major_length,
        variant_major_stride,
        site_count,
        haplotype_count,
        variant_views: &variant_views,
        variants_length: variants_length.unwrap_or(0),
        site_chunks,
        chunk_starts,
        buffers: &buffers,
        scores,
        progress,
        progress_context_address: progress_context as usize,
        next_chunk: AtomicUsize::new(0),
        completed: AtomicUsize::new(0),
        status: AtomicU32::new(STATUS_OK),
        failed_chunk: AtomicUsize::new(usize::MAX),
        serialized_progress: Mutex::new(()),
    };
    let execution_threads = core::cmp::min(worker_count, chunk_count);
    if execution_threads == 1 {
        run_solve_all_worker(&shared);
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(execution_threads);
            for _ in 0..execution_threads {
                let shared = &shared;
                match thread::Builder::new().spawn_scoped(scope, move || {
                    run_solve_all_worker(shared);
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => record_solve_all_failure(shared, STATUS_THREAD_FAILURE, usize::MAX),
                }
            }
            for handle in handles {
                if handle.join().is_err() {
                    record_solve_all_failure(&shared, STATUS_THREAD_FAILURE, usize::MAX);
                }
            }
        });
    }
    let status = shared.status.load(Ordering::SeqCst);
    *result = PbwtBatchResultV1 {
        completed: shared.completed.load(Ordering::SeqCst),
        failed_chunk: shared.failed_chunk.load(Ordering::SeqCst),
    };
    if status != STATUS_OK {
        return status;
    }

    let max_rows = match site_count.checked_add(7) {
        Some(value) => value & !7,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let max_cols = match target_haplotype_count.checked_add(7) {
        Some(value) => value & !7,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let required_haplotype_major = match haplotype_major_rows.checked_mul(haplotype_major_stride) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if haplotype_major_rows < max_cols || required_haplotype_major > haplotype_major_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    shapeit_bitmatrix_transpose_v1(
        variant_major,
        variant_major_length,
        variant_major_rows,
        variant_major_stride,
        max_rows,
        max_cols,
        haplotype_major,
        haplotype_major_length,
        haplotype_major_stride,
    )
}

#[no_mangle]
/// Select IBD2-aware PBWT neighbours for one independently writable chunk.
///
/// Concurrent calls may share every input and the neighbour allocation when
/// chunk mappings select disjoint groups. Each call writes only the raw
/// group-major slots belonging to its selected sites.
///
/// # Safety
///
/// Every non-empty buffer must be valid for its stated length. Inputs must stay
/// immutable for the duration of the call. Concurrent output calls must write
/// disjoint groups. Invalid layouts are rejected before any output is written.
pub unsafe extern "C" fn shapeit_pbwt_select_chunk_v1(
    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    target_individual_count: usize,
    evaluated_sites: *const u8,
    evaluated_sites_length: usize,
    selected_sites: *const u8,
    selected_sites_length: usize,
    site_groups: *const i32,
    site_groups_length: usize,
    group_count: usize,
    site_chunks: *const i32,
    site_chunks_length: usize,
    chunk: usize,
    buffer_start: usize,
    depth: usize,
    ibd2: *const Ibd2TracksV1,
    neighbors: *mut i32,
    neighbors_length: usize,
) -> u32 {
    if haplotypes.is_null()
        || evaluated_sites.is_null()
        || selected_sites.is_null()
        || site_groups.is_null()
        || site_chunks.is_null()
        || ibd2.is_null()
        || neighbors.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let evaluated_sites = slice::from_raw_parts(evaluated_sites, evaluated_sites_length);
    let selected_sites = slice::from_raw_parts(selected_sites, selected_sites_length);
    let site_groups = slice::from_raw_parts(site_groups, site_groups_length);
    let site_chunks = slice::from_raw_parts(site_chunks, site_chunks_length);
    let ibd2 = &*ibd2;
    let layout = match validate_select_layout(SelectValidation {
        haplotypes_length,
        haplotype_stride,
        site_count,
        haplotype_count,
        target_individual_count,
        evaluated_sites,
        selected_sites,
        site_groups,
        group_count,
        site_chunks,
        chunk,
        buffer_start,
        depth,
        ibd2,
        neighbors_length,
    }) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status = select_chunk(SelectParameters {
        haplotypes,
        haplotype_stride,
        haplotype_count,
        target_haplotype_count: layout.target_haplotype_count,
        evaluated_sites,
        selected_sites,
        site_groups,
        site_chunks,
        chunk: chunk as i32,
        buffer_start,
        last_current: layout.last_current,
        depth,
        ibd2,
        neighbors,
        neighbor_slab: layout.neighbor_slab,
    });
    match status {
        Ok(()) => STATUS_OK,
        Err(status) => status,
    }
}

#[no_mangle]
/// Transpose raw group-major PBWT neighbours into HMM haplotype-major order.
///
/// The allocation must contain one extra slab, used as scratch exactly as in
/// the established common-phasing layout.
///
/// # Safety
///
/// `neighbors` must be writable for `neighbors_length` elements.
pub unsafe extern "C" fn shapeit_pbwt_transpose_neighbors_v1(
    neighbors: *mut i32,
    neighbors_length: usize,
    target_haplotype_count: usize,
    group_count: usize,
    depth: usize,
) -> u32 {
    if neighbors.is_null() {
        return STATUS_NULL_POINTER;
    }
    if target_haplotype_count == 0 || group_count == 0 || depth == 0 {
        return STATUS_INVALID_DIMENSIONS;
    }
    let slab = match target_haplotype_count.checked_mul(group_count) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let required = match depth
        .checked_add(1)
        .and_then(|count| count.checked_mul(slab))
    {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if required > neighbors_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let neighbors = slice::from_raw_parts_mut(neighbors, neighbors_length);
    let (output, scratch_and_remainder) = neighbors.split_at_mut(depth * slab);
    let scratch = &mut scratch_and_remainder[..slab];
    for selected_depth in 0..depth {
        let source = &output[selected_depth * slab..(selected_depth + 1) * slab];
        for group in 0..group_count {
            for haplotype in 0..target_haplotype_count {
                scratch[haplotype * group_count + group] =
                    source[group * target_haplotype_count + haplotype];
            }
        }
        output[selected_depth * slab..(selected_depth + 1) * slab].copy_from_slice(scratch);
    }
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_selection_uses_one_logical_stream_per_group() {
        let evaluated = [1u8, 0, 1, 1, 1, 0, 1];
        let groups = [0i32, 0, 0, 1, 1, 2, 2];
        let mut selected = [0xa5u8; 7];
        let status = unsafe {
            shapeit_pbwt_select_sites_v1(
                evaluated.as_ptr(),
                evaluated.len(),
                groups.as_ptr(),
                groups.len(),
                3,
                15_052_011,
                1,
                4,
                selected.as_mut_ptr(),
                selected.len(),
            )
        };
        assert_eq!(status, STATUS_OK);
        let candidates = [vec![0usize, 2], vec![3usize, 4], vec![6usize]];
        let mut expected = [0u8; 7];
        for (group, loci) in candidates.iter().enumerate() {
            let chosen = LogicalRng::new(15_052_011, 1, 4, group as u64)
                .next_bounded(loci.len() as u32) as usize;
            expected[loci[chosen]] = 1;
        }
        assert_eq!(selected, expected);
    }

    #[test]
    fn selection_preserves_pbwt_neighbor_order_and_transpose_layout() {
        let haplotypes = [0x30u8, 0x50];
        let evaluated = [1u8, 1];
        let selected = [1u8, 1];
        let groups = [0i32, 1];
        let chunks = [0i32, 0];
        let ibd2 = Ibd2TracksV1::new(2, &[0.0, 1.0]).unwrap();
        let mut neighbors = [-1i32; 16];
        let status = unsafe {
            select_chunk(SelectParameters {
                haplotypes: haplotypes.as_ptr(),
                haplotype_stride: 1,
                haplotype_count: 4,
                target_haplotype_count: 4,
                evaluated_sites: &evaluated,
                selected_sites: &selected,
                site_groups: &groups,
                site_chunks: &chunks,
                chunk: 0,
                buffer_start: 0,
                last_current: 1,
                depth: 1,
                ibd2: &ibd2,
                neighbors: neighbors.as_mut_ptr(),
                neighbor_slab: 8,
            })
        };
        assert_eq!(status, Ok(()));
        assert_eq!(&neighbors[..8], &[2, 2, 1, 1, 2, 3, 0, 1]);
        let status = unsafe {
            shapeit_pbwt_transpose_neighbors_v1(neighbors.as_mut_ptr(), neighbors.len(), 4, 2, 1)
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(&neighbors[..8], &[2, 2, 2, 3, 1, 0, 1, 1]);
    }

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
        let scores = [0.0f32, 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()];
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
