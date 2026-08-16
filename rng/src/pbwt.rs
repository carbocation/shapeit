use core::slice;

use crate::genotype::LogicalRng;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;
const STATUS_INSUFFICIENT_STATES: u32 = 5;

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
    ibd_offsets: &'a [usize],
    ibd_individuals: &'a [i32],
    ibd_from: &'a [i32],
    ibd_to: &'a [i32],
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
        ibd_offsets,
        ibd_individuals,
        ibd_from,
        ibd_to,
        neighbors_length,
    } = parameters;
    let expected_ibd_offsets = target_individual_count
        .checked_add(1)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
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
        || ibd_offsets.len() != expected_ibd_offsets
        || ibd_individuals.len() != ibd_from.len()
        || ibd_individuals.len() != ibd_to.len()
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

    if ibd_offsets.first().copied() != Some(0)
        || ibd_offsets.last().copied() != Some(ibd_individuals.len())
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    for source in 0..target_individual_count {
        let start = ibd_offsets[source];
        let stop = ibd_offsets[source + 1];
        if start > stop || stop > ibd_individuals.len() {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
        let mut previous_individual = -1i32;
        let mut previous_from = -1i32;
        for track in start..stop {
            let individual = ibd_individuals[track];
            let from = ibd_from[track];
            let to = ibd_to[track];
            if individual < source as i32
                || individual >= target_individual_count as i32
                || from < 0
                || from > to
                || to as usize >= site_count
                || individual < previous_individual
                || (individual == previous_individual && from < previous_from)
            {
                return Err(STATUS_INVALID_DIMENSIONS);
            }
            previous_individual = individual;
            previous_from = from;
        }
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

struct IbdTracks<'a> {
    offsets: &'a [usize],
    individuals: &'a [i32],
    from: &'a [i32],
    to: &'a [i32],
}

impl IbdTracks<'_> {
    #[inline]
    fn allows(&self, haplotype0: usize, haplotype1: usize, locus: usize) -> bool {
        let individual0 = haplotype0 / 2;
        let individual1 = haplotype1 / 2;
        let source = core::cmp::min(individual0, individual1);
        let target = core::cmp::max(individual0, individual1);
        if source == target {
            return false;
        }
        for track in self.offsets[source]..self.offsets[source + 1] {
            let tracked_individual = self.individuals[track] as usize;
            if tracked_individual > target {
                break;
            }
            if tracked_individual == target
                && self.from[track] as usize <= locus
                && locus <= self.to[track] as usize
            {
                return false;
            }
        }
        true
    }
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
    ibd: IbdTracks<'a>,
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
        ibd,
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
                    left.filter(|&(candidate, _)| ibd.allows(target_haplotype, candidate, locus));
                let allowed_right =
                    right.filter(|&(candidate, _)| ibd.allows(target_haplotype, candidate, locus));

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
    ibd_offsets: *const usize,
    ibd_offsets_length: usize,
    ibd_individuals: *const i32,
    ibd_from: *const i32,
    ibd_to: *const i32,
    ibd_track_count: usize,
    neighbors: *mut i32,
    neighbors_length: usize,
) -> u32 {
    if haplotypes.is_null()
        || evaluated_sites.is_null()
        || selected_sites.is_null()
        || site_groups.is_null()
        || site_chunks.is_null()
        || ibd_offsets.is_null()
        || neighbors.is_null()
        || (ibd_track_count != 0
            && (ibd_individuals.is_null() || ibd_from.is_null() || ibd_to.is_null()))
    {
        return STATUS_NULL_POINTER;
    }
    let evaluated_sites = slice::from_raw_parts(evaluated_sites, evaluated_sites_length);
    let selected_sites = slice::from_raw_parts(selected_sites, selected_sites_length);
    let site_groups = slice::from_raw_parts(site_groups, site_groups_length);
    let site_chunks = slice::from_raw_parts(site_chunks, site_chunks_length);
    let ibd_offsets = slice::from_raw_parts(ibd_offsets, ibd_offsets_length);
    let ibd_individuals = if ibd_track_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(ibd_individuals, ibd_track_count)
    };
    let ibd_from = if ibd_track_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(ibd_from, ibd_track_count)
    };
    let ibd_to = if ibd_track_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(ibd_to, ibd_track_count)
    };
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
        ibd_offsets,
        ibd_individuals,
        ibd_from,
        ibd_to,
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
        ibd: IbdTracks {
            offsets: ibd_offsets,
            individuals: ibd_individuals,
            from: ibd_from,
            to: ibd_to,
        },
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
        let ibd_offsets = [0usize, 0, 0];
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
                ibd: IbdTracks {
                    offsets: &ibd_offsets,
                    individuals: &[],
                    from: &[],
                    to: &[],
                },
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
