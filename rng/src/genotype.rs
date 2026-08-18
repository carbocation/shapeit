use core::slice;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

const ABI_VERSION: u32 = 1;
pub(crate) const STATUS_OK: u32 = 0;
pub(crate) const STATUS_NULL_POINTER: u32 = 1;
pub(crate) const STATUS_INVALID_DIMENSIONS: u32 = 2;
pub(crate) const STATUS_OUT_OF_BOUNDS: u32 = 3;
pub(crate) const STATUS_INTEGER_OVERFLOW: u32 = 4;
const STATUS_THREAD_FAILURE: u32 = 5;

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

pub(crate) struct LogicalRng {
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
    next_block: u64,
    words: [u32; 4],
    next_word: usize,
}

impl LogicalRng {
    pub(crate) fn new(seed: u64, domain: u32, iteration: u32, item: u64) -> Self {
        Self {
            seed,
            domain,
            iteration,
            item,
            next_block: 0,
            words: [0; 4],
            next_word: 4,
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.next_word == self.words.len() {
            self.words = super::application_block(
                self.seed,
                self.domain,
                self.iteration,
                self.item,
                self.next_block,
            );
            self.next_block = self.next_block.wrapping_add(1);
            self.next_word = 0;
        }
        let value = self.words[self.next_word];
        self.next_word += 1;
        value
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let bits = (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32());
        ((bits >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    #[inline]
    pub(crate) fn next_bounded(&mut self, range: u32) -> u32 {
        debug_assert!(range != 0);
        let mut value = self.next_u32();
        let mut product = u64::from(value) * u64::from(range);
        let mut low = product as u32;
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                value = self.next_u32();
                product = u64::from(value) * u64::from(range);
                low = product as u32;
            }
        }
        (product >> 32) as u32
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SampleLayout {
    transitions: usize,
    missing: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SampleError {
    Status(u32),
    DegenerateDistribution,
}

impl SampleError {
    fn status(self) -> u32 {
        match self {
            Self::Status(status) => status,
            Self::DegenerateDistribution => STATUS_INVALID_DIMENSIONS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenotypeWindowV1 {
    pub(crate) start_locus: i32,
    pub(crate) start_segment: i32,
    pub(crate) start_ambiguous: i32,
    pub(crate) start_missing: i32,
    pub(crate) start_transition: i32,
    pub(crate) stop_locus: i32,
    pub(crate) stop_segment: i32,
    pub(crate) stop_ambiguous: i32,
    pub(crate) stop_missing: i32,
    pub(crate) stop_transition: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenotypeStorageViewV1 {
    transition_count: usize,
    transition_mask: *const u8,
    transition_mask_length: usize,
    transition_probabilities: *const f32,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    storage_events: u32,
}

pub struct GenotypeStorageV1 {
    transition_count: usize,
    transition_mask: Vec<u8>,
    transition_indexes: Vec<u32>,
    transition_probabilities: Vec<f32>,
    missing_probabilities: Vec<f32>,
    storage_events: u32,
}

pub type GenotypeProgressV1 = unsafe extern "C" fn(usize, usize, *mut c_void);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GenotypeBatchResultV1 {
    completed: usize,
    failed_graph: usize,
    segments: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenotypeGraphViewV1 {
    variant_count: usize,
    variants: *const u8,
    variants_length: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    missing_count: usize,
    transition_count: u32,
}

pub struct GenotypeGraphV1 {
    variant_count: usize,
    variants: Vec<u8>,
    ambiguous: Vec<u8>,
    diplotypes: Vec<u64>,
    segment_lengths: Vec<u16>,
    missing_count: usize,
    transition_count: u32,
    storage: Option<GenotypeStorageV1>,
    built: bool,
    haploid: bool,
    double_precision: bool,
}

impl GenotypeGraphV1 {
    fn view(&self) -> GenotypeGraphViewV1 {
        GenotypeGraphViewV1 {
            variant_count: self.variant_count,
            variants: self.variants.as_ptr(),
            variants_length: self.variants.len(),
            ambiguous: self.ambiguous.as_ptr(),
            ambiguous_length: self.ambiguous.len(),
            diplotypes: self.diplotypes.as_ptr(),
            diplotypes_length: self.diplotypes.len(),
            segment_lengths: self.segment_lengths.as_ptr(),
            segment_lengths_length: self.segment_lengths.len(),
            missing_count: self.missing_count,
            transition_count: self.transition_count,
        }
    }

    pub(crate) fn hmm_arrays(&self) -> (&[u8], &[u8], &[u16], &[u64]) {
        (
            &self.variants,
            &self.ambiguous,
            &self.segment_lengths,
            &self.diplotypes,
        )
    }

    pub(crate) fn packed_variants(&self) -> &[u8] {
        &self.variants
    }

    pub(crate) fn hmm_dimensions(&self) -> (usize, usize, usize) {
        (
            self.variant_count,
            self.transition_count as usize,
            self.missing_count,
        )
    }

    pub(crate) fn is_built(&self) -> bool {
        self.built
    }

    pub(crate) fn requires_double_precision(&self) -> bool {
        self.double_precision
    }

    pub(crate) fn require_double_precision(&mut self) {
        self.double_precision = true;
    }
}

impl GenotypeStorageV1 {
    fn new(transition_probabilities: &[f64], missing_probabilities: &[f32]) -> Result<Self, u32> {
        if transition_probabilities.len() > u32::MAX as usize {
            return Err(STATUS_INTEGER_OVERFLOW);
        }
        let mut transition_mask = vec![0u8; transition_probabilities.len().div_ceil(8)];
        let mut transition_indexes = Vec::new();
        let mut stored_probabilities = Vec::new();
        for (index, &probability) in transition_probabilities.iter().enumerate() {
            if probability >= 1e-6 {
                transition_mask[index >> 3] |= 1 << (index & 7);
                transition_indexes.push(index as u32);
                stored_probabilities.push(probability as f32);
            }
        }
        Ok(Self {
            transition_count: transition_probabilities.len(),
            transition_mask,
            transition_indexes,
            transition_probabilities: stored_probabilities,
            missing_probabilities: missing_probabilities.to_vec(),
            storage_events: 1,
        })
    }

    fn update(
        &mut self,
        transition_probabilities: &[f64],
        missing_probabilities: &[f32],
    ) -> Result<(), u32> {
        if transition_probabilities.len() != self.transition_count
            || missing_probabilities.len() != self.missing_probabilities.len()
        {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let storage_events = self
            .storage_events
            .checked_add(1)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        for (&index, stored) in self
            .transition_indexes
            .iter()
            .zip(self.transition_probabilities.iter_mut())
        {
            *stored = (f64::from(*stored) + transition_probabilities[index as usize]) as f32;
        }
        for (stored, &current) in self
            .missing_probabilities
            .iter_mut()
            .zip(missing_probabilities)
        {
            *stored += current;
        }
        self.storage_events = storage_events;
        Ok(())
    }

    fn view(&self) -> GenotypeStorageViewV1 {
        GenotypeStorageViewV1 {
            transition_count: self.transition_count,
            transition_mask: self.transition_mask.as_ptr(),
            transition_mask_length: self.transition_mask.len(),
            transition_probabilities: self.transition_probabilities.as_ptr(),
            transition_probabilities_length: self.transition_probabilities.len(),
            missing_probabilities: self.missing_probabilities.as_ptr(),
            missing_probabilities_length: self.missing_probabilities.len(),
            storage_events: self.storage_events,
        }
    }
}

#[inline]
fn variant_nibble(variants: &[u8], locus: usize) -> u8 {
    (variants[locus >> 1] >> ((locus & 1) << 2)) & 0x0f
}

#[inline]
fn set_variant_nibble(variants: &mut [u8], locus: usize, value: u8) {
    let shift = (locus & 1) << 2;
    variants[locus >> 1] = (variants[locus >> 1] & !(0x0f << shift)) | (value << shift);
}

#[derive(Clone, Copy)]
enum PedigreeMode {
    Trio,
    Father,
    Mother,
}

fn scaffold_pedigree(
    child: &mut [u8],
    father: Option<&[u8]>,
    mother: Option<&[u8]>,
    variant_count: usize,
    mode: PedigreeMode,
    counts: &mut [u32],
) {
    for locus in 0..variant_count {
        let mut child_code = variant_nibble(child, locus);
        let child_graph = graph_code(child_code);
        let father_code = father.map(|variants| variant_nibble(variants, locus));
        let mother_code = mother.map(|variants| variant_nibble(variants, locus));
        if child_graph == 2 {
            let father_homozygous = father_code.is_some_and(|code| graph_code(code) == 0);
            let mother_homozygous = mother_code.is_some_and(|code| graph_code(code) == 0);
            match mode {
                PedigreeMode::Trio if father_homozygous && mother_homozygous => {
                    let father_allele = father_code.unwrap() & 4 != 0;
                    let mother_allele = mother_code.unwrap() & 4 != 0;
                    if father_allele != mother_allele {
                        child_code |= 3;
                        counts[2] = counts[2].wrapping_add(1);
                        if father_allele {
                            child_code = (child_code | 4) & !8;
                        } else {
                            child_code = (child_code & !4) | 8;
                        }
                    } else {
                        counts[0] = counts[0].wrapping_add(1);
                    }
                }
                PedigreeMode::Trio | PedigreeMode::Father if father_homozygous => {
                    let father_allele = father_code.unwrap() & 4 != 0;
                    child_code |= 3;
                    counts[2] = counts[2].wrapping_add(1);
                    if father_allele {
                        child_code = (child_code | 4) & !8;
                    } else {
                        child_code = (child_code & !4) | 8;
                    }
                }
                PedigreeMode::Trio | PedigreeMode::Mother if mother_homozygous => {
                    let mother_allele = mother_code.unwrap() & 4 != 0;
                    child_code |= 3;
                    counts[2] = counts[2].wrapping_add(1);
                    if mother_allele {
                        child_code = (child_code & !4) | 8;
                    } else {
                        child_code = (child_code | 4) & !8;
                    }
                }
                _ => counts[3] = counts[3].wrapping_add(1),
            }
            counts[1] = counts[1].wrapping_add(1);
            set_variant_nibble(child, locus, child_code);
        } else if child_graph == 0 {
            let child_allele = child_code & 4 != 0;
            if matches!(mode, PedigreeMode::Trio | PedigreeMode::Father)
                && father_code
                    .is_some_and(|code| graph_code(code) == 0 && (code & 4 != 0) != child_allele)
            {
                counts[0] = counts[0].wrapping_add(1);
            }
            if matches!(mode, PedigreeMode::Trio | PedigreeMode::Mother)
                && mother_code
                    .is_some_and(|code| graph_code(code) == 0 && (code & 4 != 0) != child_allele)
            {
                counts[0] = counts[0].wrapping_add(1);
            }
            counts[1] = counts[1].wrapping_add(1);
        } else if child_graph == 1 && matches!(mode, PedigreeMode::Trio) {
            let father_allele = father_code.unwrap() & 4 != 0;
            let mother_allele = mother_code.unwrap() & 4 != 0;
            if father_allele != mother_allele {
                child_code |= 3;
            } else {
                child_code &= !3;
            }
            if father_allele {
                child_code |= 4;
            } else {
                child_code &= !4;
            }
            if mother_allele {
                child_code |= 8;
            } else {
                child_code &= !8;
            }
            set_variant_nibble(child, locus, child_code);
        }
    }
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

fn byte_ranges_overlap(
    first: *const u8,
    first_length: usize,
    second: *const u8,
    second_length: usize,
) -> Result<bool, u32> {
    if first_length == 0 || second_length == 0 {
        return Ok(false);
    }
    let first_start = first as usize;
    let second_start = second as usize;
    let first_stop = first_start
        .checked_add(first_length)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let second_stop = second_start
        .checked_add(second_length)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    Ok(first_start < second_stop && second_start < first_stop)
}

struct PrunedGraph {
    ambiguous: Vec<u8>,
    diplotypes: Vec<u64>,
    segment_lengths: Vec<u16>,
    transition_count: u32,
}

#[derive(Clone, Copy)]
struct PruneStatistic {
    entropy: f64,
    segment: usize,
    mergeable: bool,
}

fn collect_diplotypes(mask: u64) -> Vec<u8> {
    let mut codes = Vec::with_capacity(mask.count_ones() as usize);
    let mut active = mask;
    while active != 0 {
        codes.push(active.trailing_zeros() as u8);
        active &= active - 1;
    }
    codes
}

fn rank_transitions(probabilities: &[f64], order: &mut Vec<usize>) {
    order.clear();
    order.extend(0..probabilities.len());
    order.sort_unstable_by(|&first, &second| {
        probabilities[second]
            .total_cmp(&probabilities[first])
            .then_with(|| first.cmp(&second))
    });
}

fn map_prune_merges(
    variants: &[u8],
    diplotypes: &[u64],
    segment_lengths: &[u16],
    transition_probabilities: &[f64],
    threshold_probability_mass: f64,
) -> Vec<bool> {
    let mut statistics = Vec::with_capacity(diplotypes.len().saturating_sub(1));
    let mut transition_order = Vec::with_capacity(4096);
    let mut previous_diplotypes = collect_diplotypes(diplotypes[0]);
    let mut transition_offset = previous_diplotypes.len();
    let mut locus_offset = 0usize;

    for segment in 1..diplotypes.len() {
        let current_diplotypes = collect_diplotypes(diplotypes[segment]);
        let transition_count = previous_diplotypes.len() * current_diplotypes.len();
        let mut statistic = PruneStatistic {
            entropy: 4096.0,
            segment,
            mergeable: false,
        };
        let merged_length =
            usize::from(segment_lengths[segment - 1]) + usize::from(segment_lengths[segment]);
        if merged_length < usize::from(u16::MAX) {
            let merged_ambiguous = (locus_offset..locus_offset + merged_length)
                .filter(|&locus| graph_code(variant_nibble(variants, locus)) > 1)
                .count();
            if merged_ambiguous < MAX_AMBIGUOUS_PER_SEGMENT {
                let probabilities = &transition_probabilities
                    [transition_offset..transition_offset + transition_count];
                rank_transitions(probabilities, &mut transition_order);
                statistic.entropy = transition_order.iter().fold(0.0, |entropy, &index| {
                    let probability = probabilities[index];
                    let information = if probability == 0.0 {
                        0.0
                    } else {
                        -probability.log10()
                    };
                    entropy + probability * information
                });

                let mut mapped_haplotypes = [-1i16; 64];
                let mut haplotype_count = 0i16;
                let mut cumulative_probability = 0.0;
                for &index in &transition_order {
                    cumulative_probability += probabilities[index];
                    let previous =
                        usize::from(previous_diplotypes[index / current_diplotypes.len()]);
                    let current = usize::from(current_diplotypes[index % current_diplotypes.len()]);
                    let merged_haplotype0 = (previous >> 3) * 8 + (current >> 3);
                    let merged_haplotype1 = (previous & 7) * 8 + (current & 7);
                    if mapped_haplotypes[merged_haplotype0] < 0 {
                        mapped_haplotypes[merged_haplotype0] = haplotype_count;
                        haplotype_count += 1;
                    }
                    if merged_haplotype0 != merged_haplotype1
                        && mapped_haplotypes[merged_haplotype1] < 0
                    {
                        mapped_haplotypes[merged_haplotype1] = haplotype_count;
                        haplotype_count += 1;
                    }
                    if haplotype_count == 8 && cumulative_probability > threshold_probability_mass {
                        statistic.mergeable = true;
                    }
                }
            }
        }
        statistics.push(statistic);
        locus_offset += usize::from(segment_lengths[segment - 1]);
        transition_offset += transition_count;
        previous_diplotypes = current_diplotypes;
    }

    statistics.sort_unstable_by(|first, second| {
        first
            .entropy
            .total_cmp(&second.entropy)
            .then_with(|| first.segment.cmp(&second.segment))
    });
    let mut merge_flags = vec![false; diplotypes.len() + 1];
    for statistic in statistics {
        let no_adjacent_merges =
            !merge_flags[statistic.segment - 1] && !merge_flags[statistic.segment + 1];
        merge_flags[statistic.segment] = no_adjacent_merges && statistic.mergeable;
    }
    merge_flags
}

fn copy_ambiguous_segment(
    variants: &[u8],
    ambiguous: &[u8],
    output: &mut [u8],
    locus_offset: usize,
    locus_count: usize,
    ambiguous_offset: usize,
) {
    let mut relative_ambiguous = 0usize;
    for locus in locus_offset..locus_offset + locus_count {
        if graph_code(variant_nibble(variants, locus)) > 1 {
            output[ambiguous_offset + relative_ambiguous] =
                ambiguous[ambiguous_offset + relative_ambiguous];
            relative_ambiguous += 1;
        }
    }
}

fn count_graph_transitions(diplotypes: &[u64]) -> Result<u32, u32> {
    let mut previous_count = 1usize;
    let mut transition_count = 0usize;
    for &mask in diplotypes {
        let current_count = mask.count_ones() as usize;
        if current_count == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        transition_count = transition_count
            .checked_add(
                previous_count
                    .checked_mul(current_count)
                    .ok_or(STATUS_INTEGER_OVERFLOW)?,
            )
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        previous_count = current_count;
    }
    u32::try_from(transition_count).map_err(|_| STATUS_INTEGER_OVERFLOW)
}

fn prune_graph(
    variants: &[u8],
    ambiguous: &[u8],
    diplotypes: &[u64],
    segment_lengths: &[u16],
    transition_probabilities: &[f64],
    threshold_probability_mass: f64,
) -> Result<PrunedGraph, u32> {
    let merge_flags = map_prune_merges(
        variants,
        diplotypes,
        segment_lengths,
        transition_probabilities,
        threshold_probability_mass,
    );
    let merge_count = merge_flags.iter().filter(|&&merge| merge).count();
    let mut output_ambiguous = vec![0u8; ambiguous.len()];
    let mut output_diplotypes = Vec::with_capacity(diplotypes.len() - merge_count);
    let mut output_segment_lengths = Vec::with_capacity(diplotypes.len() - merge_count);
    let mut transition_order = Vec::with_capacity(4096);
    let mut previous_diplotypes = collect_diplotypes(diplotypes[0]);
    let mut transition_offset = previous_diplotypes.len();
    let mut ambiguous_offset = 0usize;
    let mut locus_offset = 0usize;

    for segment in 1..diplotypes.len() {
        let current_diplotypes = collect_diplotypes(diplotypes[segment]);
        let transition_count = previous_diplotypes.len() * current_diplotypes.len();
        if merge_flags[segment] {
            let previous_length = usize::from(segment_lengths[segment - 1]);
            let merged_length = previous_length + usize::from(segment_lengths[segment]);
            output_segment_lengths.push(merged_length as u16);
            let probabilities =
                &transition_probabilities[transition_offset..transition_offset + transition_count];
            rank_transitions(probabilities, &mut transition_order);
            let mut mapped_haplotypes = [-1i16; 64];
            let mut haplotype_count = 0usize;
            let mut output_mask = 0u64;
            for &index in &transition_order {
                let previous = usize::from(previous_diplotypes[index / current_diplotypes.len()]);
                let current = usize::from(current_diplotypes[index % current_diplotypes.len()]);
                let previous_haplotype0 = previous >> 3;
                let previous_haplotype1 = previous & 7;
                let current_haplotype0 = current >> 3;
                let current_haplotype1 = current & 7;
                let merged_haplotype0 = previous_haplotype0 * 8 + current_haplotype0;
                let merged_haplotype1 = previous_haplotype1 * 8 + current_haplotype1;
                let new_haplotype0 = mapped_haplotypes[merged_haplotype0] < 0;
                let new_haplotype1 = merged_haplotype0 != merged_haplotype1
                    && mapped_haplotypes[merged_haplotype1] < 0;
                if haplotype_count + usize::from(new_haplotype0) + usize::from(new_haplotype1) <= 8
                {
                    if new_haplotype0 {
                        mapped_haplotypes[merged_haplotype0] = haplotype_count as i16;
                        let mut relative_ambiguous = 0usize;
                        for relative_locus in 0..merged_length {
                            let locus = locus_offset + relative_locus;
                            if graph_code(variant_nibble(variants, locus)) > 1 {
                                let source_haplotype = if relative_locus < previous_length {
                                    previous_haplotype0
                                } else {
                                    current_haplotype0
                                };
                                if (ambiguous[ambiguous_offset + relative_ambiguous]
                                    >> source_haplotype)
                                    & 1
                                    != 0
                                {
                                    output_ambiguous[ambiguous_offset + relative_ambiguous] |=
                                        1 << haplotype_count;
                                }
                                relative_ambiguous += 1;
                            }
                        }
                        haplotype_count += 1;
                    }
                    if new_haplotype1 {
                        mapped_haplotypes[merged_haplotype1] = haplotype_count as i16;
                        let mut relative_ambiguous = 0usize;
                        for relative_locus in 0..merged_length {
                            let locus = locus_offset + relative_locus;
                            if graph_code(variant_nibble(variants, locus)) > 1 {
                                let source_haplotype = if relative_locus < previous_length {
                                    previous_haplotype1
                                } else {
                                    current_haplotype1
                                };
                                if (ambiguous[ambiguous_offset + relative_ambiguous]
                                    >> source_haplotype)
                                    & 1
                                    != 0
                                {
                                    output_ambiguous[ambiguous_offset + relative_ambiguous] |=
                                        1 << haplotype_count;
                                }
                                relative_ambiguous += 1;
                            }
                        }
                        haplotype_count += 1;
                    }
                    let mapped0 = mapped_haplotypes[merged_haplotype0];
                    let mapped1 = mapped_haplotypes[merged_haplotype1];
                    if mapped0 < 0 || mapped1 < 0 {
                        return Err(STATUS_INVALID_DIMENSIONS);
                    }
                    output_mask |= 1u64
                        << (usize::try_from(mapped0).unwrap() * 8
                            + usize::try_from(mapped1).unwrap());
                }
            }
            if haplotype_count != 8 {
                return Err(STATUS_INVALID_DIMENSIONS);
            }
            output_diplotypes.push(output_mask);
        } else if !merge_flags[segment - 1] {
            copy_ambiguous_segment(
                variants,
                ambiguous,
                &mut output_ambiguous,
                locus_offset,
                usize::from(segment_lengths[segment - 1]),
                ambiguous_offset,
            );
            output_segment_lengths.push(segment_lengths[segment - 1]);
            output_diplotypes.push(diplotypes[segment - 1]);
        }

        let previous_length = usize::from(segment_lengths[segment - 1]);
        ambiguous_offset += (locus_offset..locus_offset + previous_length)
            .filter(|&locus| graph_code(variant_nibble(variants, locus)) > 1)
            .count();
        locus_offset += previous_length;
        transition_offset += transition_count;
        previous_diplotypes = current_diplotypes;
    }

    if !merge_flags[diplotypes.len() - 1] {
        copy_ambiguous_segment(
            variants,
            ambiguous,
            &mut output_ambiguous,
            locus_offset,
            usize::from(*segment_lengths.last().unwrap()),
            ambiguous_offset,
        );
        output_segment_lengths.push(*segment_lengths.last().unwrap());
        output_diplotypes.push(*diplotypes.last().unwrap());
    }
    let transition_count = count_graph_transitions(&output_diplotypes)?;
    Ok(PrunedGraph {
        ambiguous: output_ambiguous,
        diplotypes: output_diplotypes,
        segment_lengths: output_segment_lengths,
        transition_count,
    })
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

impl GenotypeGraphV1 {
    fn new(variants: &[u8], variant_count: usize) -> Self {
        let sizes = graph_sizes(variants, variant_count);
        let mut segment_lengths = vec![0u16; sizes.segments];
        let mut ambiguous = vec![0u8; sizes.ambiguous];
        let mut diplotypes = vec![0u64; sizes.segments];
        let transition_count = build_graph(
            variants,
            variant_count,
            &mut segment_lengths,
            &mut ambiguous,
            &mut diplotypes,
        );
        Self {
            variant_count,
            variants: variants.to_vec(),
            ambiguous,
            diplotypes,
            segment_lengths,
            missing_count: sizes.missing,
            transition_count,
            storage: None,
            built: true,
            haploid: false,
            double_precision: false,
        }
    }

    fn allocate(variant_count: usize, variants_length: usize) -> Self {
        Self {
            variant_count,
            variants: vec![0u8; variants_length],
            ambiguous: Vec::new(),
            diplotypes: Vec::new(),
            segment_lengths: Vec::new(),
            missing_count: 0,
            transition_count: 0,
            storage: None,
            built: false,
            haploid: false,
            double_precision: false,
        }
    }

    fn build_in_place(&mut self) -> Result<(), u32> {
        if self.built {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let sizes = graph_sizes(&self.variants, self.variant_count);
        let mut segment_lengths = vec![0u16; sizes.segments];
        let mut ambiguous = vec![0u8; sizes.ambiguous];
        let mut diplotypes = vec![0u64; sizes.segments];
        let transition_count = build_graph(
            &self.variants,
            self.variant_count,
            &mut segment_lengths,
            &mut ambiguous,
            &mut diplotypes,
        );
        self.ambiguous = ambiguous;
        self.diplotypes = diplotypes;
        self.segment_lengths = segment_lengths;
        self.missing_count = sizes.missing;
        self.transition_count = transition_count;
        self.built = true;
        Ok(())
    }
}

fn validate_sample_layout(
    variants: &[u8],
    variant_count: usize,
    ambiguous_length: usize,
    diplotypes: &[u64],
    segment_lengths: &[u16],
    transition_probabilities_length: usize,
    missing_probabilities_length: usize,
) -> Result<SampleLayout, u32> {
    if diplotypes.is_empty() || diplotypes.len() != segment_lengths.len() {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let mut loci = 0usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    for &length in segment_lengths {
        loci = loci
            .checked_add(usize::from(length))
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
    }
    if loci != variant_count {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    for locus in 0..variant_count {
        match graph_code(variant_nibble(variants, locus)) {
            1 => missing += 1,
            2 | 3 => ambiguous += 1,
            _ => {}
        }
    }
    if ambiguous != ambiguous_length {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let required_missing = missing.checked_mul(8).ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_missing > missing_probabilities_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let mut previous = 1usize;
    let mut transitions = 0usize;
    for &diplotype in diplotypes {
        let current = diplotype.count_ones() as usize;
        if current == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        transitions = transitions
            .checked_add(
                previous
                    .checked_mul(current)
                    .ok_or(STATUS_INTEGER_OVERFLOW)?,
            )
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        previous = current;
    }
    if transitions > transition_probabilities_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(SampleLayout {
        transitions,
        missing,
    })
}

#[inline]
fn sample_probabilities(
    probabilities: &[f64],
    total: f64,
    rng: &mut LogicalRng,
) -> Result<usize, SampleError> {
    if probabilities.is_empty() || !total.is_finite() || total <= 0.0 {
        return Err(SampleError::DegenerateDistribution);
    }
    let mut cumulative = probabilities[0];
    let draw = rng.next_f64() * total;
    for index in 0..probabilities.len() - 1 {
        if draw < cumulative {
            return Ok(index);
        }
        cumulative += probabilities[index + 1];
    }
    Ok(probabilities.len() - 1)
}

#[inline]
fn diplotype_code(mask: u64, index: usize) -> u8 {
    let mut active = mask;
    for _ in 0..index {
        active &= active - 1;
    }
    active.trailing_zeros() as u8
}

fn sample_forward(
    diplotypes: &[u64],
    transition_probabilities: &[f64],
    sampled: &mut [u8],
    rng: &mut LogicalRng,
) -> Result<(), SampleError> {
    let mut probabilities = [0.0f64; 64];
    let mut previous_sampled = 0usize;
    let mut previous_count = 1usize;
    let mut transition_offset = 0usize;
    for (segment, &diplotype) in diplotypes.iter().enumerate() {
        let current_count = diplotype.count_ones() as usize;
        let start = transition_offset + previous_sampled * current_count;
        let mut total = 0.0f64;
        for relative in 0..current_count {
            probabilities[relative] = transition_probabilities[start + relative];
            if !probabilities[relative].is_finite() || probabilities[relative] < 0.0 {
                return Err(SampleError::DegenerateDistribution);
            }
            total += probabilities[relative];
        }
        previous_sampled = sample_probabilities(&probabilities[..current_count], total, rng)?;
        sampled[segment] = diplotype_code(diplotype, previous_sampled);
        transition_offset += previous_count * current_count;
        previous_count = current_count;
    }
    Ok(())
}

fn sample_backward(
    diplotypes: &[u64],
    transition_probabilities: &[f64],
    transitions: usize,
    sampled: &mut [u8],
    rng: &mut LogicalRng,
) -> Result<(), SampleError> {
    let mut probabilities = [0.0f64; 64 * 64];
    let mut next_sampled = None;
    let mut next_count = diplotypes.last().unwrap().count_ones() as usize;
    let mut transition_offset = transitions;
    for segment in (0..diplotypes.len().saturating_sub(1)).rev() {
        let current_count = diplotypes[segment].count_ones() as usize;
        transition_offset -= next_count * current_count;
        if let Some(next_sampled_index) = next_sampled {
            let mut total = 0.0f64;
            for relative in 0..current_count {
                probabilities[relative] = transition_probabilities
                    [transition_offset + next_sampled_index + relative * next_count];
                if !probabilities[relative].is_finite() || probabilities[relative] < 0.0 {
                    return Err(SampleError::DegenerateDistribution);
                }
                total += probabilities[relative];
            }
            let current_sampled =
                sample_probabilities(&probabilities[..current_count], total, rng)?;
            sampled[segment] = diplotype_code(diplotypes[segment], current_sampled);
            next_sampled = Some(current_sampled);
        } else {
            let block_length = next_count * current_count;
            let mut total = 0.0f64;
            for relative in 0..block_length {
                probabilities[relative] = transition_probabilities[transition_offset + relative];
                if !probabilities[relative].is_finite() || probabilities[relative] < 0.0 {
                    return Err(SampleError::DegenerateDistribution);
                }
                total += probabilities[relative];
            }
            let joint_sampled = sample_probabilities(&probabilities[..block_length], total, rng)?;
            sampled[segment + 1] =
                diplotype_code(diplotypes[segment + 1], joint_sampled % next_count);
            let current_sampled = joint_sampled / next_count;
            sampled[segment] = diplotype_code(diplotypes[segment], current_sampled);
            next_sampled = Some(current_sampled);
        }
        next_count = current_count;
    }
    Ok(())
}

struct SampleParameters<'a> {
    variants: &'a mut [u8],
    ambiguous: &'a [u8],
    diplotypes: &'a [u64],
    segment_lengths: &'a [u16],
    transition_probabilities: &'a [f64],
    missing_probabilities: &'a [f32],
    haploid: bool,
}

fn sample_validated(
    parameters: SampleParameters<'_>,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> Result<(), SampleError> {
    let SampleParameters {
        variants,
        ambiguous,
        diplotypes,
        segment_lengths,
        transition_probabilities,
        missing_probabilities,
        haploid,
    } = parameters;
    let mut rng = LogicalRng::new(seed, domain, iteration, item);
    let mut sampled = vec![0u8; diplotypes.len()];
    if rng.next_f64() < 0.5 {
        sample_forward(diplotypes, transition_probabilities, &mut sampled, &mut rng)?;
    } else {
        sample_backward(
            diplotypes,
            transition_probabilities,
            transition_probabilities.len(),
            &mut sampled,
            &mut rng,
        )?;
    }
    apply_sample(
        ApplySample {
            variants,
            ambiguous,
            segment_lengths,
            sampled: &sampled,
            missing_probabilities,
            haploid,
        },
        &mut rng,
    );
    Ok(())
}

#[inline]
fn set_haplotype(variants: &mut [u8], locus: usize, haplotype: usize, allele: bool) {
    let byte = &mut variants[locus >> 1];
    let shift = ((locus & 1) << 2) + 2 + haplotype;
    let mask = 1u8 << shift;
    *byte = (*byte & !mask) | (u8::from(allele) << shift);
}

struct ApplySample<'a> {
    variants: &'a mut [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    sampled: &'a [u8],
    missing_probabilities: &'a [f32],
    haploid: bool,
}

fn apply_sample(parameters: ApplySample<'_>, rng: &mut LogicalRng) {
    let ApplySample {
        variants,
        ambiguous,
        segment_lengths,
        sampled,
        missing_probabilities,
        haploid,
    } = parameters;
    let mut absolute_locus = 0usize;
    let mut ambiguous_index = 0usize;
    let mut missing_index = 0usize;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let haplotype0 = usize::from(sampled[segment] >> 3);
        let haplotype1 = usize::from(sampled[segment] & 7);
        for _ in 0..length {
            let code = graph_code(variant_nibble(variants, absolute_locus));
            if code == 1 {
                let start = missing_index * 8;
                if haploid {
                    let probability0 = missing_probabilities[start + haplotype0];
                    let probability1 = missing_probabilities[start + haplotype1];
                    let probability00 = (1.0 - probability0) * (1.0 - probability1);
                    let probability11 = probability0 * probability1;
                    let allele = rng.next_f64()
                        <= f64::from(probability11 / (probability00 + probability11));
                    set_haplotype(variants, absolute_locus, 0, allele);
                    set_haplotype(variants, absolute_locus, 1, allele);
                } else {
                    let allele0 =
                        rng.next_f64() <= f64::from(missing_probabilities[start + haplotype0]);
                    let allele1 =
                        rng.next_f64() <= f64::from(missing_probabilities[start + haplotype1]);
                    set_haplotype(variants, absolute_locus, 0, allele0);
                    set_haplotype(variants, absolute_locus, 1, allele1);
                }
                missing_index += 1;
            }
            if code > 1 {
                let graph = ambiguous[ambiguous_index];
                set_haplotype(
                    variants,
                    absolute_locus,
                    0,
                    ((graph >> haplotype0) & 1) != 0,
                );
                set_haplotype(
                    variants,
                    absolute_locus,
                    1,
                    ((graph >> haplotype1) & 1) != 0,
                );
                ambiguous_index += 1;
            }
            absolute_locus += 1;
        }
    }
}

fn solve_path(
    diplotypes: &[u64],
    transition_count: usize,
    stored_transition_indexes: &[u32],
    stored_transition_probabilities: &[f32],
) -> Vec<u8> {
    let mut maximum_probabilities: Vec<Vec<f64>> = Vec::with_capacity(diplotypes.len());
    let mut maximum_indexes: Vec<Vec<usize>> = Vec::with_capacity(diplotypes.len());
    let mut previous_count = 1usize;
    let mut transition_offset = 0usize;
    let mut stored_relative = 0usize;

    for (segment, &diplotype) in diplotypes.iter().enumerate() {
        let current_count = diplotype.count_ones() as usize;
        let mut probabilities = vec![0.0f64; current_count];
        let mut indexes = vec![0usize; current_count];
        for relative_transition in 0..previous_count * current_count {
            let previous = relative_transition / current_count;
            let current = relative_transition % current_count;
            let absolute_transition = transition_offset + relative_transition;
            let stored = stored_relative < stored_transition_indexes.len()
                && stored_transition_indexes[stored_relative] as usize == absolute_transition;
            let transition_probability = if stored {
                let value = f64::from(stored_transition_probabilities[stored_relative]);
                stored_relative += 1;
                value
            } else {
                1e-6
            };
            let previous_probability = if segment == 0 {
                1.0
            } else {
                maximum_probabilities[segment - 1][previous]
            };
            let probability = previous_probability * transition_probability;
            if probability > probabilities[current] {
                probabilities[current] = probability;
                indexes[current] = previous;
            }
        }
        let mut total = 0.0f64;
        for &probability in &probabilities {
            total += probability;
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        maximum_probabilities.push(probabilities);
        maximum_indexes.push(indexes);
        transition_offset += previous_count * current_count;
        previous_count = current_count;
    }
    debug_assert_eq!(transition_offset, transition_count);
    debug_assert_eq!(stored_relative, stored_transition_indexes.len());

    let final_probabilities = maximum_probabilities.last().unwrap();
    let mut best = 0usize;
    for index in 1..final_probabilities.len() {
        if final_probabilities[index] > final_probabilities[best] {
            best = index;
        }
    }
    let mut sampled = vec![0u8; diplotypes.len()];
    let final_segment = diplotypes.len() - 1;
    sampled[final_segment] = diplotype_code(diplotypes[final_segment], best);
    for segment in (0..final_segment).rev() {
        best = maximum_indexes[segment + 1][best];
        sampled[segment] = diplotype_code(diplotypes[segment], best);
    }
    sampled
}

struct ApplySolution<'a> {
    variants: &'a mut [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    sampled: &'a [u8],
    missing_probabilities: &'a [f32],
    storage_events: u32,
    haploid: bool,
}

fn apply_solution(parameters: ApplySolution<'_>) {
    let ApplySolution {
        variants,
        ambiguous,
        segment_lengths,
        sampled,
        missing_probabilities,
        storage_events,
        haploid,
    } = parameters;
    let event_count = storage_events as f32;
    let mut absolute_locus = 0usize;
    let mut ambiguous_index = 0usize;
    let mut missing_index = 0usize;
    for (segment, &length) in segment_lengths.iter().enumerate() {
        let haplotype0 = usize::from(sampled[segment] >> 3);
        let haplotype1 = usize::from(sampled[segment] & 7);
        for _ in 0..length {
            let code = graph_code(variant_nibble(variants, absolute_locus));
            if code == 1 {
                let start = missing_index * 8;
                let probability0 = missing_probabilities[start + haplotype0];
                let probability1 = missing_probabilities[start + haplotype1];
                if haploid {
                    let normalized0 = probability0 / event_count;
                    let normalized1 = probability1 / event_count;
                    let probability00 = (1.0 - normalized0) * (1.0 - normalized1);
                    let probability11 = normalized0 * normalized1;
                    let allele = probability11 > probability00;
                    set_haplotype(variants, absolute_locus, 0, allele);
                    set_haplotype(variants, absolute_locus, 1, allele);
                } else {
                    let threshold = 0.5 * event_count;
                    set_haplotype(variants, absolute_locus, 0, probability0 >= threshold);
                    set_haplotype(variants, absolute_locus, 1, probability1 >= threshold);
                }
                missing_index += 1;
            }
            if code > 1 {
                let graph = ambiguous[ambiguous_index];
                set_haplotype(
                    variants,
                    absolute_locus,
                    0,
                    ((graph >> haplotype0) & 1) != 0,
                );
                set_haplotype(
                    variants,
                    absolute_locus,
                    1,
                    ((graph >> haplotype1) & 1) != 0,
                );
                ambiguous_index += 1;
            }
            absolute_locus += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct SegmentCoordinates {
    locus_start: usize,
    locus_count: usize,
    ambiguous_start: usize,
    ambiguous_count: usize,
    missing_start: usize,
    missing_count: usize,
    transition_start: usize,
    transition_count: usize,
    start_centimorgans: f64,
    stop_centimorgans: f64,
}

fn segment_coordinates(
    variants: &[u8],
    variant_count: usize,
    diplotypes: &[u64],
    segment_lengths: &[u16],
    segment_start_centimorgans: &[f64],
    segment_stop_centimorgans: &[f64],
) -> Result<Vec<SegmentCoordinates>, u32> {
    if diplotypes.is_empty()
        || diplotypes.len() != segment_lengths.len()
        || diplotypes.len() != segment_start_centimorgans.len()
        || diplotypes.len() != segment_stop_centimorgans.len()
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let mut coordinates = Vec::with_capacity(diplotypes.len());
    let mut locus = 0usize;
    let mut ambiguous = 0usize;
    let mut missing = 0usize;
    let mut transition = 0usize;
    let mut previous_diplotypes = 1usize;
    for segment in 0..diplotypes.len() {
        let locus_count = usize::from(segment_lengths[segment]);
        if locus_count == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let stop = locus
            .checked_add(locus_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if stop > variant_count {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
        let mut ambiguous_count = 0usize;
        let mut missing_count = 0usize;
        for absolute_locus in locus..stop {
            match graph_code(variant_nibble(variants, absolute_locus)) {
                1 => missing_count += 1,
                2 | 3 => ambiguous_count += 1,
                _ => {}
            }
        }
        let current_diplotypes = diplotypes[segment].count_ones() as usize;
        if current_diplotypes == 0 {
            return Err(STATUS_INVALID_DIMENSIONS);
        }
        let transition_count = previous_diplotypes
            .checked_mul(current_diplotypes)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        coordinates.push(SegmentCoordinates {
            locus_start: locus,
            locus_count,
            ambiguous_start: ambiguous,
            ambiguous_count,
            missing_start: missing,
            missing_count,
            transition_start: transition,
            transition_count,
            start_centimorgans: segment_start_centimorgans[segment],
            stop_centimorgans: segment_stop_centimorgans[segment],
        });
        locus = stop;
        ambiguous = ambiguous
            .checked_add(ambiguous_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        missing = missing
            .checked_add(missing_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        transition = transition
            .checked_add(transition_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        previous_diplotypes = current_diplotypes;
    }
    if locus != variant_count {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    Ok(coordinates)
}

fn split_windows(
    minimum_centimorgans: f64,
    left: usize,
    right: usize,
    segments: &[SegmentCoordinates],
    rng: &mut LogicalRng,
) -> Option<Vec<(usize, usize)>> {
    let number_of_segments = right - left + 1;
    let number_of_variants =
        segments[right].locus_start + segments[right].locus_count - segments[left].locus_start;
    let length_centimorgans = segments[right].stop_centimorgans - segments[left].start_centimorgans;
    if number_of_segments < 4
        || number_of_variants < 100
        || length_centimorgans < minimum_centimorgans
    {
        return None;
    }

    let random_span = (number_of_segments / 2) as u32;
    let split = rng.next_bounded(random_span) as usize + number_of_segments / 4 + 1;
    let left_windows = split_windows(minimum_centimorgans, left, left + split, segments, rng);
    let right_windows = split_windows(minimum_centimorgans, left + split, right, segments, rng);
    match (left_windows, right_windows) {
        (Some(mut left_windows), Some(right_windows)) => {
            left_windows.extend(right_windows);
            Some(left_windows)
        }
        _ => Some(vec![(left, right)]),
    }
}

fn signed_stop(start: usize, count: usize) -> Result<i32, u32> {
    let stop = i64::try_from(start)
        .map_err(|_| STATUS_INTEGER_OVERFLOW)?
        .checked_add(i64::try_from(count).map_err(|_| STATUS_INTEGER_OVERFLOW)?)
        .and_then(|value| value.checked_sub(1))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    i32::try_from(stop).map_err(|_| STATUS_INTEGER_OVERFLOW)
}

fn output_window(
    start: usize,
    stop: usize,
    segments: &[SegmentCoordinates],
) -> Result<GenotypeWindowV1, u32> {
    let start_segment = segments[start];
    let stop_segment = segments[stop];
    let as_i32 = |value: usize| i32::try_from(value).map_err(|_| STATUS_INTEGER_OVERFLOW);
    Ok(GenotypeWindowV1 {
        start_locus: as_i32(start_segment.locus_start)?,
        start_segment: as_i32(start)?,
        start_ambiguous: as_i32(start_segment.ambiguous_start)?,
        start_missing: as_i32(start_segment.missing_start)?,
        start_transition: as_i32(
            start_segment
                .transition_start
                .checked_add(start_segment.transition_count)
                .ok_or(STATUS_INTEGER_OVERFLOW)?,
        )?,
        stop_locus: signed_stop(stop_segment.locus_start, stop_segment.locus_count)?,
        stop_segment: as_i32(stop)?,
        stop_ambiguous: signed_stop(stop_segment.ambiguous_start, stop_segment.ambiguous_count)?,
        stop_missing: signed_stop(stop_segment.missing_start, stop_segment.missing_count)?,
        stop_transition: signed_stop(stop_segment.transition_start, stop_segment.transition_count)?,
    })
}

pub(crate) struct WindowInputs<'a> {
    pub(crate) variants: &'a [u8],
    pub(crate) variant_count: usize,
    pub(crate) diplotypes: &'a [u64],
    pub(crate) segment_lengths: &'a [u16],
    pub(crate) segment_start_centimorgans: &'a [f64],
    pub(crate) segment_stop_centimorgans: &'a [f64],
    pub(crate) minimum_window_centimorgans: f32,
}

pub(crate) fn build_windows(
    inputs: WindowInputs<'_>,
    rng: &mut LogicalRng,
) -> Result<Vec<GenotypeWindowV1>, u32> {
    let segments = segment_coordinates(
        inputs.variants,
        inputs.variant_count,
        inputs.diplotypes,
        inputs.segment_lengths,
        inputs.segment_start_centimorgans,
        inputs.segment_stop_centimorgans,
    )?;
    if segments.len() > i32::MAX as usize {
        return Err(STATUS_INTEGER_OVERFLOW);
    }
    let ranges = split_windows(
        f64::from(inputs.minimum_window_centimorgans),
        0,
        segments.len() - 1,
        &segments,
        rng,
    )
    .unwrap_or_else(|| vec![(0, segments.len() - 1)]);
    let mut output = Vec::with_capacity(ranges.len());
    for (start, stop) in ranges {
        output.push(output_window(start, stop, &segments)?);
    }
    Ok(output)
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

#[no_mangle]
/// Build and retain one complete genotype graph in Rust-owned storage.
///
/// # Safety
///
/// `variants` must be readable for the packed variant length and `graph` must
/// be writable for one pointer. The returned graph must eventually be freed by
/// `shapeit_genotype_graph_free_v1`.
pub unsafe extern "C" fn shapeit_genotype_graph_create_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    graph: *mut *mut GenotypeGraphV1,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    *graph = core::ptr::null_mut();
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
    *graph = Box::into_raw(Box::new(GenotypeGraphV1::new(variants, variant_count)));
    STATUS_OK
}

#[no_mangle]
/// Allocate zeroed packed variants in a Rust-owned, not-yet-built graph.
///
/// # Safety
///
/// `graph` must be writable for one pointer. The returned graph must eventually
/// be freed by `shapeit_genotype_graph_free_v1`.
pub unsafe extern "C" fn shapeit_genotype_graph_allocate_v1(
    variant_count: usize,
    graph: *mut *mut GenotypeGraphV1,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    *graph = core::ptr::null_mut();
    let variants_length = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    *graph = Box::into_raw(Box::new(GenotypeGraphV1::allocate(
        variant_count,
        variants_length,
    )));
    STATUS_OK
}

#[no_mangle]
/// Borrow the stable mutable packed-variant allocation for a graph's lifetime.
///
/// The allocation is never resized. Callers must not access it concurrently
/// with a mutable graph operation.
///
/// # Safety
///
/// `graph` must be live and both output pointers writable.
pub unsafe extern "C" fn shapeit_genotype_graph_variants_mut_v1(
    graph: *mut GenotypeGraphV1,
    variants: *mut *mut u8,
    variants_length: *mut usize,
) -> u32 {
    if graph.is_null() || variants.is_null() || variants_length.is_null() {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    *variants = graph.variants.as_mut_ptr();
    *variants_length = graph.variants.len();
    STATUS_OK
}

#[no_mangle]
/// Set whether this sample follows haploid sampling and solving rules.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed.
pub unsafe extern "C" fn shapeit_genotype_graph_set_haploid_v1(
    graph: *mut GenotypeGraphV1,
    haploid: u8,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    if haploid > 1 {
        return STATUS_INVALID_DIMENSIONS;
    }
    (*graph).haploid = haploid != 0;
    STATUS_OK
}

#[no_mangle]
/// Return persistent per-sample graph flags.
///
/// # Safety
///
/// `graph` must be live and both flag pointers writable.
pub unsafe extern "C" fn shapeit_genotype_graph_flags_v1(
    graph: *const GenotypeGraphV1,
    haploid: *mut u8,
    double_precision: *mut u8,
) -> u32 {
    if graph.is_null() || haploid.is_null() || double_precision.is_null() {
        return STATUS_NULL_POINTER;
    }
    *haploid = u8::from((*graph).haploid);
    *double_precision = u8::from((*graph).double_precision);
    STATUS_OK
}

#[no_mangle]
/// Persist single-precision underflow recovery for later HMM windows.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed.
pub unsafe extern "C" fn shapeit_genotype_graph_require_double_v1(
    graph: *mut GenotypeGraphV1,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    (*graph).double_precision = true;
    STATUS_OK
}

#[no_mangle]
/// Finalize all derived graph arrays from previously populated packed variants.
///
/// # Safety
///
/// `graph` must be a live, exclusively borrowed graph returned by the allocate
/// function and may be finalized only once.
pub unsafe extern "C" fn shapeit_genotype_graph_build_in_place_v1(
    graph: *mut GenotypeGraphV1,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    match (*graph).build_in_place() {
        Ok(()) => STATUS_OK,
        Err(status) => status,
    }
}

#[no_mangle]
/// Borrow graph buffers until the next mutable graph call.
///
/// # Safety
///
/// `graph` must be a live Rust-owned graph and `view` writable for one view.
pub unsafe extern "C" fn shapeit_genotype_graph_borrow_v1(
    graph: *const GenotypeGraphV1,
    view: *mut GenotypeGraphViewV1,
) -> u32 {
    if graph.is_null() || view.is_null() {
        return STATUS_NULL_POINTER;
    }
    if !(*graph).built {
        return STATUS_INVALID_DIMENSIONS;
    }
    *view = (*graph).view();
    STATUS_OK
}

#[no_mangle]
/// Release a Rust-owned genotype graph. A null pointer is accepted.
///
/// # Safety
///
/// A non-null pointer must have been returned by
/// `shapeit_genotype_graph_create_v1` and not already freed.
pub unsafe extern "C" fn shapeit_genotype_graph_free_v1(graph: *mut GenotypeGraphV1) {
    if !graph.is_null() {
        drop(Box::from_raw(graph));
    }
}

#[no_mangle]
/// Apply trio or duo Mendelian scaffolding to one packed child genotype.
///
/// # Safety
///
/// Required parent buffers must be readable for the packed variant length,
/// `child_variants` must be writable for that length, and `counts` must be
/// writable for four `u32` values. Mutable buffers must not overlap a parent.
pub unsafe extern "C" fn shapeit_genotype_pedigree_scaffold_v1(
    child_variants: *mut u8,
    child_variants_length: usize,
    variant_count: usize,
    father_variants: *const u8,
    father_variants_length: usize,
    mother_variants: *const u8,
    mother_variants_length: usize,
    pedigree_mode: u32,
    counts: *mut u32,
    counts_length: usize,
) -> u32 {
    if counts.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required > child_variants_length || counts_length < 4 {
        return STATUS_OUT_OF_BOUNDS;
    }
    if required != 0 && child_variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let mode = match pedigree_mode {
        0 => PedigreeMode::Trio,
        1 => PedigreeMode::Father,
        2 => PedigreeMode::Mother,
        _ => return STATUS_INVALID_DIMENSIONS,
    };
    let father_required = matches!(mode, PedigreeMode::Trio | PedigreeMode::Father);
    let mother_required = matches!(mode, PedigreeMode::Trio | PedigreeMode::Mother);
    if required != 0
        && ((father_required && father_variants.is_null())
            || (mother_required && mother_variants.is_null()))
    {
        return STATUS_NULL_POINTER;
    }
    if (father_required && required > father_variants_length)
        || (mother_required && required > mother_variants_length)
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    let counts_bytes = 4 * core::mem::size_of::<u32>();
    let child = child_variants.cast_const();
    let counts_bytes_pointer = counts.cast::<u8>().cast_const();
    let child_counts_overlap =
        byte_ranges_overlap(child, required, counts_bytes_pointer, counts_bytes);
    let father_child_overlap = if father_required {
        byte_ranges_overlap(child, required, father_variants, required)
    } else {
        Ok(false)
    };
    let mother_child_overlap = if mother_required {
        byte_ranges_overlap(child, required, mother_variants, required)
    } else {
        Ok(false)
    };
    let father_counts_overlap = if father_required {
        byte_ranges_overlap(
            father_variants,
            required,
            counts_bytes_pointer,
            counts_bytes,
        )
    } else {
        Ok(false)
    };
    let mother_counts_overlap = if mother_required {
        byte_ranges_overlap(
            mother_variants,
            required,
            counts_bytes_pointer,
            counts_bytes,
        )
    } else {
        Ok(false)
    };
    for overlap in [
        child_counts_overlap,
        father_child_overlap,
        mother_child_overlap,
        father_counts_overlap,
        mother_counts_overlap,
    ] {
        match overlap {
            Ok(true) => return STATUS_INVALID_DIMENSIONS,
            Err(status) => return status,
            Ok(false) => {}
        }
    }

    let child = if required == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(child_variants, required)
    };
    let father = if father_required {
        Some(if required == 0 {
            &[]
        } else {
            slice::from_raw_parts(father_variants, required)
        })
    } else {
        None
    };
    let mother = if mother_required {
        Some(if required == 0 {
            &[]
        } else {
            slice::from_raw_parts(mother_variants, required)
        })
    } else {
        None
    };
    let counts = slice::from_raw_parts_mut(counts, 4);
    scaffold_pedigree(child, father, mother, variant_count, mode, counts);
    STATUS_OK
}

#[no_mangle]
/// Preserve the established packed-bit reset of ambiguous haploid genotypes.
///
/// # Safety
///
/// `variants` must be writable for the packed variant length and `reset_count`
/// must be writable for one `u32`; the two outputs must not overlap.
pub unsafe extern "C" fn shapeit_genotype_reset_haploid_hets_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    reset_count: *mut u32,
) -> u32 {
    if reset_count.is_null() {
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
    match byte_ranges_overlap(
        variants.cast_const(),
        required,
        reset_count.cast::<u8>().cast_const(),
        core::mem::size_of::<u32>(),
    ) {
        Ok(true) => return STATUS_INVALID_DIMENSIONS,
        Err(status) => return status,
        Ok(false) => {}
    }

    let variants = if required == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(variants, required)
    };
    let mut count = 0u32;
    for locus in 0..variant_count {
        let code = variant_nibble(variants, locus);
        if graph_code(code) > 1 {
            set_variant_nibble(variants, locus, code | 1);
            count = count.wrapping_add(1);
        }
    }
    *reset_count = count;
    STATUS_OK
}

#[no_mangle]
/// Select and perform one complete round of genotype-graph pruning.
///
/// Equal transition probabilities and equal segment entropies are ordered by
/// their original indexes, making the result independent of sort implementation
/// details. The output capacities may equal their corresponding input lengths.
///
/// # Safety
///
/// Input buffers must be readable and output buffers writable for their stated
/// lengths. No output buffer may overlap an input or another output buffer.
pub unsafe extern "C" fn shapeit_genotype_prune_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    threshold_probability_mass: f64,
    output_ambiguous: *mut u8,
    output_ambiguous_length: usize,
    output_diplotypes: *mut u64,
    output_diplotypes_capacity: usize,
    output_segment_lengths: *mut u16,
    output_segment_lengths_capacity: usize,
    output_segment_count: *mut usize,
    output_transition_count: *mut u32,
) -> u32 {
    if output_segment_count.is_null() || output_transition_count.is_null() {
        return STATUS_NULL_POINTER;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if diplotypes_length == 0 || diplotypes_length != segment_lengths_length {
        return STATUS_INVALID_DIMENSIONS;
    }
    if output_ambiguous_length < ambiguous_length
        || output_diplotypes_capacity < diplotypes_length
        || output_segment_lengths_capacity < diplotypes_length
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    if (required_variants != 0 && variants.is_null())
        || (ambiguous_length != 0 && ambiguous.is_null())
        || diplotypes.is_null()
        || segment_lengths.is_null()
        || (transition_probabilities_length != 0 && transition_probabilities.is_null())
        || (ambiguous_length != 0 && output_ambiguous.is_null())
        || output_diplotypes.is_null()
        || output_segment_lengths.is_null()
    {
        return STATUS_NULL_POINTER;
    }

    let variants = if required_variants == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, required_variants)
    };
    let ambiguous = if ambiguous_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts(diplotypes, diplotypes_length);
    let segment_lengths = slice::from_raw_parts(segment_lengths, segment_lengths_length);
    let transition_probabilities = if transition_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(transition_probabilities, transition_probabilities_length)
    };
    let mut loci = 0usize;
    for &length in segment_lengths {
        if length == 0 {
            return STATUS_INVALID_DIMENSIONS;
        }
        loci = match loci.checked_add(usize::from(length)) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
    }
    if loci != variant_count {
        return STATUS_INVALID_DIMENSIONS;
    }
    let expected_ambiguous = (0..variant_count)
        .filter(|&locus| graph_code(variant_nibble(variants, locus)) > 1)
        .count();
    if expected_ambiguous != ambiguous_length {
        return STATUS_INVALID_DIMENSIONS;
    }
    let expected_transitions = match count_graph_transitions(diplotypes) {
        Ok(value) => value as usize,
        Err(status) => return status,
    };
    if expected_transitions != transition_probabilities_length {
        return STATUS_INVALID_DIMENSIONS;
    }
    if transition_probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return STATUS_INVALID_DIMENSIONS;
    }

    let output = match prune_graph(
        variants,
        ambiguous,
        diplotypes,
        segment_lengths,
        transition_probabilities,
        threshold_probability_mass,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if !output.ambiguous.is_empty() {
        let destination = slice::from_raw_parts_mut(output_ambiguous, output_ambiguous_length);
        destination[..output.ambiguous.len()].copy_from_slice(&output.ambiguous);
    }
    let output_diplotype_slice =
        slice::from_raw_parts_mut(output_diplotypes, output_diplotypes_capacity);
    output_diplotype_slice[..output.diplotypes.len()].copy_from_slice(&output.diplotypes);
    let output_length_slice =
        slice::from_raw_parts_mut(output_segment_lengths, output_segment_lengths_capacity);
    output_length_slice[..output.segment_lengths.len()].copy_from_slice(&output.segment_lengths);
    *output_segment_count = output.diplotypes.len();
    *output_transition_count = output.transition_count;
    STATUS_OK
}

#[no_mangle]
/// Prune a Rust-owned graph in place.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed for this call. Transition
/// probabilities must be readable for their stated length.
pub unsafe extern "C" fn shapeit_genotype_graph_prune_v1(
    graph: *mut GenotypeGraphV1,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    threshold_probability_mass: f64,
) -> u32 {
    if graph.is_null()
        || (transition_probabilities_length != 0 && transition_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    if !graph.built {
        return STATUS_INVALID_DIMENSIONS;
    }
    if transition_probabilities_length != graph.transition_count as usize {
        return STATUS_INVALID_DIMENSIONS;
    }
    let transition_probabilities = if transition_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(transition_probabilities, transition_probabilities_length)
    };
    if transition_probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let output = match prune_graph(
        &graph.variants,
        &graph.ambiguous,
        &graph.diplotypes,
        &graph.segment_lengths,
        transition_probabilities,
        threshold_probability_mass,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    graph.ambiguous = output.ambiguous;
    graph.diplotypes = output.diplotypes;
    graph.segment_lengths = output.segment_lengths;
    graph.transition_count = output.transition_count;
    STATUS_OK
}

#[no_mangle]
/// Sample one complete genotype graph and update its packed haplotype alleles.
///
/// RNG coordinates identify a fresh logical Philox stream. The function
/// consumes that stream in the same order as the established forward/backward
/// sampler and missing-genotype imputation.
///
/// # Safety
///
/// Every buffer must be valid for its stated length. The mutable packed-variant
/// buffer must not overlap any immutable input. Invalid layouts are rejected
/// before any allele is modified.
pub unsafe extern "C" fn shapeit_genotype_sample_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    haploid: u8,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> u32 {
    if variants.is_null()
        || diplotypes.is_null()
        || segment_lengths.is_null()
        || transition_probabilities.is_null()
        || (ambiguous_length != 0 && ambiguous.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    if haploid > 1 {
        return STATUS_INVALID_DIMENSIONS;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variants = slice::from_raw_parts_mut(variants, variants_length);
    let ambiguous = if ambiguous_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts(diplotypes, diplotypes_length);
    let segment_lengths = slice::from_raw_parts(segment_lengths, segment_lengths_length);
    let transition_probabilities =
        slice::from_raw_parts(transition_probabilities, transition_probabilities_length);
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    let layout = match validate_sample_layout(
        variants,
        variant_count,
        ambiguous.len(),
        diplotypes,
        segment_lengths,
        transition_probabilities.len(),
        missing_probabilities.len(),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let missing_probabilities = &missing_probabilities[..layout.missing * 8];

    match sample_validated(
        SampleParameters {
            variants,
            ambiguous,
            diplotypes,
            segment_lengths,
            transition_probabilities: &transition_probabilities[..layout.transitions],
            missing_probabilities,
            haploid: haploid != 0,
        },
        seed,
        domain,
        iteration,
        item,
    ) {
        Ok(()) => STATUS_OK,
        Err(error) => error.status(),
    }
}

pub(crate) fn sample_graph_current(
    graph: &mut GenotypeGraphV1,
    transition_probabilities: &[f64],
    missing_probabilities: &[f32],
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> Result<(), SampleError> {
    if !graph.built {
        return Err(SampleError::Status(STATUS_INVALID_DIMENSIONS));
    }
    let layout = validate_sample_layout(
        &graph.variants,
        graph.variant_count,
        graph.ambiguous.len(),
        &graph.diplotypes,
        &graph.segment_lengths,
        transition_probabilities.len(),
        missing_probabilities.len(),
    )
    .map_err(SampleError::Status)?;
    sample_validated(
        SampleParameters {
            variants: &mut graph.variants,
            ambiguous: &graph.ambiguous,
            diplotypes: &graph.diplotypes,
            segment_lengths: &graph.segment_lengths,
            transition_probabilities: &transition_probabilities[..layout.transitions],
            missing_probabilities: &missing_probabilities[..layout.missing * 8],
            haploid: graph.haploid,
        },
        seed,
        domain,
        iteration,
        item,
    )
}

#[no_mangle]
/// Sample a Rust-owned graph in place.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed. Probability buffers must be
/// readable for their stated lengths.
pub unsafe extern "C" fn shapeit_genotype_graph_sample_v1(
    graph: *mut GenotypeGraphV1,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    haploid: u8,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    if !graph.built {
        return STATUS_INVALID_DIMENSIONS;
    }
    shapeit_genotype_sample_v1(
        graph.variants.as_mut_ptr(),
        graph.variants.len(),
        graph.variant_count,
        graph.ambiguous.as_ptr(),
        graph.ambiguous.len(),
        graph.diplotypes.as_ptr(),
        graph.diplotypes.len(),
        graph.segment_lengths.as_ptr(),
        graph.segment_lengths.len(),
        transition_probabilities,
        transition_probabilities_length,
        missing_probabilities,
        missing_probabilities_length,
        haploid,
        seed,
        domain,
        iteration,
        item,
    )
}

#[no_mangle]
/// Sample using the haploid state retained by a Rust-owned graph.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed. Probability buffers must be
/// readable for their stated lengths.
pub unsafe extern "C" fn shapeit_genotype_graph_sample_current_v1(
    graph: *mut GenotypeGraphV1,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    let haploid = u8::from((*graph).haploid);
    shapeit_genotype_graph_sample_v1(
        graph,
        transition_probabilities,
        transition_probabilities_length,
        missing_probabilities,
        missing_probabilities_length,
        haploid,
        seed,
        domain,
        iteration,
        item,
    )
}

#[no_mangle]
/// Select the maximum-probability stored path and apply it to packed alleles.
///
/// Stored transition indexes must be strictly increasing and correspond
/// one-for-one with the compressed stored probabilities.
///
/// # Safety
///
/// Every buffer must be valid for its stated length. The mutable packed-variant
/// buffer must not overlap any immutable input. Invalid layouts are rejected
/// before any allele is modified.
pub unsafe extern "C" fn shapeit_genotype_solve_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    stored_transition_indexes: *const u32,
    stored_transition_indexes_length: usize,
    stored_transition_probabilities: *const f32,
    stored_transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
    haploid: u8,
    storage_events: u32,
) -> u32 {
    if diplotypes.is_null()
        || segment_lengths.is_null()
        || (stored_transition_indexes_length != 0 && stored_transition_indexes.is_null())
        || (stored_transition_probabilities_length != 0
            && stored_transition_probabilities.is_null())
        || (ambiguous_length != 0 && ambiguous.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    if haploid > 1
        || storage_events == 0
        || stored_transition_indexes_length != stored_transition_probabilities_length
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    if variants_length != 0 && variants.is_null() {
        return STATUS_NULL_POINTER;
    }
    let variants = if variants_length == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(variants, variants_length)
    };
    let ambiguous = if ambiguous_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(ambiguous, ambiguous_length)
    };
    let diplotypes = slice::from_raw_parts(diplotypes, diplotypes_length);
    let segment_lengths = slice::from_raw_parts(segment_lengths, segment_lengths_length);
    let stored_transition_indexes = if stored_transition_indexes_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(stored_transition_indexes, stored_transition_indexes_length)
    };
    let stored_transition_probabilities = if stored_transition_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(
            stored_transition_probabilities,
            stored_transition_probabilities_length,
        )
    };
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    let layout = match validate_sample_layout(
        variants,
        variant_count,
        ambiguous.len(),
        diplotypes,
        segment_lengths,
        usize::MAX,
        missing_probabilities.len(),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if layout.transitions > u32::MAX as usize {
        return STATUS_INTEGER_OVERFLOW;
    }
    let mut previous = None;
    for &index in stored_transition_indexes {
        if index as usize >= layout.transitions || previous.is_some_and(|value| index <= value) {
            return STATUS_INVALID_DIMENSIONS;
        }
        previous = Some(index);
    }

    let sampled = solve_path(
        diplotypes,
        layout.transitions,
        stored_transition_indexes,
        stored_transition_probabilities,
    );
    apply_solution(ApplySolution {
        variants,
        ambiguous,
        segment_lengths,
        sampled: &sampled,
        missing_probabilities: &missing_probabilities[..layout.missing * 8],
        storage_events,
        haploid: haploid != 0,
    });
    STATUS_OK
}

#[no_mangle]
/// Accumulate one main-iteration probability set in Rust-owned storage.
///
/// A null storage pointer allocates the persistent mask from transition
/// probabilities at or above `1e-6`. Later calls preserve that mask and apply
/// the established float accumulation order.
///
/// # Safety
///
/// `storage` must be writable and contain null or a live storage object from
/// this function. Non-empty probability buffers must be readable.
pub unsafe extern "C" fn shapeit_genotype_storage_update_v1(
    storage: *mut *mut GenotypeStorageV1,
    transition_probabilities: *const f64,
    transition_count: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
) -> u32 {
    if storage.is_null()
        || (transition_count != 0 && transition_probabilities.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let transition_probabilities = if transition_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(transition_probabilities, transition_count)
    };
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    if (*storage).is_null() {
        let value = match GenotypeStorageV1::new(transition_probabilities, missing_probabilities) {
            Ok(value) => value,
            Err(status) => return status,
        };
        *storage = Box::into_raw(Box::new(value));
        STATUS_OK
    } else {
        match (*(*storage)).update(transition_probabilities, missing_probabilities) {
            Ok(()) => STATUS_OK,
            Err(status) => status,
        }
    }
}

#[no_mangle]
/// Free Rust-owned genotype probability storage. Null is accepted.
///
/// # Safety
///
/// `storage` must be null or a live pointer returned by the update function,
/// and it must be freed at most once.
pub unsafe extern "C" fn shapeit_genotype_storage_free_v1(storage: *mut GenotypeStorageV1) {
    if !storage.is_null() {
        drop(Box::from_raw(storage));
    }
}

#[no_mangle]
/// Borrow the serialized mask and accumulated probability arrays.
///
/// # Safety
///
/// `storage` must be live and `view` writable. Borrowed pointers remain valid
/// only until the next storage update or free.
pub unsafe extern "C" fn shapeit_genotype_storage_borrow_v1(
    storage: *const GenotypeStorageV1,
    view: *mut GenotypeStorageViewV1,
) -> u32 {
    if storage.is_null() || view.is_null() {
        return STATUS_NULL_POINTER;
    }
    *view = (*storage).view();
    STATUS_OK
}

#[no_mangle]
/// Accumulate one probability set in storage owned by a Rust genotype graph.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed. Probability buffers must be
/// readable for their stated lengths.
pub unsafe extern "C" fn shapeit_genotype_graph_store_v1(
    graph: *mut GenotypeGraphV1,
    transition_probabilities: *const f64,
    transition_probabilities_length: usize,
    missing_probabilities: *const f32,
    missing_probabilities_length: usize,
) -> u32 {
    if graph.is_null()
        || (transition_probabilities_length != 0 && transition_probabilities.is_null())
        || (missing_probabilities_length != 0 && missing_probabilities.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    if !graph.built {
        return STATUS_INVALID_DIMENSIONS;
    }
    let expected_missing = match graph.missing_count.checked_mul(8) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if transition_probabilities_length != graph.transition_count as usize
        || missing_probabilities_length != expected_missing
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let transition_probabilities = if transition_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(transition_probabilities, transition_probabilities_length)
    };
    let missing_probabilities = if missing_probabilities_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(missing_probabilities, missing_probabilities_length)
    };
    match graph.storage.as_mut() {
        Some(storage) => match storage.update(transition_probabilities, missing_probabilities) {
            Ok(()) => STATUS_OK,
            Err(status) => status,
        },
        None => match GenotypeStorageV1::new(transition_probabilities, missing_probabilities) {
            Ok(storage) => {
                graph.storage = Some(storage);
                STATUS_OK
            }
            Err(status) => status,
        },
    }
}

#[no_mangle]
/// Borrow probability storage retained by a Rust genotype graph.
///
/// # Safety
///
/// `graph` must be live and `view` writable. Returned pointers remain valid
/// until the next mutable graph call.
pub unsafe extern "C" fn shapeit_genotype_graph_storage_borrow_v1(
    graph: *const GenotypeGraphV1,
    view: *mut GenotypeStorageViewV1,
) -> u32 {
    if graph.is_null() || view.is_null() {
        return STATUS_NULL_POINTER;
    }
    let storage = match (*graph).storage.as_ref() {
        Some(value) => value,
        None => return STATUS_INVALID_DIMENSIONS,
    };
    *view = storage.view();
    STATUS_OK
}

#[no_mangle]
/// Solve a genotype graph directly from Rust-owned accumulated storage.
///
/// # Safety
///
/// Graph buffers follow `shapeit_genotype_solve_v1`; `storage` must be live and
/// must not overlap the graph buffers.
pub unsafe extern "C" fn shapeit_genotype_solve_storage_v1(
    variants: *mut u8,
    variants_length: usize,
    variant_count: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    storage: *const GenotypeStorageV1,
    haploid: u8,
) -> u32 {
    if storage.is_null() {
        return STATUS_NULL_POINTER;
    }
    let storage = &*storage;
    shapeit_genotype_solve_v1(
        variants,
        variants_length,
        variant_count,
        ambiguous,
        ambiguous_length,
        diplotypes,
        diplotypes_length,
        segment_lengths,
        segment_lengths_length,
        storage.transition_indexes.as_ptr(),
        storage.transition_indexes.len(),
        storage.transition_probabilities.as_ptr(),
        storage.transition_probabilities.len(),
        storage.missing_probabilities.as_ptr(),
        storage.missing_probabilities.len(),
        haploid,
        storage.storage_events,
    )
}

#[no_mangle]
/// Solve a Rust-owned genotype graph from Rust-owned accumulated storage.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed; `storage` must be live.
pub unsafe extern "C" fn shapeit_genotype_graph_solve_storage_v1(
    graph: *mut GenotypeGraphV1,
    storage: *const GenotypeStorageV1,
    haploid: u8,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    if !graph.built {
        return STATUS_INVALID_DIMENSIONS;
    }
    shapeit_genotype_solve_storage_v1(
        graph.variants.as_mut_ptr(),
        graph.variants.len(),
        graph.variant_count,
        graph.ambiguous.as_ptr(),
        graph.ambiguous.len(),
        graph.diplotypes.as_ptr(),
        graph.diplotypes.len(),
        graph.segment_lengths.as_ptr(),
        graph.segment_lengths.len(),
        storage,
        haploid,
    )
}

#[no_mangle]
/// Solve a Rust-owned genotype graph from its internally retained storage.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed.
pub unsafe extern "C" fn shapeit_genotype_graph_solve_v1(
    graph: *mut GenotypeGraphV1,
    haploid: u8,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    let graph = &mut *graph;
    if !graph.built {
        return STATUS_INVALID_DIMENSIONS;
    }
    let storage = match graph.storage.as_ref() {
        Some(value) => value,
        None => return STATUS_INVALID_DIMENSIONS,
    };
    shapeit_genotype_solve_v1(
        graph.variants.as_mut_ptr(),
        graph.variants.len(),
        graph.variant_count,
        graph.ambiguous.as_ptr(),
        graph.ambiguous.len(),
        graph.diplotypes.as_ptr(),
        graph.diplotypes.len(),
        graph.segment_lengths.as_ptr(),
        graph.segment_lengths.len(),
        storage.transition_indexes.as_ptr(),
        storage.transition_indexes.len(),
        storage.transition_probabilities.as_ptr(),
        storage.transition_probabilities.len(),
        storage.missing_probabilities.as_ptr(),
        storage.missing_probabilities.len(),
        haploid,
        storage.storage_events,
    )
}

#[no_mangle]
/// Solve using the haploid state retained by a Rust-owned graph.
///
/// # Safety
///
/// `graph` must be live and exclusively borrowed.
pub unsafe extern "C" fn shapeit_genotype_graph_solve_current_v1(
    graph: *mut GenotypeGraphV1,
) -> u32 {
    if graph.is_null() {
        return STATUS_NULL_POINTER;
    }
    let haploid = u8::from((*graph).haploid);
    shapeit_genotype_graph_solve_v1(graph, haploid)
}

#[derive(Clone, Copy)]
enum GenotypeBatchOperation {
    Build,
    Solve,
}

struct GenotypeBatchShared<'a> {
    graph_addresses: &'a [usize],
    operation: GenotypeBatchOperation,
    progress: Option<GenotypeProgressV1>,
    progress_context_address: usize,
    next_graph: AtomicUsize,
    completed: AtomicUsize,
    status: AtomicU32,
    failed_graph: AtomicUsize,
    serialized_progress: Mutex<()>,
}

fn record_batch_failure(shared: &GenotypeBatchShared<'_>, status: u32, graph: usize) {
    if shared
        .status
        .compare_exchange(STATUS_OK, status, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        shared.failed_graph.store(graph, Ordering::SeqCst);
    }
}

fn run_genotype_batch_worker(shared: &GenotypeBatchShared<'_>) {
    loop {
        if shared.status.load(Ordering::SeqCst) != STATUS_OK {
            break;
        }
        let index = shared.next_graph.fetch_add(1, Ordering::SeqCst);
        if index >= shared.graph_addresses.len() {
            break;
        }
        let graph = shared.graph_addresses[index] as *mut GenotypeGraphV1;
        let status = unsafe {
            match shared.operation {
                GenotypeBatchOperation::Build => match (*graph).build_in_place() {
                    Ok(()) => STATUS_OK,
                    Err(status) => status,
                },
                GenotypeBatchOperation::Solve => shapeit_genotype_graph_solve_current_v1(graph),
            }
        };
        if status != STATUS_OK {
            record_batch_failure(shared, status, index);
            break;
        }
        let _guard = match shared.serialized_progress.lock() {
            Ok(value) => value,
            Err(_) => {
                record_batch_failure(shared, STATUS_THREAD_FAILURE, index);
                break;
            }
        };
        let completed = shared.completed.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(progress) = shared.progress {
            unsafe {
                progress(
                    completed,
                    shared.graph_addresses.len(),
                    shared.progress_context_address as *mut c_void,
                );
            }
        }
    }
}

unsafe fn run_genotype_batch(
    worker_count: usize,
    graphs: *const *mut GenotypeGraphV1,
    graph_count: usize,
    progress: Option<GenotypeProgressV1>,
    progress_context: *mut c_void,
    result: *mut GenotypeBatchResultV1,
    operation: GenotypeBatchOperation,
) -> u32 {
    if result.is_null() {
        return STATUS_NULL_POINTER;
    }
    if worker_count == 0 || graph_count == 0 {
        return STATUS_INVALID_DIMENSIONS;
    }
    if graphs.is_null() {
        return STATUS_NULL_POINTER;
    }
    let graphs = slice::from_raw_parts(graphs, graph_count);
    if graphs.iter().any(|graph| graph.is_null()) {
        return STATUS_NULL_POINTER;
    }
    let graph_addresses: Vec<usize> = graphs.iter().map(|&graph| graph as usize).collect();
    let shared = GenotypeBatchShared {
        graph_addresses: &graph_addresses,
        operation,
        progress,
        progress_context_address: progress_context as usize,
        next_graph: AtomicUsize::new(0),
        completed: AtomicUsize::new(0),
        status: AtomicU32::new(STATUS_OK),
        failed_graph: AtomicUsize::new(usize::MAX),
        serialized_progress: Mutex::new(()),
    };
    let execution_threads = core::cmp::min(worker_count, graph_count);
    if execution_threads == 1 {
        run_genotype_batch_worker(&shared);
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(execution_threads);
            for _ in 0..execution_threads {
                let shared = &shared;
                match thread::Builder::new().spawn_scoped(scope, move || {
                    run_genotype_batch_worker(shared);
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => record_batch_failure(shared, STATUS_THREAD_FAILURE, usize::MAX),
                }
            }
            for handle in handles {
                if handle.join().is_err() {
                    record_batch_failure(&shared, STATUS_THREAD_FAILURE, usize::MAX);
                }
            }
        });
    }
    let status = shared.status.load(Ordering::SeqCst);
    let mut segments = 0usize;
    if status == STATUS_OK {
        for &address in &graph_addresses {
            segments = match segments
                .checked_add((*(address as *const GenotypeGraphV1)).segment_lengths.len())
            {
                Some(value) => value,
                None => {
                    *result = GenotypeBatchResultV1 {
                        completed: shared.completed.load(Ordering::SeqCst),
                        failed_graph: usize::MAX,
                        segments: 0,
                    };
                    return STATUS_INTEGER_OVERFLOW;
                }
            };
        }
    }
    *result = GenotypeBatchResultV1 {
        completed: shared.completed.load(Ordering::SeqCst),
        failed_graph: shared.failed_graph.load(Ordering::SeqCst),
        segments,
    };
    status
}

#[no_mangle]
/// Build every Rust-owned genotype graph on scoped Rust worker threads.
///
/// # Safety
///
/// Every graph pointer must be live, distinct, and exclusively borrowed until
/// the call returns. `result` must be writable.
pub unsafe extern "C" fn shapeit_genotype_graphs_build_v1(
    worker_count: usize,
    graphs: *const *mut GenotypeGraphV1,
    graph_count: usize,
    progress: Option<GenotypeProgressV1>,
    progress_context: *mut c_void,
    result: *mut GenotypeBatchResultV1,
) -> u32 {
    run_genotype_batch(
        worker_count,
        graphs,
        graph_count,
        progress,
        progress_context,
        result,
        GenotypeBatchOperation::Build,
    )
}

#[no_mangle]
/// Solve every Rust-owned genotype graph from its accumulated storage.
///
/// # Safety
///
/// Every graph pointer must be live, distinct, and exclusively borrowed until
/// the call returns. `result` must be writable.
pub unsafe extern "C" fn shapeit_genotype_graphs_solve_current_v1(
    worker_count: usize,
    graphs: *const *mut GenotypeGraphV1,
    graph_count: usize,
    progress: Option<GenotypeProgressV1>,
    progress_context: *mut c_void,
    result: *mut GenotypeBatchResultV1,
) -> u32 {
    run_genotype_batch(
        worker_count,
        graphs,
        graph_count,
        progress,
        progress_context,
        result,
        GenotypeBatchOperation::Solve,
    )
}

#[no_mangle]
/// Build all HMM windows for one common-phasing genotype graph.
///
/// RNG coordinates identify a fresh logical Philox stream. Recursive split
/// decisions consume it in the established depth-first order.
///
/// # Safety
///
/// Every input buffer must be readable for its stated length. `windows` must be
/// writable for `windows_capacity` records and `windows_length` must be
/// writable. Invalid layouts are rejected before output is modified.
pub unsafe extern "C" fn shapeit_genotype_windows_v1(
    variants: *const u8,
    variants_length: usize,
    variant_count: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    segment_start_centimorgans: *const f64,
    segment_start_centimorgans_length: usize,
    segment_stop_centimorgans: *const f64,
    segment_stop_centimorgans_length: usize,
    minimum_window_centimorgans: f32,
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
    windows: *mut GenotypeWindowV1,
    windows_capacity: usize,
    windows_length: *mut usize,
) -> u32 {
    if windows_length.is_null()
        || (variants_length != 0 && variants.is_null())
        || (diplotypes_length != 0 && diplotypes.is_null())
        || (segment_lengths_length != 0 && segment_lengths.is_null())
        || (segment_start_centimorgans_length != 0 && segment_start_centimorgans.is_null())
        || (segment_stop_centimorgans_length != 0 && segment_stop_centimorgans.is_null())
        || (windows_capacity != 0 && windows.is_null())
    {
        return STATUS_NULL_POINTER;
    }
    let required_variants = match required_variant_bytes(variant_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if required_variants > variants_length {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variants = if variants_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(variants, variants_length)
    };
    let diplotypes = if diplotypes_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(diplotypes, diplotypes_length)
    };
    let segment_lengths = if segment_lengths_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(segment_lengths, segment_lengths_length)
    };
    let segment_start_centimorgans = if segment_start_centimorgans_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(
            segment_start_centimorgans,
            segment_start_centimorgans_length,
        )
    };
    let segment_stop_centimorgans = if segment_stop_centimorgans_length == 0 {
        &[]
    } else {
        slice::from_raw_parts(segment_stop_centimorgans, segment_stop_centimorgans_length)
    };
    let mut rng = LogicalRng::new(seed, domain, iteration, item);
    let output = match build_windows(
        WindowInputs {
            variants,
            variant_count,
            diplotypes,
            segment_lengths,
            segment_start_centimorgans,
            segment_stop_centimorgans,
            minimum_window_centimorgans,
        },
        &mut rng,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if output.len() > windows_capacity {
        return STATUS_OUT_OF_BOUNDS;
    }
    if !output.is_empty() {
        let windows = slice::from_raw_parts_mut(windows, windows_capacity);
        windows[..output.len()].copy_from_slice(&output);
    }
    *windows_length = output.len();
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
    fn storage_preserves_mask_format_and_cpp_float_accumulation() {
        let first_transitions = [0.0f64, 1e-6, 0.25, 9e-7, 1.0];
        let first_missing = [0.125f32, 0.5];
        let mut storage = GenotypeStorageV1::new(&first_transitions, &first_missing).unwrap();
        assert_eq!(storage.transition_mask, [0b0001_0110]);
        assert_eq!(storage.transition_indexes, [1, 2, 4]);
        assert_eq!(storage.storage_events, 1);

        let next_transitions = [0.75f64, 0.333_333_333_333, 0.125, 2.0, 1e-8];
        let next_missing = [0.25f32, 0.125];
        let expected_transitions: Vec<f32> = storage
            .transition_indexes
            .iter()
            .zip(storage.transition_probabilities.iter())
            .map(|(&index, &stored)| (f64::from(stored) + next_transitions[index as usize]) as f32)
            .collect();
        storage.update(&next_transitions, &next_missing).unwrap();
        assert_eq!(storage.transition_probabilities, expected_transitions);
        assert_eq!(storage.missing_probabilities, [0.375, 0.625]);
        assert_eq!(storage.storage_events, 2);

        let view = storage.view();
        assert_eq!(view.transition_count, first_transitions.len());
        assert_eq!(view.transition_mask_length, 1);
        assert_eq!(view.transition_probabilities_length, 3);
        assert_eq!(view.missing_probabilities_length, 2);
        assert_eq!(view.storage_events, 2);
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
    fn pedigree_scaffolding_preserves_trio_and_duo_rules() {
        let father = pack(&[12, 12, 6, 0, 12]);
        let mother = pack(&[0, 6, 12, 12, 0]);
        let mut child = pack(&[6, 10, 6, 12, 1]);
        let mut counts = [0u32; 4];
        let status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                child.as_mut_ptr(),
                child.len(),
                5,
                father.as_ptr(),
                father.len(),
                mother.as_ptr(),
                mother.len(),
                0,
                counts.as_mut_ptr(),
                counts.len(),
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(child, pack(&[7, 7, 11, 12, 7]));
        assert_eq!(counts, [1, 4, 3, 0]);

        let parent = pack(&[0]);
        let mut paternal = pack(&[6]);
        let mut maternal = pack(&[6]);
        let mut paternal_counts = [0u32; 4];
        let mut maternal_counts = [0u32; 4];
        let paternal_status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                paternal.as_mut_ptr(),
                paternal.len(),
                1,
                parent.as_ptr(),
                parent.len(),
                core::ptr::null(),
                0,
                1,
                paternal_counts.as_mut_ptr(),
                paternal_counts.len(),
            )
        };
        let maternal_status = unsafe {
            shapeit_genotype_pedigree_scaffold_v1(
                maternal.as_mut_ptr(),
                maternal.len(),
                1,
                core::ptr::null(),
                0,
                parent.as_ptr(),
                parent.len(),
                2,
                maternal_counts.as_mut_ptr(),
                maternal_counts.len(),
            )
        };
        assert_eq!(paternal_status, STATUS_OK);
        assert_eq!(maternal_status, STATUS_OK);
        assert_eq!(paternal, pack(&[11]));
        assert_eq!(maternal, pack(&[7]));
        assert_eq!(paternal_counts, [0, 1, 1, 0]);
        assert_eq!(maternal_counts, [0, 1, 1, 0]);
    }

    #[test]
    fn haploid_reset_preserves_existing_bitwise_missing_semantics() {
        let mut variants = pack(&[2, 3, 0, 1]);
        let mut reset_count = 0u32;
        let status = unsafe {
            shapeit_genotype_reset_haploid_hets_v1(
                variants.as_mut_ptr(),
                variants.len(),
                4,
                &mut reset_count,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, pack(&[3, 3, 0, 1]));
        assert_eq!(reset_count, 2);
    }

    #[test]
    fn pruning_merges_eight_ranked_haplotypes_and_preserves_tie_order() {
        let variants = pack(&[2, 2]);
        let ambiguous = [0xaau8, 0xcc];
        let diagonal = (0..8).fold(0u64, |mask, haplotype| {
            mask | (1u64 << (haplotype * 8 + haplotype))
        });
        let diplotypes = [diagonal, diagonal];
        let lengths = [1u16, 1u16];
        let mut probabilities = vec![0.0f64; 72];
        for haplotype in 0..8 {
            probabilities[8 + haplotype * 8 + haplotype] = 0.125;
        }
        let mut output_ambiguous = [0u8; 2];
        let mut output_diplotypes = [0u64; 2];
        let mut output_lengths = [0u16; 2];
        let mut output_segments = 0usize;
        let mut output_transitions = 0u32;
        let status = unsafe {
            shapeit_genotype_prune_v1(
                variants.as_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                probabilities.as_ptr(),
                probabilities.len(),
                0.999,
                output_ambiguous.as_mut_ptr(),
                output_ambiguous.len(),
                output_diplotypes.as_mut_ptr(),
                output_diplotypes.len(),
                output_lengths.as_mut_ptr(),
                output_lengths.len(),
                &mut output_segments,
                &mut output_transitions,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(output_ambiguous, ambiguous);
        assert_eq!(output_diplotypes[0], diagonal);
        assert_eq!(output_lengths[0], 2);
        assert_eq!(output_segments, 1);
        assert_eq!(output_transitions, 8);

        let status = unsafe {
            shapeit_genotype_prune_v1(
                variants.as_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                probabilities.as_ptr(),
                probabilities.len(),
                1.0,
                output_ambiguous.as_mut_ptr(),
                output_ambiguous.len(),
                output_diplotypes.as_mut_ptr(),
                output_diplotypes.len(),
                output_lengths.as_mut_ptr(),
                output_lengths.len(),
                &mut output_segments,
                &mut output_transitions,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(output_ambiguous, ambiguous);
        assert_eq!(output_diplotypes, diplotypes);
        assert_eq!(output_lengths, lengths);
        assert_eq!(output_segments, 2);
        assert_eq!(output_transitions, 72);
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

    #[test]
    fn owned_graph_lifecycle_exposes_the_complete_built_view() {
        let variants = pack(&[2, 2, 2, 2, 1, 0]);
        let expected = build(&[2, 2, 2, 2, 1, 0]);
        let mut graph = core::ptr::null_mut();
        let create_status = unsafe {
            shapeit_genotype_graph_create_v1(variants.as_ptr(), variants.len(), 6, &mut graph)
        };
        assert_eq!(create_status, STATUS_OK);
        assert!(!graph.is_null());

        let mut view = GenotypeGraphViewV1 {
            variant_count: 0,
            variants: core::ptr::null(),
            variants_length: 0,
            ambiguous: core::ptr::null(),
            ambiguous_length: 0,
            diplotypes: core::ptr::null(),
            diplotypes_length: 0,
            segment_lengths: core::ptr::null(),
            segment_lengths_length: 0,
            missing_count: 0,
            transition_count: 0,
        };
        let borrow_status = unsafe { shapeit_genotype_graph_borrow_v1(graph, &mut view) };
        assert_eq!(borrow_status, STATUS_OK);
        assert_eq!(view.variant_count, 6);
        assert_eq!(view.variants_length, variants.len());
        assert_eq!(view.ambiguous_length, expected.0.ambiguous);
        assert_eq!(view.diplotypes_length, expected.0.segments);
        assert_eq!(view.segment_lengths_length, expected.0.segments);
        assert_eq!(view.missing_count, expected.0.missing);
        assert_eq!(view.transition_count, expected.4);
        unsafe {
            assert_eq!(
                slice::from_raw_parts(view.variants, view.variants_length),
                variants
            );
            assert_eq!(
                slice::from_raw_parts(view.ambiguous, view.ambiguous_length),
                expected.2
            );
            assert_eq!(
                slice::from_raw_parts(view.diplotypes, view.diplotypes_length),
                expected.3
            );
            assert_eq!(
                slice::from_raw_parts(view.segment_lengths, view.segment_lengths_length),
                expected.1
            );
            shapeit_genotype_graph_free_v1(graph);
        }
    }

    #[test]
    fn owned_graph_allocates_variants_before_in_place_build() {
        let expected_variants = pack(&[2, 2, 2, 2, 1, 0]);
        let mut graph = core::ptr::null_mut();
        let allocate_status = unsafe { shapeit_genotype_graph_allocate_v1(6, &mut graph) };
        assert_eq!(allocate_status, STATUS_OK);

        let mut haploid = 0xa5;
        let mut double_precision = 0xa5;
        assert_eq!(
            unsafe { shapeit_genotype_graph_flags_v1(graph, &mut haploid, &mut double_precision,) },
            STATUS_OK
        );
        assert_eq!((haploid, double_precision), (0, 0));
        assert_eq!(
            unsafe { shapeit_genotype_graph_set_haploid_v1(graph, 2) },
            STATUS_INVALID_DIMENSIONS
        );
        assert_eq!(
            unsafe { shapeit_genotype_graph_set_haploid_v1(graph, 1) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { shapeit_genotype_graph_require_double_v1(graph) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { shapeit_genotype_graph_flags_v1(graph, &mut haploid, &mut double_precision,) },
            STATUS_OK
        );
        assert_eq!((haploid, double_precision), (1, 1));

        let mut variants = core::ptr::null_mut();
        let mut variants_length = 0usize;
        let variants_status = unsafe {
            shapeit_genotype_graph_variants_mut_v1(graph, &mut variants, &mut variants_length)
        };
        assert_eq!(variants_status, STATUS_OK);
        assert_eq!(variants_length, expected_variants.len());
        unsafe {
            slice::from_raw_parts_mut(variants, variants_length)
                .copy_from_slice(&expected_variants);
        }
        let build_status = unsafe { shapeit_genotype_graph_build_in_place_v1(graph) };
        assert_eq!(build_status, STATUS_OK);
        assert_eq!(
            unsafe { shapeit_genotype_graph_build_in_place_v1(graph) },
            STATUS_INVALID_DIMENSIONS
        );
        let mut view = unsafe { (*graph).view() };
        let borrow_status = unsafe { shapeit_genotype_graph_borrow_v1(graph, &mut view) };
        assert_eq!(borrow_status, STATUS_OK);
        assert_eq!(view.variant_count, 6);
        assert_eq!(view.variants_length, expected_variants.len());
        unsafe { shapeit_genotype_graph_free_v1(graph) };
    }

    #[test]
    fn owned_graph_retains_probability_storage_through_solving() {
        let variants = pack(&[3 | 4]);
        let mut graph = Box::new(GenotypeGraphV1::new(&variants, 1));
        let probabilities = vec![1.0f64; graph.transition_count as usize];
        let first_status = unsafe {
            shapeit_genotype_graph_store_v1(
                graph.as_mut(),
                probabilities.as_ptr(),
                probabilities.len(),
                core::ptr::null(),
                0,
            )
        };
        let second_status = unsafe {
            shapeit_genotype_graph_store_v1(
                graph.as_mut(),
                probabilities.as_ptr(),
                probabilities.len(),
                core::ptr::null(),
                0,
            )
        };
        assert_eq!(first_status, STATUS_OK);
        assert_eq!(second_status, STATUS_OK);

        let mut storage_view = GenotypeStorageViewV1 {
            transition_count: 0,
            transition_mask: core::ptr::null(),
            transition_mask_length: 0,
            transition_probabilities: core::ptr::null(),
            transition_probabilities_length: 0,
            missing_probabilities: core::ptr::null(),
            missing_probabilities_length: 0,
            storage_events: 0,
        };
        let borrow_status =
            unsafe { shapeit_genotype_graph_storage_borrow_v1(graph.as_ref(), &mut storage_view) };
        assert_eq!(borrow_status, STATUS_OK);
        assert_eq!(storage_view.transition_count, probabilities.len());
        assert_eq!(storage_view.storage_events, 2);
        let solve_status = unsafe { shapeit_genotype_graph_solve_v1(graph.as_mut(), 0) };
        assert_eq!(solve_status, STATUS_OK);
    }

    #[test]
    fn graph_sampler_applies_sampled_diplotypes_without_changing_graph_codes() {
        let mut variants = [0x22u8];
        let ambiguous = [0xaa, 0xcc];
        let diplotypes = [1u64 << 9, 1u64 << 18];
        let lengths = [1u16, 1u16];
        let transitions = [1.0f64, 1.0f64];
        let status = unsafe {
            shapeit_genotype_sample_v1(
                variants.as_mut_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                transitions.as_ptr(),
                transitions.len(),
                core::ptr::null(),
                0,
                0,
                15_052_011,
                3,
                7,
                11,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0xee]);
    }

    #[test]
    fn forward_sampler_rejects_zero_mass_row_before_indexing_the_next_block() {
        let diplotypes = [0b11u64; 3];
        let transitions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25];
        let mut sampled = [0xa5u8; 3];
        let mut rng = LogicalRng::new(15_052_011, 3, 7, 11);

        assert_eq!(
            sample_forward(&diplotypes, &transitions, &mut sampled, &mut rng),
            Err(SampleError::DegenerateDistribution)
        );
        assert_eq!(sampled[2], 0xa5);
    }

    #[test]
    fn backward_sampler_rejects_zero_mass_conditional_row() {
        let diplotypes = [0b11u64; 3];
        let transitions = [0.5, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0];
        let mut sampled = [0xa5u8; 3];
        let mut rng = LogicalRng::new(15_052_011, 3, 7, 11);

        assert_eq!(
            sample_backward(
                &diplotypes,
                &transitions,
                transitions.len(),
                &mut sampled,
                &mut rng,
            ),
            Err(SampleError::DegenerateDistribution)
        );
        assert_eq!(sampled[0], 0xa5);
    }

    #[test]
    fn graph_solver_applies_stored_path_and_missing_consensus() {
        let mut variants = [0x21u8];
        let ambiguous = [0xaau8];
        let diplotypes = [1u64 << 9];
        let lengths = [2u16];
        let stored_indexes = [0u32];
        let stored_probabilities = [1.0f32];
        let mut missing_probabilities = [0.0f32; 8];
        missing_probabilities[1] = 1.0;
        let status = unsafe {
            shapeit_genotype_solve_v1(
                variants.as_mut_ptr(),
                variants.len(),
                2,
                ambiguous.as_ptr(),
                ambiguous.len(),
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                stored_indexes.as_ptr(),
                stored_indexes.len(),
                stored_probabilities.as_ptr(),
                stored_probabilities.len(),
                missing_probabilities.as_ptr(),
                missing_probabilities.len(),
                0,
                2,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0xed]);
    }

    #[test]
    fn graph_solver_preserves_haploid_missing_consensus_rule() {
        let mut variants = [0x01u8];
        let diplotypes = [1u64 << 9];
        let lengths = [1u16];
        let stored_indexes = [0u32];
        let stored_probabilities = [1.0f32];
        let mut missing_probabilities = [0.0f32; 8];
        missing_probabilities[1] = 1.5;
        let status = unsafe {
            shapeit_genotype_solve_v1(
                variants.as_mut_ptr(),
                variants.len(),
                1,
                core::ptr::null(),
                0,
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                stored_indexes.as_ptr(),
                stored_indexes.len(),
                stored_probabilities.as_ptr(),
                stored_probabilities.len(),
                missing_probabilities.as_ptr(),
                missing_probabilities.len(),
                1,
                2,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(variants, [0x0d]);
    }

    #[test]
    fn window_builder_preserves_graph_coordinate_conventions() {
        let variants = [0u8; 50];
        let diplotypes = [1u64; 4];
        let lengths = [25u16; 4];
        let start_centimorgans = [0.0f64, 1.0, 2.0, 3.0];
        let stop_centimorgans = [0.9f64, 1.9, 2.9, 4.0];
        let mut windows = [GenotypeWindowV1::default(); 4];
        let mut window_count = 0usize;
        let status = unsafe {
            shapeit_genotype_windows_v1(
                variants.as_ptr(),
                variants.len(),
                100,
                diplotypes.as_ptr(),
                diplotypes.len(),
                lengths.as_ptr(),
                lengths.len(),
                start_centimorgans.as_ptr(),
                start_centimorgans.len(),
                stop_centimorgans.as_ptr(),
                stop_centimorgans.len(),
                1.0,
                15_052_011,
                2,
                3,
                7,
                windows.as_mut_ptr(),
                windows.len(),
                &mut window_count,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(window_count, 1);
        assert_eq!(
            windows[0],
            GenotypeWindowV1 {
                start_locus: 0,
                start_segment: 0,
                start_ambiguous: 0,
                start_missing: 0,
                start_transition: 1,
                stop_locus: 99,
                stop_segment: 3,
                stop_ambiguous: -1,
                stop_missing: -1,
                stop_transition: 3,
            }
        );
    }
}
