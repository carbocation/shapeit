#![allow(clippy::needless_range_loop)]

use core::{mem, slice};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use self::single::{shapeit_hmm_run_segment_single_prevalidated_v1, HmmSegmentSingleV1};
use crate::bitmatrix::{
    shapeit_bitmatrix_refresh_haplotypes_v1, shapeit_bitmatrix_subset_transpose_v1,
    shapeit_bitmatrix_transpose_v1,
};
use crate::conditioning::{
    conditioning_graph_job_build_prevalidated_v1, shapeit_conditioning_graph_job_build_v1,
    validate_conditioning_graph_job_shared_v1, ConditioningGraphBuildV1, ConditioningJobV1,
    ConditioningSharedLayout,
};
use crate::genotype::{
    sample_graph_current, shapeit_genotype_graph_prune_v1, shapeit_genotype_graph_store_v1,
    GenotypeGraphV1, GenotypeWindowV1, SampleError,
};
use crate::ibd2::{Ibd2StatsV1, Ibd2TracksV1};
use crate::pbwt::{
    pbwt_select_chunk_prevalidated_v1, shapeit_pbwt_select_sites_v1,
    shapeit_pbwt_transpose_neighbors_v1, validate_pbwt_select_job_v1, PbwtSelectJobLayout,
    PbwtSelectJobV1,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
    _mm256_setzero_pd, _mm256_storeu_pd,
};

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;
const STATUS_THREAD_FAILURE: u32 = 6;
const HAPLOTYPES: usize = 8;
const STAGE_BURN: u32 = 0;
const STAGE_PRUNE: u32 = 1;
const STAGE_MAIN: u32 = 2;

#[repr(C)]
pub struct HmmSegmentDoubleV1 {
    abi_version: u32,
    struct_size: u32,

    variants: *const u8,
    variants_length: usize,
    ambiguous: *const u8,
    ambiguous_length: usize,
    segment_lengths: *const u16,
    segment_lengths_length: usize,
    diplotypes: *const u64,
    diplotypes_length: usize,

    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    conditioning_haplotypes: usize,
    locus_offset: u32,

    centimorgans: *const f32,
    centimorgans_length: usize,
    recombination: *const f32,
    recombination_length: usize,
    rare_alleles: *const i8,
    rare_alleles_length: usize,
    effective_population_size: i32,
    total_haplotypes: i32,
    emission_match: f64,
    emission_mismatch: f64,

    segment_first: i32,
    segment_last: i32,
    locus_first: i32,
    locus_last: i32,
    ambiguous_first: i32,
    ambiguous_last: i32,
    missing_first: i32,
    missing_last: i32,
    transition_first: i32,
    transition_last: i32,

    transition_probabilities: *mut f64,
    transition_probabilities_length: usize,
    missing_probabilities: *mut f32,
    missing_probabilities_length: usize,

    scratch: *mut f64,
    scratch_length: usize,
    alpha_locus_scratch: *mut i32,
    alpha_locus_scratch_length: usize,
}

#[repr(C)]
pub struct HmmJobV1 {
    abi_version: u32,
    struct_size: u32,
    graph: *mut GenotypeGraphV1,
    conditioning_job: *mut ConditioningJobV1,
    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    centimorgans: *const f32,
    centimorgans_length: usize,
    recombination: *const f32,
    recombination_length: usize,
    rare_alleles: *const i8,
    rare_alleles_length: usize,
    effective_population_size: i32,
    total_haplotypes: i32,
    emission_match: f64,
    emission_mismatch: f64,
    transition_probabilities: *mut f64,
    transition_probabilities_length: usize,
    missing_probabilities: *mut f32,
    missing_probabilities_length: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HmmJobResultV1 {
    underflow_recovered_summing: i32,
    underflow_recovered_precision: u32,
    fatal_outcome: i32,
    windows_completed: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HmmPhaseJobV1 {
    abi_version: u32,
    struct_size: u32,
    graph: *mut GenotypeGraphV1,
    conditioning_job: *mut ConditioningJobV1,
    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    centimorgans: *const f32,
    centimorgans_length: usize,
    recombination: *const f32,
    recombination_length: usize,
    rare_alleles: *const i8,
    rare_alleles_length: usize,
    effective_population_size: i32,
    total_haplotypes: i32,
    emission_match: f64,
    emission_mismatch: f64,
    stage: u32,
    prune_threshold: f64,
    sample_seed: u64,
    sample_domain: u32,
    sample_iteration: u32,
    sample_item: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CommonPhaseJobV1 {
    abi_version: u32,
    struct_size: usize,
    conditioning: ConditioningGraphBuildV1,
    phase: HmmPhaseJobV1,
}

pub type CommonProgressV1 = unsafe extern "C" fn(usize, usize, *mut c_void);

#[repr(C)]
pub struct CommonIterationV1 {
    abi_version: u32,
    struct_size: usize,
    sample_template: CommonPhaseJobV1,
    base_pair_positions: *const i32,
    base_pair_positions_length: usize,
    ibd2_registry: *mut Ibd2TracksV1,
    progress: Option<CommonProgressV1>,
    progress_context: *mut c_void,
}

#[repr(C)]
pub struct CommonPbwtSelectionV1 {
    abi_version: u32,
    struct_size: usize,
    haplotypes: *const u8,
    haplotypes_length: usize,
    haplotype_stride: usize,
    site_count: usize,
    haplotype_count: usize,
    target_individual_count: usize,
    evaluated_sites: *const u8,
    evaluated_sites_length: usize,
    selected_sites: *mut u8,
    selected_sites_length: usize,
    site_groups: *const i32,
    site_groups_length: usize,
    group_count: usize,
    site_chunks: *const i32,
    site_chunks_length: usize,
    chunk_starts: *const i32,
    chunk_count: usize,
    depth: usize,
    ibd2_registry: *const Ibd2TracksV1,
    neighbors: *mut i32,
    neighbors_length: usize,
    seed: u64,
    domain: u32,
    iteration: u32,
    progress: Option<CommonProgressV1>,
    progress_context: *mut c_void,
}

#[repr(C)]
pub struct CommonFullIterationV1 {
    abi_version: u32,
    struct_size: usize,
    pbwt: CommonPbwtSelectionV1,
    phase: CommonIterationV1,
    haplotype_major: *mut u8,
    haplotype_major_length: usize,
    haplotype_major_rows: usize,
    haplotype_major_stride: usize,
    variant_major: *mut u8,
    variant_major_length: usize,
    variant_major_rows: usize,
    variant_major_stride: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CommonIterationResultV1 {
    underflow_recovered_summing: u64,
    underflow_recovered_precision: u64,
    fatal_outcome: i32,
    failed_sample: usize,
    windows: usize,
    conditioning_states_mean: f64,
    conditioning_states_sd: f64,
    window_megabases_mean: f64,
    window_megabases_sd: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CommonFullIterationResultV1 {
    phase: CommonIterationResultV1,
    ibd2: Ibd2StatsV1,
    failed_pbwt_chunk: usize,
    pbwt_seconds: f64,
    hmm_seconds: f64,
    ibd2_seconds: f64,
    haplotype_refresh_seconds: f64,
    transpose_seconds: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CommonFallbackV1 {
    sample: usize,
    window: usize,
    states: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct CommonStats {
    count: usize,
    sum: f64,
    sum_squares: f64,
}

impl CommonStats {
    #[inline]
    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.sum_squares += other.sum_squares;
    }

    fn mean(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn standard_deviation(self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let count = self.count as f64;
        let variance = (self.sum_squares - self.sum * self.sum / count) / (count - 1.0);
        variance.max(0.0).sqrt()
    }
}

#[derive(Default)]
struct CommonWorkerIteration {
    underflow_recovered_summing: u64,
    underflow_recovered_precision: u64,
    conditioning_states: CommonStats,
    window_megabases: CommonStats,
    fallbacks: Vec<CommonFallbackV1>,
}

struct CommonWorkerV1 {
    conditioning_job: *mut ConditioningJobV1,
    iteration: CommonWorkerIteration,
}

// The worker exclusively owns this allocation and is borrowed by at most one
// scoped execution thread at a time.
unsafe impl Send for CommonWorkerV1 {}

impl CommonWorkerV1 {
    fn new() -> Self {
        Self {
            conditioning_job: Box::into_raw(Box::new(ConditioningJobV1::default())),
            iteration: CommonWorkerIteration::default(),
        }
    }

    fn reset_iteration(&mut self) {
        self.iteration.underflow_recovered_summing = 0;
        self.iteration.underflow_recovered_precision = 0;
        self.iteration.conditioning_states = CommonStats::default();
        self.iteration.window_megabases = CommonStats::default();
        self.iteration.fallbacks.clear();
    }
}

impl Drop for CommonWorkerV1 {
    fn drop(&mut self) {
        if !self.conditioning_job.is_null() {
            unsafe {
                drop(Box::from_raw(self.conditioning_job));
            }
            self.conditioning_job = core::ptr::null_mut();
        }
    }
}

pub struct CommonWorkersV1 {
    workers: Vec<CommonWorkerV1>,
    graph_addresses: Vec<usize>,
    haploid_individuals: Vec<u8>,
    variant_count: usize,
    fallbacks: Vec<CommonFallbackV1>,
}

#[derive(Clone, Copy)]
struct SharedCommonTemplate(CommonPhaseJobV1);

// Every pointer in the template addresses immutable iteration input. Mutable
// graph and worker-job pointers are installed into a private copy per sample.
unsafe impl Sync for SharedCommonTemplate {}

struct CommonIterationShared<'a> {
    template: SharedCommonTemplate,
    conditioning_validation: ConditioningSharedLayout,
    graph_addresses: &'a [usize],
    haploid_individuals: &'a [u8],
    base_pair_positions: &'a [i32],
    ibd2_registry_address: usize,
    progress: Option<CommonProgressV1>,
    progress_context_address: usize,
    next_sample: AtomicUsize,
    completed_samples: AtomicUsize,
    status: AtomicU32,
    failed_sample: AtomicUsize,
    fatal_outcome: AtomicI32,
    serialized_output: Mutex<()>,
}

struct CommonPbwtShared<'a> {
    parameters: &'a CommonPbwtSelectionV1,
    job: PbwtSelectJobV1,
    validation: PbwtSelectJobLayout,
    next_chunk: AtomicUsize,
    completed_chunks: AtomicUsize,
    status: AtomicU32,
    failed_chunk: AtomicUsize,
    serialized_progress: Mutex<()>,
}

// All input pointers are immutable for the selection, while output neighbour
// slabs are disjoint by PBWT chunk. Progress callbacks are serialized.
unsafe impl Sync for CommonPbwtSelectionV1 {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScratchLayout {
    states: usize,
    segment_count: usize,
    missing_count: usize,
    total: usize,
}

fn scratch_layout(
    conditioning_haplotypes: usize,
    segment_count: usize,
    missing_count: usize,
) -> Result<ScratchLayout, u32> {
    if conditioning_haplotypes == 0 || segment_count == 0 {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let states = conditioning_haplotypes
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha = segment_count
        .checked_mul(states)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha_sum = segment_count
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha_missing = missing_count
        .checked_mul(states)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha_sum_missing = missing_count
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let total = states
        .checked_add(conditioning_haplotypes)
        .and_then(|value| value.checked_add(alpha))
        .and_then(|value| value.checked_add(alpha_sum))
        .and_then(|value| value.checked_add(segment_count))
        .and_then(|value| value.checked_add(alpha_missing))
        .and_then(|value| value.checked_add(alpha_sum_missing))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    Ok(ScratchLayout {
        states,
        segment_count,
        missing_count,
        total,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ValidatedLayout {
    scratch: ScratchLayout,
    segment_first: usize,
    segment_last: usize,
    locus_first: usize,
    locus_last: usize,
    ambiguous_first: usize,
    missing_first: usize,
    transition_last: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HmmGraphPrefix {
    loci: usize,
    ambiguous: usize,
    missing: usize,
    transitions: usize,
}

#[derive(Debug)]
struct ValidatedHmmGraph {
    // Entry i contains the cumulative graph dimensions before segment i.
    prefixes: Vec<HmmGraphPrefix>,
}

#[derive(Debug)]
struct DoubleJobValidation {
    // This cache is scoped to one job invocation (and its precision retries),
    // so graph pruning or conditioning-job reuse cannot make it stale.
    graph: ValidatedHmmGraph,
}

#[cfg(test)]
std::thread_local! {
    static DOUBLE_GRAPH_VALIDATION_SCANS: core::cell::Cell<usize> = const {
        core::cell::Cell::new(0)
    };
}

#[cfg(test)]
fn record_double_graph_validation_scan() {
    DOUBLE_GRAPH_VALIDATION_SCANS.with(|scans| scans.set(scans.get() + 1));
}

#[cfg(not(test))]
#[inline]
fn record_double_graph_validation_scan() {}

#[inline]
fn variant_code(variants: &[u8], locus: usize) -> u8 {
    (variants[locus >> 1] >> ((locus & 1) << 2)) & 3
}

#[inline]
fn diplotype_count(mask: u64) -> usize {
    mask.count_ones() as usize
}

fn require_const_pointer<T>(pointer: *const T, length: usize) -> Result<(), u32> {
    if length != 0 && pointer.is_null() {
        Err(STATUS_NULL_POINTER)
    } else {
        Ok(())
    }
}

fn require_mut_pointer<T>(pointer: *mut T, length: usize) -> Result<(), u32> {
    if length != 0 && pointer.is_null() {
        Err(STATUS_NULL_POINTER)
    } else {
        Ok(())
    }
}

unsafe fn const_slice<'a, T>(pointer: *const T, length: usize) -> &'a [T] {
    if length == 0 {
        &[]
    } else {
        slice::from_raw_parts(pointer, length)
    }
}

unsafe fn mut_slice<'a, T>(pointer: *mut T, length: usize) -> &'a mut [T] {
    if length == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(pointer, length)
    }
}

fn usize_coordinate(value: i32) -> Result<usize, u32> {
    usize::try_from(value).map_err(|_| STATUS_INVALID_DIMENSIONS)
}

fn signed_stop(count: usize) -> Result<i32, u32> {
    if count == 0 {
        Ok(-1)
    } else {
        i32::try_from(count - 1).map_err(|_| STATUS_INTEGER_OVERFLOW)
    }
}

fn validate_double_scalars(parameters: &HmmSegmentDoubleV1) -> Result<(), u32> {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size as usize != mem::size_of::<HmmSegmentDoubleV1>()
        || parameters.effective_population_size <= 0
        || parameters.total_haplotypes <= 0
        || !parameters.emission_match.is_finite()
        || !parameters.emission_mismatch.is_finite()
        || parameters.emission_match == 0.0
    {
        Err(STATUS_INVALID_DIMENSIONS)
    } else {
        Ok(())
    }
}

fn validate_hmm_graph(
    variants: &[u8],
    segment_lengths: &[u16],
    diplotypes: &[u64],
) -> Result<ValidatedHmmGraph, u32> {
    record_double_graph_validation_scan();
    if segment_lengths.len() > diplotypes.len() {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    if segment_lengths.contains(&0) || diplotypes[..segment_lengths.len()].contains(&0) {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let total_loci = segment_lengths
        .iter()
        .try_fold(0usize, |sum, &length| sum.checked_add(length as usize))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let required_variant_bytes = total_loci.checked_add(1).ok_or(STATUS_INTEGER_OVERFLOW)? >> 1;
    if variants.len() < required_variant_bytes {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    let prefix_capacity = segment_lengths
        .len()
        .checked_add(1)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let mut prefixes = Vec::with_capacity(prefix_capacity);
    prefixes.push(HmmGraphPrefix::default());
    let mut prefix = HmmGraphPrefix::default();
    let mut previous_diplotypes = 1usize;
    for (&length, &diplotype_mask) in segment_lengths.iter().zip(diplotypes.iter()) {
        let segment_stop = prefix
            .loci
            .checked_add(length as usize)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        for locus in prefix.loci..segment_stop {
            match variant_code(variants, locus) {
                1 => {
                    prefix.missing = prefix
                        .missing
                        .checked_add(1)
                        .ok_or(STATUS_INTEGER_OVERFLOW)?
                }
                2 | 3 => {
                    prefix.ambiguous = prefix
                        .ambiguous
                        .checked_add(1)
                        .ok_or(STATUS_INTEGER_OVERFLOW)?
                }
                _ => {}
            }
        }
        prefix.loci = segment_stop;

        let current_diplotypes = diplotype_count(diplotype_mask);
        let transition_count = previous_diplotypes
            .checked_mul(current_diplotypes)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        prefix.transitions = prefix
            .transitions
            .checked_add(transition_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        prefixes.push(prefix);
        previous_diplotypes = current_diplotypes;
    }

    debug_assert_eq!(prefix.loci, total_loci);
    Ok(ValidatedHmmGraph { prefixes })
}

fn validate_hmm_window(
    parameters: &HmmSegmentDoubleV1,
    graph: &ValidatedHmmGraph,
) -> Result<ValidatedLayout, u32> {
    validate_double_scalars(parameters)?;

    let segment_first = usize_coordinate(parameters.segment_first)?;
    let segment_last = usize_coordinate(parameters.segment_last)?;
    let locus_first = usize_coordinate(parameters.locus_first)?;
    let locus_last = usize_coordinate(parameters.locus_last)?;
    let segment_count = graph.prefixes.len().saturating_sub(1);
    if segment_first > segment_last || segment_last >= segment_count || locus_first > locus_last {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    let before = graph.prefixes[segment_first];
    let after_first = graph.prefixes[segment_first + 1];
    let after = graph.prefixes[segment_last + 1];
    let expected_locus_first = before.loci;
    let expected_locus_last = after.loci - 1;
    let expected_ambiguous_stop = signed_stop(after.ambiguous)?;
    let expected_missing_stop = signed_stop(after.missing)?;
    let expected_transition_first = after_first.transitions;
    let expected_transition_last = after.transitions - 1;

    if locus_first != expected_locus_first
        || locus_last != expected_locus_last
        || parameters.ambiguous_first
            != i32::try_from(before.ambiguous).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.ambiguous_last != expected_ambiguous_stop
        || parameters.missing_first
            != i32::try_from(before.missing).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.missing_last != expected_missing_stop
        || parameters.transition_first
            != i32::try_from(expected_transition_first).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.transition_last
            != i32::try_from(expected_transition_last).map_err(|_| STATUS_INTEGER_OVERFLOW)?
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let window_missing_count = after.missing - before.missing;
    let scratch = scratch_layout(
        parameters.conditioning_haplotypes,
        segment_last - segment_first + 1,
        window_missing_count,
    )?;
    if parameters.scratch_length < scratch.total
        || parameters.alpha_locus_scratch_length < scratch.segment_count
    {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    let expected_stride = parameters
        .conditioning_haplotypes
        .checked_add(7)
        .ok_or(STATUS_INTEGER_OVERFLOW)?
        >> 3;
    if parameters.haplotype_stride != expected_stride || parameters.locus_offset >= 8 {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let haplotype_rows = (locus_last - locus_first + 1)
        .checked_add(parameters.locus_offset as usize)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let required_haplotype_length = haplotype_rows
        .checked_mul(parameters.haplotype_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let totals = graph.prefixes[segment_count];
    if required_haplotype_length > parameters.haplotypes_length
        || totals.ambiguous > parameters.ambiguous_length
        || totals.loci > parameters.centimorgans_length
        || totals.loci > parameters.rare_alleles_length
        || totals.loci.saturating_sub(1) > parameters.recombination_length
        || totals.transitions > parameters.transition_probabilities_length
        || totals
            .missing
            .checked_mul(HAPLOTYPES)
            .ok_or(STATUS_INTEGER_OVERFLOW)?
            > parameters.missing_probabilities_length
    {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    Ok(ValidatedLayout {
        scratch,
        segment_first,
        segment_last,
        locus_first,
        locus_last,
        ambiguous_first: before.ambiguous,
        missing_first: before.missing,
        transition_last: expected_transition_last,
    })
}

fn validate(
    parameters: &HmmSegmentDoubleV1,
    variants: &[u8],
    segment_lengths: &[u16],
    diplotypes: &[u64],
) -> Result<ValidatedLayout, u32> {
    validate_double_scalars(parameters)?;
    let graph = validate_hmm_graph(variants, segment_lengths, diplotypes)?;
    validate_hmm_window(parameters, &graph)
}

extern "C" {
    fn expm1f(value: f32) -> f32;
}

#[inline]
fn sum8(values: &[f64]) -> f64 {
    values[0] + values[1] + values[2] + values[3] + values[4] + values[5] + values[6] + values[7]
}

struct DoubleEngine<'a> {
    variants: &'a [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    diplotypes: &'a [u64],
    haplotypes: &'a [u8],
    haplotype_stride: usize,
    conditioning_haplotypes: usize,
    locus_offset: usize,
    centimorgans: &'a [f32],
    recombination: &'a [f32],
    rare_alleles: &'a [i8],
    effective_population_size: i32,
    total_haplotypes: i32,
    mismatch: f64,

    segment_first: usize,
    segment_last: usize,
    locus_first: usize,
    locus_last: usize,
    ambiguous_first: usize,
    missing_first: usize,
    transition_last: usize,

    prob: &'a mut [f64],
    prob_sum_k: &'a mut [f64],
    alpha: &'a mut [f64],
    alpha_sum: &'a mut [f64],
    alpha_sum_sum: &'a mut [f64],
    alpha_missing: &'a mut [f64],
    alpha_sum_missing: &'a mut [f64],
    alpha_locus: &'a mut [i32],
    transition_probabilities: &'a mut [f64],
    missing_probabilities: &'a mut [f32],

    prob_sum_t: f64,
    prob_sum_h: [f64; HAPLOTYPES],
    sum_h_probs: f64,
    sum_d_probs: f64,
    h_probs: [f64; HAPLOTYPES * HAPLOTYPES],
    d_probs: [f64; HAPLOTYPES * HAPLOTYPES * HAPLOTYPES * HAPLOTYPES],
}

impl DoubleEngine<'_> {
    #[inline]
    fn allele(&self, relative_locus: usize, conditioning_haplotype: usize) -> bool {
        let row = relative_locus + self.locus_offset;
        let value = self.haplotypes[row * self.haplotype_stride + (conditioning_haplotype >> 3)];
        ((value >> (7 - (conditioning_haplotype & 7))) & 1) != 0
    }

    #[inline]
    fn hap0(&self, locus: usize) -> bool {
        let value = self.variants[locus >> 1];
        (value & (4 << ((locus & 1) << 2))) != 0
    }

    #[inline]
    fn is_ambiguous(&self, locus: usize) -> bool {
        variant_code(self.variants, locus) > 1
    }

    #[inline]
    fn is_missing(&self, locus: usize) -> bool {
        variant_code(self.variants, locus) == 1
    }

    fn transition_probability(&self, previous: usize, current: usize) -> f64 {
        debug_assert_ne!(previous, current);
        if previous.abs_diff(current) == 1 {
            return self.recombination[previous.min(current)] as f64;
        }
        let mut distance = if previous < current {
            self.centimorgans[current] - self.centimorgans[previous]
        } else {
            self.centimorgans[previous] - self.centimorgans[current]
        };
        if f64::from(distance) <= 1.0e-7 {
            distance = 1.0e-7_f64 as f32;
        }
        let argument = (-0.04_f64 * f64::from(self.effective_population_size) * f64::from(distance)
            / f64::from(self.total_haplotypes)) as f32;
        // SAFETY: expm1f is a pure C math-library function for every f32 input.
        -(unsafe { expm1f(argument) } as f64)
    }

    #[inline]
    fn update_total(&mut self, sums: [f64; HAPLOTYPES]) {
        self.prob_sum_h = sums;
        self.prob_sum_t = sum8(&self.prob_sum_h);
    }

    fn init_hom(&mut self, locus: usize, relative_locus: usize) {
        let genotype_allele = self.hap0(locus);
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let emission = if genotype_allele != self.allele(relative_locus, k) {
                self.mismatch
            } else {
                1.0
            };
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                self.prob[start + h] = emission;
                sums[h] += emission;
            }
        }
        self.update_total(sums);
    }

    fn init_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize) {
        let code = self.ambiguous[ambiguous_index];
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let hap = ((code >> h) & 1) != 0;
                let emission = if hap != allele { self.mismatch } else { 1.0 };
                self.prob[start + h] = emission;
                sums[h] += emission;
            }
        }
        self.update_total(sums);
    }

    fn init_missing(&mut self) {
        let probability =
            f64::from(1.0f32 / (HAPLOTYPES as f32 * self.conditioning_haplotypes as f32));
        self.prob.fill(probability);
        self.prob_sum_h.fill(f64::from(1.0f32 / HAPLOTYPES as f32));
        self.prob_sum_t = 1.0;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_hom_avx2(
        &mut self,
        locus: usize,
        relative_locus: usize,
        transition: f64,
    ) -> bool {
        let genotype_allele = self.hap0(locus);
        let rare_allele = self.rare_alleles[locus];
        if rare_allele >= 0 && genotype_allele != (rare_allele != 0) {
            return false;
        }
        let stay = 1.0 - transition;
        let factor = transition / (self.conditioning_haplotypes as f64 * self.prob_sum_t);
        let stay_factor = stay / self.prob_sum_t;
        let factor = _mm256_set1_pd(factor);
        let stay_factor = _mm256_set1_pd(stay_factor);
        let transferred0 = _mm256_mul_pd(_mm256_loadu_pd(self.prob_sum_h.as_ptr()), factor);
        let transferred1 = _mm256_mul_pd(_mm256_loadu_pd(self.prob_sum_h.as_ptr().add(4)), factor);
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mismatch = _mm256_set1_pd(self.mismatch);
        let matched = _mm256_set1_pd(1.0);
        let probability = self.prob.as_mut_ptr();
        for k in 0..self.conditioning_haplotypes {
            let emission: __m256d = if genotype_allele != self.allele(relative_locus, k) {
                mismatch
            } else {
                matched
            };
            let start = k * HAPLOTYPES;
            let value0 = _mm256_mul_pd(
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(probability.add(start)),
                    stay_factor,
                    transferred0,
                ),
                emission,
            );
            let value1 = _mm256_mul_pd(
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(probability.add(start + 4)),
                    stay_factor,
                    transferred1,
                ),
                emission,
            );
            sum0 = _mm256_add_pd(sum0, value0);
            sum1 = _mm256_add_pd(sum1, value1);
            _mm256_storeu_pd(probability.add(start), value0);
            _mm256_storeu_pd(probability.add(start + 4), value1);
        }
        let mut sums = [0.0; HAPLOTYPES];
        _mm256_storeu_pd(sums.as_mut_ptr(), sum0);
        _mm256_storeu_pd(sums.as_mut_ptr().add(4), sum1);
        self.update_total(sums);
        true
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_ambiguous_avx2(
        &mut self,
        relative_locus: usize,
        ambiguous_index: usize,
        transition: f64,
    ) {
        let code = self.ambiguous[ambiguous_index];
        let stay = 1.0 - transition;
        let factor = transition / (self.conditioning_haplotypes as f64 * self.prob_sum_t);
        let stay_factor = stay / self.prob_sum_t;
        let factor = _mm256_set1_pd(factor);
        let stay_factor = _mm256_set1_pd(stay_factor);
        let transferred0 = _mm256_mul_pd(_mm256_loadu_pd(self.prob_sum_h.as_ptr()), factor);
        let transferred1 = _mm256_mul_pd(_mm256_loadu_pd(self.prob_sum_h.as_ptr().add(4)), factor);
        let mut emission_when_zero = [0.0; HAPLOTYPES];
        let mut emission_when_one = [0.0; HAPLOTYPES];
        for h in 0..HAPLOTYPES {
            let hap = ((code >> h) & 1) != 0;
            emission_when_zero[h] = if hap { self.mismatch } else { 1.0 };
            emission_when_one[h] = if hap { 1.0 } else { self.mismatch };
        }
        let emission00 = _mm256_loadu_pd(emission_when_zero.as_ptr());
        let emission01 = _mm256_loadu_pd(emission_when_zero.as_ptr().add(4));
        let emission10 = _mm256_loadu_pd(emission_when_one.as_ptr());
        let emission11 = _mm256_loadu_pd(emission_when_one.as_ptr().add(4));
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let probability = self.prob.as_mut_ptr();
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let (emission0, emission1) = if allele {
                (emission10, emission11)
            } else {
                (emission00, emission01)
            };
            let start = k * HAPLOTYPES;
            let value0 = _mm256_mul_pd(
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(probability.add(start)),
                    stay_factor,
                    transferred0,
                ),
                emission0,
            );
            let value1 = _mm256_mul_pd(
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(probability.add(start + 4)),
                    stay_factor,
                    transferred1,
                ),
                emission1,
            );
            sum0 = _mm256_add_pd(sum0, value0);
            sum1 = _mm256_add_pd(sum1, value1);
            _mm256_storeu_pd(probability.add(start), value0);
            _mm256_storeu_pd(probability.add(start + 4), value1);
        }
        let mut sums = [0.0; HAPLOTYPES];
        _mm256_storeu_pd(sums.as_mut_ptr(), sum0);
        _mm256_storeu_pd(sums.as_mut_ptr().add(4), sum1);
        self.update_total(sums);
    }

    fn run_hom(&mut self, locus: usize, relative_locus: usize, transition: f64) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // The enclosing SHAPEIT common-phasing binary already requires AVX2 and FMA.
            return unsafe { self.run_hom_avx2(locus, relative_locus, transition) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let genotype_allele = self.hap0(locus);
            let rare_allele = self.rare_alleles[locus];
            if rare_allele >= 0 && genotype_allele != (rare_allele != 0) {
                return false;
            }
            let stay = 1.0 - transition;
            let factor = transition / (self.conditioning_haplotypes as f64 * self.prob_sum_t);
            let stay_factor = stay / self.prob_sum_t;
            let mut transferred = [0.0; HAPLOTYPES];
            for h in 0..HAPLOTYPES {
                transferred[h] = self.prob_sum_h[h] * factor;
            }
            let mut sums = [0.0; HAPLOTYPES];
            for k in 0..self.conditioning_haplotypes {
                let emission = if genotype_allele != self.allele(relative_locus, k) {
                    self.mismatch
                } else {
                    1.0
                };
                let start = k * HAPLOTYPES;
                for h in 0..HAPLOTYPES {
                    let value =
                        self.prob[start + h].mul_add(stay_factor, transferred[h]) * emission;
                    self.prob[start + h] = value;
                    sums[h] += value;
                }
            }
            self.update_total(sums);
            true
        }
    }

    fn run_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize, transition: f64) {
        #[cfg(target_arch = "x86_64")]
        {
            // The enclosing SHAPEIT common-phasing binary already requires AVX2 and FMA.
            unsafe { self.run_ambiguous_avx2(relative_locus, ambiguous_index, transition) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let code = self.ambiguous[ambiguous_index];
            let stay = 1.0 - transition;
            let factor = transition / (self.conditioning_haplotypes as f64 * self.prob_sum_t);
            let stay_factor = stay / self.prob_sum_t;
            let mut transferred = [0.0; HAPLOTYPES];
            for h in 0..HAPLOTYPES {
                transferred[h] = self.prob_sum_h[h] * factor;
            }
            let mut sums = [0.0; HAPLOTYPES];
            for k in 0..self.conditioning_haplotypes {
                let allele = self.allele(relative_locus, k);
                let start = k * HAPLOTYPES;
                for h in 0..HAPLOTYPES {
                    let hap = ((code >> h) & 1) != 0;
                    let emission = if hap != allele { self.mismatch } else { 1.0 };
                    let value =
                        self.prob[start + h].mul_add(stay_factor, transferred[h]) * emission;
                    self.prob[start + h] = value;
                    sums[h] += value;
                }
            }
            self.update_total(sums);
        }
    }

    fn run_missing(&mut self, transition: f64) {
        let stay = 1.0 - transition;
        let factor = transition / (self.conditioning_haplotypes as f64 * self.prob_sum_t);
        let stay_factor = stay / self.prob_sum_t;
        let mut transferred = [0.0; HAPLOTYPES];
        for h in 0..HAPLOTYPES {
            transferred[h] = self.prob_sum_h[h] * factor;
        }
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let value = self.prob[start + h].mul_add(stay_factor, transferred[h]);
                self.prob[start + h] = value;
                sums[h] += value;
            }
        }
        self.update_total(sums);
    }

    fn collapse_hom(&mut self, locus: usize, relative_locus: usize, transition: f64) {
        let genotype_allele = self.hap0(locus);
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let transferred = transition / self.conditioning_haplotypes as f64;
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let emission = if genotype_allele != self.allele(relative_locus, k) {
                self.mismatch
            } else {
                1.0
            };
            let value = self.prob_sum_k[k].mul_add(stay_factor, transferred) * emission;
            let start = k * HAPLOTYPES;
            for (h, sum) in sums.iter_mut().enumerate() {
                self.prob[start + h] = value;
                *sum += value;
            }
        }
        self.update_total(sums);
    }

    fn collapse_ambiguous(
        &mut self,
        relative_locus: usize,
        ambiguous_index: usize,
        transition: f64,
    ) {
        let code = self.ambiguous[ambiguous_index];
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let transferred = transition / self.conditioning_haplotypes as f64;
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let base = self.prob_sum_k[k].mul_add(stay_factor, transferred);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let hap = ((code >> h) & 1) != 0;
                let emission = if hap != allele { self.mismatch } else { 1.0 };
                let value = base * emission;
                self.prob[start + h] = value;
                sums[h] += value;
            }
        }
        self.update_total(sums);
    }

    fn collapse_missing(&mut self, transition: f64) {
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let transferred = transition / self.conditioning_haplotypes as f64;
        let mut sums = [0.0; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let value = self.prob_sum_k[k].mul_add(stay_factor, transferred);
            let start = k * HAPLOTYPES;
            for (h, sum) in sums.iter_mut().enumerate() {
                self.prob[start + h] = value;
                *sum += value;
            }
        }
        self.update_total(sums);
    }

    fn sum_conditioning_haplotypes(&mut self) {
        for k in 0..self.conditioning_haplotypes {
            let start = k * HAPLOTYPES;
            self.prob_sum_k[k] = sum8(&self.prob[start..start + HAPLOTYPES]);
        }
    }

    fn save_alpha(&mut self, relative_segment: usize, previous_locus: usize) {
        let state_start = relative_segment * self.prob.len();
        self.alpha[state_start..state_start + self.prob.len()].copy_from_slice(self.prob);
        let sum_start = relative_segment * HAPLOTYPES;
        self.alpha_sum[sum_start..sum_start + HAPLOTYPES].copy_from_slice(&self.prob_sum_h);
        self.alpha_sum_sum[relative_segment] = self.prob_sum_t;
        self.alpha_locus[relative_segment] = previous_locus as i32;
    }

    fn save_missing(&mut self, relative_missing: usize) {
        let state_start = relative_missing * self.prob.len();
        self.alpha_missing[state_start..state_start + self.prob.len()].copy_from_slice(self.prob);
        let sum_start = relative_missing * HAPLOTYPES;
        self.alpha_sum_missing[sum_start..sum_start + HAPLOTYPES].copy_from_slice(&self.prob_sum_h);
    }
}

impl DoubleEngine<'_> {
    fn transition_haplotypes(&mut self, relative_segment: usize, previous_locus: usize) -> bool {
        let alpha_segment = relative_segment - 1;
        let alpha_sum_total = self.alpha_sum_sum[alpha_segment];
        let alpha_locus = self.alpha_locus[alpha_segment] as usize;
        let transition = self.transition_probability(alpha_locus, previous_locus);
        let stay_factor = (1.0 - transition) / alpha_sum_total;
        let alpha_state_start = alpha_segment * self.prob.len();
        let alpha_sum_start = alpha_segment * HAPLOTYPES;
        let mut total = 0.0;

        for h1 in 0..HAPLOTYPES {
            let transferred = (self.alpha_sum[alpha_sum_start + h1] / alpha_sum_total) * transition
                / self.conditioning_haplotypes as f64;
            let row_start = h1 * HAPLOTYPES;
            let mut sums = [0.0; HAPLOTYPES];
            for k in 0..self.conditioning_haplotypes {
                let state_start = k * HAPLOTYPES;
                let alpha = self.alpha[alpha_state_start + state_start + h1]
                    .mul_add(stay_factor, transferred);
                for h2 in 0..HAPLOTYPES {
                    sums[h2] = alpha.mul_add(self.prob[state_start + h2], sums[h2]);
                }
            }
            self.h_probs[row_start..row_start + HAPLOTYPES].copy_from_slice(&sums);
            total += sum8(&sums);
        }
        self.sum_h_probs = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    fn transition_diplotypes_multiply(&mut self, previous: u64, current: u64) -> bool {
        let scaling = 1.0 / self.sum_h_probs;
        let mut total = 0.0;
        let mut index = 0usize;
        let mut previous_active = previous;
        while previous_active != 0 {
            let previous_diplotype = previous_active.trailing_zeros() as usize;
            previous_active &= previous_active - 1;
            let mut current_active = current;
            while current_active != 0 {
                let current_diplotype = current_active.trailing_zeros() as usize;
                current_active &= current_active - 1;
                let first = self.h_probs
                    [(previous_diplotype >> 3) * HAPLOTYPES + (current_diplotype >> 3)]
                    * scaling;
                let second = self.h_probs
                    [(previous_diplotype & 7) * HAPLOTYPES + (current_diplotype & 7)]
                    * scaling;
                let value = first * second;
                self.d_probs[index] = value;
                total += value;
                index += 1;
            }
        }
        self.sum_d_probs = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    fn transition_diplotypes_add(&mut self, previous: u64, current: u64) -> bool {
        let scaling = 1.0 / self.sum_h_probs;
        let mut total = 0.0;
        let mut index = 0usize;
        let mut previous_active = previous;
        while previous_active != 0 {
            let previous_diplotype = previous_active.trailing_zeros() as usize;
            previous_active &= previous_active - 1;
            let mut current_active = current;
            while current_active != 0 {
                let current_diplotype = current_active.trailing_zeros() as usize;
                current_active &= current_active - 1;
                let first = self.h_probs
                    [(previous_diplotype >> 3) * HAPLOTYPES + (current_diplotype >> 3)]
                    * scaling;
                let second =
                    self.h_probs[(previous_diplotype & 7) * HAPLOTYPES + (current_diplotype & 7)];
                let value = second.mul_add(scaling, first);
                self.d_probs[index] = value;
                total += value;
                index += 1;
            }
        }
        self.sum_d_probs = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    fn set_first_transitions(&mut self) -> bool {
        if !self.prob_sum_t.is_finite() || self.prob_sum_t < f64::MIN_POSITIVE {
            return true;
        }
        let scale = 1.0 / self.prob_sum_t;
        let mut probabilities = [0.0; 64];
        let mut total = 0.0;
        let mut count = 0usize;
        let mut active = self.diplotypes[0];
        while active != 0 {
            let diplotype = active.trailing_zeros() as usize;
            active &= active - 1;
            let value = (self.prob_sum_h[diplotype >> 3] * scale)
                * (self.prob_sum_h[diplotype & 7] * scale);
            probabilities[count] = value;
            total += value;
            count += 1;
        }
        // The dense HMM can retain finite mass while every diplotype allowed
        // by the first graph segment has underflowed to zero. This contraction
        // was the one initial-distribution normalization without an underflow
        // guard in the established implementation.
        if !total.is_finite() || total < f64::MIN_POSITIVE {
            return true;
        }
        let scale_diplotype = 1.0 / total;
        for (target, &value) in self.transition_probabilities[..count]
            .iter_mut()
            .zip(probabilities.iter())
        {
            *target = value * scale_diplotype;
        }
        false
    }

    fn set_other_transitions(
        &mut self,
        segment: usize,
        previous_locus: usize,
        transition_cursor: &mut isize,
    ) -> i32 {
        let relative_segment = segment - self.segment_first;
        if self.transition_haplotypes(relative_segment, previous_locus) {
            return -1;
        }
        let previous = self.diplotypes[segment - 1];
        let current = self.diplotypes[segment];
        let mut recovered = 0;
        if self.transition_diplotypes_multiply(previous, current) {
            if self.transition_diplotypes_add(previous, current) {
                return -2;
            }
            recovered = 1;
        }

        let transition_count = diplotype_count(previous) * diplotype_count(current);
        *transition_cursor -= transition_count as isize - 1;
        let start = *transition_cursor as usize;
        let scaling = 1.0 / self.sum_d_probs;
        for index in 0..transition_count {
            self.transition_probabilities[start + index] = self.d_probs[index] * scaling;
        }
        *transition_cursor -= 1;
        recovered
    }

    fn impute(&mut self, relative_locus: usize, relative_missing: usize, absolute_missing: usize) {
        let state_start = relative_missing * self.prob.len();
        let sum_start = relative_missing * HAPLOTYPES;
        let mut sums = [[0.0; HAPLOTYPES]; 2];
        for k in 0..self.conditioning_haplotypes {
            let allele = usize::from(self.allele(relative_locus, k));
            let conditioning_start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                // Preserve the established AVX load at AlphaSumMissing[1]
                // for lanes 4..7. This intentionally maps them to sums 1..4.
                let denominator_index = if h < 4 { h } else { h - 3 };
                let reciprocal = 1.0 / self.alpha_sum_missing[sum_start + denominator_index];
                let alpha = self.alpha_missing[state_start + conditioning_start + h] * reciprocal;
                sums[allele][h] = alpha.mul_add(self.prob[conditioning_start + h], sums[allele][h]);
            }
        }
        let output_start = absolute_missing * HAPLOTYPES;
        for h in 0..HAPLOTYPES {
            self.missing_probabilities[output_start + h] =
                (sums[1][h] / (sums[0][h] + sums[1][h])) as f32;
        }
    }

    fn forward(&mut self) {
        let mut segment = self.segment_first;
        let mut segment_locus = 0usize;
        let mut ambiguous_index = self.ambiguous_first;
        let mut missing_index = self.missing_first;
        let mut previous_locus = self.locus_first;

        for locus in self.locus_first..=self.locus_last {
            let relative_locus = locus - self.locus_first;
            let relative_missing = missing_index - self.missing_first;
            let ambiguous = self.is_ambiguous(locus);
            let missing = self.is_missing(locus);
            let homozygous = !(ambiguous || missing);
            let transition = if locus == self.locus_first {
                0.0
            } else {
                self.transition_probability(previous_locus, locus)
            };
            let mut update_previous = true;

            if relative_locus == 0 {
                if homozygous {
                    self.init_hom(locus, relative_locus);
                } else if ambiguous {
                    self.init_ambiguous(relative_locus, ambiguous_index);
                } else {
                    self.init_missing();
                }
            } else if segment_locus != 0 {
                if homozygous {
                    update_previous = self.run_hom(locus, relative_locus, transition);
                } else if ambiguous {
                    self.run_ambiguous(relative_locus, ambiguous_index, transition);
                } else {
                    self.run_missing(transition);
                }
            } else if homozygous {
                self.collapse_hom(locus, relative_locus, transition);
            } else if ambiguous {
                self.collapse_ambiguous(relative_locus, ambiguous_index, transition);
            } else {
                self.collapse_missing(transition);
            }
            if update_previous {
                previous_locus = locus;
            }

            if segment_locus + 1 == self.segment_lengths[segment] as usize {
                self.sum_conditioning_haplotypes();
                self.save_alpha(segment - self.segment_first, previous_locus);
            }
            if missing {
                self.save_missing(relative_missing);
                missing_index += 1;
            }

            segment_locus += 1;
            ambiguous_index += usize::from(ambiguous);
            if segment_locus >= self.segment_lengths[segment] as usize {
                segment += 1;
                segment_locus = 0;
            }
        }
    }

    fn backward(&mut self) -> i32 {
        let mut underflow_recovered = 0;
        let mut segment = self.segment_last;
        let mut segment_locus = self.segment_lengths[segment] as isize - 1;
        let ambiguous_count = (self.locus_first..=self.locus_last)
            .filter(|&locus| self.is_ambiguous(locus))
            .count();
        let missing_count = self.alpha_missing.len() / self.prob.len();
        let mut ambiguous_index = self.ambiguous_first as isize + ambiguous_count as isize - 1;
        let mut missing_index = self.missing_first as isize + missing_count as isize - 1;
        let mut transition_cursor = self.transition_last as isize;
        let mut previous_locus = self.locus_last;

        for locus in (self.locus_first..=self.locus_last).rev() {
            let relative_locus = locus - self.locus_first;
            let ambiguous = self.is_ambiguous(locus);
            let missing = self.is_missing(locus);
            let homozygous = !(ambiguous || missing);
            let transition = if locus == self.locus_last {
                0.0
            } else {
                self.transition_probability(previous_locus, locus)
            };
            let mut update_previous = true;

            if locus == self.locus_last {
                if homozygous {
                    self.init_hom(locus, relative_locus);
                } else if ambiguous {
                    self.init_ambiguous(relative_locus, ambiguous_index as usize);
                } else {
                    self.init_missing();
                }
            } else if segment_locus + 1 != self.segment_lengths[segment] as isize {
                if homozygous {
                    update_previous = self.run_hom(locus, relative_locus, transition);
                } else if ambiguous {
                    self.run_ambiguous(relative_locus, ambiguous_index as usize, transition);
                } else {
                    self.run_missing(transition);
                }
            } else if homozygous {
                self.collapse_hom(locus, relative_locus, transition);
            } else if ambiguous {
                self.collapse_ambiguous(relative_locus, ambiguous_index as usize, transition);
            } else {
                self.collapse_missing(transition);
            }
            if segment_locus == 0 {
                self.sum_conditioning_haplotypes();
            }
            if update_previous {
                previous_locus = locus;
            }

            if locus == 0 {
                if self.set_first_transitions() {
                    return -2;
                }
            }
            if segment_locus == 0 && locus != self.locus_first {
                let result =
                    self.set_other_transitions(segment, previous_locus, &mut transition_cursor);
                if result < 0 {
                    return result;
                }
                underflow_recovered += result;
            }
            if missing {
                let relative_missing = (missing_index - self.missing_first as isize) as usize;
                self.impute(relative_locus, relative_missing, missing_index as usize);
                missing_index -= 1;
            }

            segment_locus -= 1;
            ambiguous_index -= isize::from(ambiguous);
            if segment_locus < 0 && segment > 0 {
                segment -= 1;
                segment_locus = self.segment_lengths[segment] as isize - 1;
            }
        }
        underflow_recovered
    }

    fn run(&mut self) -> i32 {
        self.forward();
        self.backward()
    }
}

struct JobWindowInputs<'a> {
    job: &'a HmmJobV1,
    variants: &'a [u8],
    ambiguous: &'a [u8],
    segment_lengths: &'a [u16],
    diplotypes: &'a [u64],
    subset_haplotypes: &'a [u8],
    subset_stride: usize,
    conditioning_haplotypes: usize,
    locus_offset: u32,
    window: GenotypeWindowV1,
}

fn inclusive_count(first: i32, last: i32, allow_empty: bool) -> Result<usize, u32> {
    if last < first {
        return if allow_empty {
            Ok(0)
        } else {
            Err(STATUS_INVALID_DIMENSIONS)
        };
    }
    let first = usize_coordinate(first)?;
    let last = usize_coordinate(last)?;
    last.checked_sub(first)
        .and_then(|count| count.checked_add(1))
        .ok_or(STATUS_INTEGER_OVERFLOW)
}

unsafe fn run_job_window_double(
    inputs: &JobWindowInputs<'_>,
    scratch: &mut Vec<f64>,
    alpha_locus: &mut Vec<i32>,
    validation: &mut Option<DoubleJobValidation>,
) -> Result<i32, u32> {
    let segment_count = inclusive_count(
        inputs.window.start_segment,
        inputs.window.stop_segment,
        false,
    )?;
    let missing_count = inclusive_count(
        inputs.window.start_missing,
        inputs.window.stop_missing,
        true,
    )?;
    let layout = scratch_layout(inputs.conditioning_haplotypes, segment_count, missing_count)?;
    scratch.resize(layout.total, 0.0);
    alpha_locus.resize(segment_count, 0);
    let parameters = HmmSegmentDoubleV1 {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<HmmSegmentDoubleV1>() as u32,
        variants: inputs.variants.as_ptr(),
        variants_length: inputs.variants.len(),
        ambiguous: inputs.ambiguous.as_ptr(),
        ambiguous_length: inputs.ambiguous.len(),
        segment_lengths: inputs.segment_lengths.as_ptr(),
        segment_lengths_length: inputs.segment_lengths.len(),
        diplotypes: inputs.diplotypes.as_ptr(),
        diplotypes_length: inputs.diplotypes.len(),
        haplotypes: inputs.subset_haplotypes.as_ptr(),
        haplotypes_length: inputs.subset_haplotypes.len(),
        haplotype_stride: inputs.subset_stride,
        conditioning_haplotypes: inputs.conditioning_haplotypes,
        locus_offset: inputs.locus_offset,
        centimorgans: inputs.job.centimorgans,
        centimorgans_length: inputs.job.centimorgans_length,
        recombination: inputs.job.recombination,
        recombination_length: inputs.job.recombination_length,
        rare_alleles: inputs.job.rare_alleles,
        rare_alleles_length: inputs.job.rare_alleles_length,
        effective_population_size: inputs.job.effective_population_size,
        total_haplotypes: inputs.job.total_haplotypes,
        emission_match: inputs.job.emission_match,
        emission_mismatch: inputs.job.emission_mismatch,
        segment_first: inputs.window.start_segment,
        segment_last: inputs.window.stop_segment,
        locus_first: inputs.window.start_locus,
        locus_last: inputs.window.stop_locus,
        ambiguous_first: inputs.window.start_ambiguous,
        ambiguous_last: inputs.window.stop_ambiguous,
        missing_first: inputs.window.start_missing,
        missing_last: inputs.window.stop_missing,
        transition_first: inputs.window.start_transition,
        transition_last: inputs.window.stop_transition,
        transition_probabilities: inputs.job.transition_probabilities,
        transition_probabilities_length: inputs.job.transition_probabilities_length,
        missing_probabilities: inputs.job.missing_probabilities,
        missing_probabilities_length: inputs.job.missing_probabilities_length,
        scratch: scratch.as_mut_ptr(),
        scratch_length: scratch.len(),
        alpha_locus_scratch: alpha_locus.as_mut_ptr(),
        alpha_locus_scratch_length: alpha_locus.len(),
    };
    let validated_layout = if let Some(validation) = validation.as_ref() {
        validate_hmm_window(&parameters, &validation.graph)?
    } else {
        let graph = validate_hmm_graph(inputs.variants, inputs.segment_lengths, inputs.diplotypes)?;
        let layout = validate_hmm_window(&parameters, &graph)?;
        *validation = Some(DoubleJobValidation { graph });
        layout
    };
    Ok(run_segment_double_prevalidated(
        &parameters,
        validated_layout,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExtendedPrecisionOutcome {
    outcome: i32,
    used_log_domain: bool,
}

#[inline]
fn probability_is_valid<T>(probability: T) -> bool
where
    T: Copy + PartialOrd + From<u8>,
{
    probability >= T::from(0) && probability <= T::from(1)
}

/// Validate exactly the output blocks written by one window. Windows overlap
/// at their boundary segment, so a noninitial window begins after the first
/// segment's already-computed transition block.
unsafe fn window_hmm_output_is_valid(inputs: &JobWindowInputs<'_>) -> bool {
    let transitions = const_slice(
        inputs.job.transition_probabilities,
        inputs.job.transition_probabilities_length,
    );
    let segment_first = inputs.window.start_segment as usize;
    let segment_last = inputs.window.stop_segment as usize;
    let mut transition_cursor = if segment_first == 0 {
        0
    } else {
        inputs.window.start_transition as usize
    };
    let transition_segment_first = if segment_first == 0 {
        0
    } else {
        segment_first + 1
    };
    if transition_segment_first <= segment_last {
        for segment in transition_segment_first..=segment_last {
            let previous_count = if segment == 0 {
                1
            } else {
                diplotype_count(inputs.diplotypes[segment - 1])
            };
            let current_count = diplotype_count(inputs.diplotypes[segment]);
            let count = match previous_count.checked_mul(current_count) {
                Some(value) => value,
                None => return false,
            };
            let end = match transition_cursor.checked_add(count) {
                Some(value) if value <= transitions.len() => value,
                _ => return false,
            };
            let mut total = 0.0;
            for &probability in &transitions[transition_cursor..end] {
                if !probability.is_finite() || !probability_is_valid(probability) {
                    return false;
                }
                total += probability;
            }
            if !total.is_finite() || total <= 0.0 {
                return false;
            }
            transition_cursor = end;
        }
    }
    let expected_transition_end = match inputs.window.stop_transition.checked_add(1) {
        Some(value) => value as usize,
        None => return false,
    };
    if transition_cursor != expected_transition_end {
        return false;
    }

    if inputs.window.stop_missing >= inputs.window.start_missing {
        let first = match (inputs.window.start_missing as usize).checked_mul(HAPLOTYPES) {
            Some(value) => value,
            None => return false,
        };
        let end = match (inputs.window.stop_missing as usize)
            .checked_add(1)
            .and_then(|value| value.checked_mul(HAPLOTYPES))
        {
            Some(value) if value <= inputs.job.missing_probabilities_length => value,
            _ => return false,
        };
        let missing = const_slice(inputs.job.missing_probabilities, end);
        for &probability in &missing[first..end] {
            if !probability.is_finite() || !probability_is_valid(probability) {
                return false;
            }
        }
    }
    true
}

unsafe fn run_job_window_log_validated(
    inputs: &JobWindowInputs<'_>,
    scratch: &mut Vec<f64>,
    alpha_locus: &mut Vec<i32>,
) -> Result<i32, u32> {
    let outcome = log::run_job_window_log(inputs, scratch, alpha_locus)?;
    if outcome >= 0 && !window_hmm_output_is_valid(inputs) {
        Ok(-2)
    } else {
        Ok(outcome)
    }
}

/// Run the existing f64 replay first, escalating to the log semiring only when
/// f64 cannot retain a valid permitted mass. Both paths evaluate the same HMM;
/// only the numerical representation changes.
unsafe fn run_job_window_extended_precision(
    inputs: &JobWindowInputs<'_>,
    scratch: &mut Vec<f64>,
    alpha_locus: &mut Vec<i32>,
    validation: &mut Option<DoubleJobValidation>,
) -> Result<ExtendedPrecisionOutcome, u32> {
    let outcome = run_job_window_double(inputs, scratch, alpha_locus, validation)?;
    if outcome >= 0 && window_hmm_output_is_valid(inputs) {
        return Ok(ExtendedPrecisionOutcome {
            outcome,
            used_log_domain: false,
        });
    }

    let outcome = run_job_window_log_validated(inputs, scratch, alpha_locus)?;
    Ok(ExtendedPrecisionOutcome {
        outcome,
        used_log_domain: true,
    })
}

unsafe fn run_job_window_single(
    inputs: &JobWindowInputs<'_>,
    scratch: &mut Vec<f32>,
    alpha_locus: &mut Vec<i32>,
    indexes: &mut Vec<usize>,
) -> Result<i32, u32> {
    let segment_count = inclusive_count(
        inputs.window.start_segment,
        inputs.window.stop_segment,
        false,
    )?;
    let missing_count = inclusive_count(
        inputs.window.start_missing,
        inputs.window.stop_missing,
        true,
    )?;
    let layout = scratch_layout(inputs.conditioning_haplotypes, segment_count, missing_count)?;
    let index_length = segment_count
        .checked_mul(4)
        .and_then(|count| count.checked_add(1))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    scratch.resize(layout.total, 0.0);
    alpha_locus.resize(segment_count, 0);
    indexes.resize(index_length, 0);
    let parameters = HmmSegmentSingleV1 {
        abi_version: ABI_VERSION,
        struct_size: mem::size_of::<HmmSegmentSingleV1>() as u32,
        variants: inputs.variants.as_ptr(),
        variants_length: inputs.variants.len(),
        ambiguous: inputs.ambiguous.as_ptr(),
        ambiguous_length: inputs.ambiguous.len(),
        segment_lengths: inputs.segment_lengths.as_ptr(),
        segment_lengths_length: inputs.segment_lengths.len(),
        diplotypes: inputs.diplotypes.as_ptr(),
        diplotypes_length: inputs.diplotypes.len(),
        haplotypes: inputs.subset_haplotypes.as_ptr(),
        haplotypes_length: inputs.subset_haplotypes.len(),
        haplotype_stride: inputs.subset_stride,
        conditioning_haplotypes: inputs.conditioning_haplotypes,
        locus_offset: inputs.locus_offset,
        centimorgans: inputs.job.centimorgans,
        centimorgans_length: inputs.job.centimorgans_length,
        recombination: inputs.job.recombination,
        recombination_length: inputs.job.recombination_length,
        rare_alleles: inputs.job.rare_alleles,
        rare_alleles_length: inputs.job.rare_alleles_length,
        effective_population_size: inputs.job.effective_population_size,
        total_haplotypes: inputs.job.total_haplotypes,
        emission_match: inputs.job.emission_match as f32,
        emission_mismatch: inputs.job.emission_mismatch as f32,
        segment_first: inputs.window.start_segment,
        segment_last: inputs.window.stop_segment,
        locus_first: inputs.window.start_locus,
        locus_last: inputs.window.stop_locus,
        ambiguous_first: inputs.window.start_ambiguous,
        ambiguous_last: inputs.window.stop_ambiguous,
        missing_first: inputs.window.start_missing,
        missing_last: inputs.window.stop_missing,
        transition_first: inputs.window.start_transition,
        transition_last: inputs.window.stop_transition,
        transition_probabilities: inputs.job.transition_probabilities,
        transition_probabilities_length: inputs.job.transition_probabilities_length,
        missing_probabilities: inputs.job.missing_probabilities,
        missing_probabilities_length: inputs.job.missing_probabilities_length,
        scratch: scratch.as_mut_ptr(),
        scratch_length: scratch.len(),
        alpha_locus_scratch: alpha_locus.as_mut_ptr(),
        alpha_locus_scratch_length: alpha_locus.len(),
        index_scratch: indexes.as_mut_ptr(),
        index_scratch_length: indexes.len(),
    };
    let mut outcome = 0;
    let status = shapeit_hmm_run_segment_single_prevalidated_v1(&parameters, &mut outcome);
    if status == STATUS_OK {
        Ok(outcome)
    } else {
        Err(status)
    }
}

unsafe fn run_hmm_job_v1_cached(
    parameters: &HmmJobV1,
    result: &mut HmmJobResultV1,
    force_log_domain: bool,
    double_validation: &mut Option<DoubleJobValidation>,
) -> u32 {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size as usize != mem::size_of::<HmmJobV1>()
        || parameters.haplotype_stride == 0
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if parameters.graph.is_null() || parameters.conditioning_job.is_null() {
        return STATUS_NULL_POINTER;
    }
    for pointer_status in [
        require_const_pointer(parameters.haplotypes, parameters.haplotypes_length),
        require_const_pointer(parameters.centimorgans, parameters.centimorgans_length),
        require_const_pointer(parameters.recombination, parameters.recombination_length),
        require_const_pointer(parameters.rare_alleles, parameters.rare_alleles_length),
        require_mut_pointer(
            parameters.transition_probabilities,
            parameters.transition_probabilities_length,
        ),
        require_mut_pointer(
            parameters.missing_probabilities,
            parameters.missing_probabilities_length,
        ),
    ] {
        if let Err(status) = pointer_status {
            return status;
        }
    }

    let graph = &mut *parameters.graph;
    if !graph.is_built() {
        return STATUS_INVALID_DIMENSIONS;
    }
    let (variant_count, transition_count, missing_count) = graph.hmm_dimensions();
    let required_missing_probabilities = match missing_count.checked_mul(HAPLOTYPES) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if parameters.effective_population_size <= 0
        || parameters.total_haplotypes <= 0
        || !parameters.emission_match.is_finite()
        || !parameters.emission_mismatch.is_finite()
        || parameters.emission_match == 0.0
        || !(parameters.emission_match as f32).is_finite()
        || !(parameters.emission_mismatch as f32).is_finite()
        || parameters.emission_match as f32 == 0.0
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if parameters.centimorgans_length < variant_count
        || parameters.rare_alleles_length < variant_count
        || parameters.recombination_length < variant_count.saturating_sub(1)
        || parameters.transition_probabilities_length < transition_count
        || parameters.missing_probabilities_length < required_missing_probabilities
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    let mut require_double_precision = graph.requires_double_precision();
    let (variants, ambiguous, segment_lengths, diplotypes) = graph.hmm_arrays();
    let conditioning_job = &mut *parameters.conditioning_job;
    let ConditioningJobV1 {
        windows,
        states,
        subset_haplotypes,
        single_scratch,
        double_scratch,
        alpha_locus_scratch,
        index_scratch,
        ..
    } = conditioning_job;
    if windows.is_empty() || windows.len() != states.len() {
        return STATUS_INVALID_DIMENSIONS;
    }

    let mut local_result = HmmJobResultV1::default();
    for (window, conditioning_haplotypes) in windows.iter().zip(states.iter()) {
        if conditioning_haplotypes.len() < 2 {
            return STATUS_INVALID_DIMENSIONS;
        }
        let locus_first = match usize_coordinate(window.start_locus) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let locus_last = match usize_coordinate(window.stop_locus) {
            Ok(value) => value,
            Err(status) => return status,
        };
        if locus_first > locus_last {
            return STATUS_INVALID_DIMENSIONS;
        }
        let source_byte_first = locus_first >> 3;
        let source_byte_last = locus_last >> 3;
        let source_byte_count = match source_byte_last
            .checked_sub(source_byte_first)
            .and_then(|count| count.checked_add(1))
        {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        let subset_stride = match conditioning_haplotypes.len().checked_add(7) {
            Some(value) => value >> 3,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        let subset_rows = match source_byte_count.checked_mul(8) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        let subset_length = match subset_rows.checked_mul(subset_stride) {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
        subset_haplotypes.resize(subset_length, 0);
        let transpose_status = shapeit_bitmatrix_subset_transpose_v1(
            parameters.haplotypes,
            parameters.haplotypes_length,
            parameters.haplotype_stride,
            conditioning_haplotypes.as_ptr(),
            conditioning_haplotypes.len(),
            source_byte_first,
            source_byte_count,
            subset_haplotypes.as_mut_ptr(),
            subset_haplotypes.len(),
            subset_stride,
        );
        if transpose_status != STATUS_OK {
            return transpose_status;
        }

        let window_inputs = JobWindowInputs {
            job: parameters,
            variants,
            ambiguous,
            segment_lengths,
            diplotypes,
            subset_haplotypes,
            subset_stride,
            conditioning_haplotypes: conditioning_haplotypes.len(),
            locus_offset: (locus_first & 7) as u32,
            window: *window,
        };
        let outcome = if force_log_domain {
            match run_job_window_log_validated(&window_inputs, double_scratch, alpha_locus_scratch)
            {
                Ok(value) => value,
                Err(status) => return status,
            }
        } else if require_double_precision {
            match run_job_window_extended_precision(
                &window_inputs,
                double_scratch,
                alpha_locus_scratch,
                double_validation,
            ) {
                Ok(value) => {
                    if value.used_log_domain && value.outcome >= 0 {
                        local_result.underflow_recovered_precision += 1;
                    }
                    value.outcome
                }
                Err(status) => return status,
            }
        } else {
            let single_outcome = match run_job_window_single(
                &window_inputs,
                single_scratch,
                alpha_locus_scratch,
                index_scratch,
            ) {
                Ok(value) => value,
                Err(status) => return status,
            };
            if single_outcome == 0 {
                single_outcome
            } else {
                let extended_outcome = match run_job_window_extended_precision(
                    &window_inputs,
                    double_scratch,
                    alpha_locus_scratch,
                    double_validation,
                ) {
                    Ok(value) => value,
                    Err(status) => return status,
                };
                require_double_precision = true;
                if extended_outcome.outcome >= 0 {
                    local_result.underflow_recovered_precision += 1;
                }
                extended_outcome.outcome
            }
        };
        local_result.windows_completed += 1;
        if outcome < 0 {
            local_result.fatal_outcome = outcome;
            break;
        }
        local_result.underflow_recovered_summing = match local_result
            .underflow_recovered_summing
            .checked_add(outcome)
        {
            Some(value) => value,
            None => return STATUS_INTEGER_OVERFLOW,
        };
    }

    if require_double_precision {
        graph.require_double_precision();
    }
    *result = local_result;
    STATUS_OK
}

unsafe fn run_hmm_job_v1(
    parameters: &HmmJobV1,
    result: &mut HmmJobResultV1,
    force_log_domain: bool,
) -> u32 {
    let mut double_validation = None;
    run_hmm_job_v1_cached(parameters, result, force_log_domain, &mut double_validation)
}

#[no_mangle]
/// Run every HMM window in one Rust-owned conditioning job.
///
/// The function retains subset-transpose and HMM scratch capacity in the
/// worker-local conditioning job. It also owns the established precision
/// fallback decision and persists the f64 decision in the genotype graph.
///
/// # Safety
///
/// `parameters` and `result` must be valid for their types. The opaque graph
/// and conditioning job must be live and exclusively borrowed. Every other
/// buffer must be valid for its stated length and mutable buffers must not
/// overlap inputs or each other.
pub unsafe extern "C" fn shapeit_hmm_run_job_v1(
    parameters: *const HmmJobV1,
    result: *mut HmmJobResultV1,
) -> u32 {
    if parameters.is_null() || result.is_null() {
        return STATUS_NULL_POINTER;
    }
    run_hmm_job_v1(&*parameters, &mut *result, false)
}

#[inline]
fn transition_probabilities_are_valid(probabilities: &[f64]) -> bool {
    probabilities
        .iter()
        .all(|&probability| probability.is_finite() && probability_is_valid(probability))
}

#[inline]
fn missing_probabilities_are_valid(probabilities: &[f32]) -> bool {
    probabilities
        .iter()
        .all(|&probability| probability.is_finite() && probability_is_valid(probability))
}

fn sample_complete_hmm_output(
    graph: &mut GenotypeGraphV1,
    transition_probabilities: &[f64],
    missing_probabilities: &[f32],
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
) -> Result<(), SampleError> {
    // Sampling follows only one path through the graph and can therefore miss
    // an invalid transition in an unvisited block. Validate the complete HMM
    // output before sampling so every numerical failure takes the precision
    // retry path.
    if !transition_probabilities_are_valid(transition_probabilities)
        || !missing_probabilities_are_valid(missing_probabilities)
    {
        return Err(SampleError::DegenerateDistribution);
    }
    sample_graph_current(
        graph,
        transition_probabilities,
        missing_probabilities,
        seed,
        domain,
        iteration,
        item,
    )
}

#[no_mangle]
/// Run one complete common-phasing sample job, including its MCMC stage action.
///
/// Current transition and missing probabilities remain in reusable Rust-owned
/// worker storage. On a successful HMM run the genotype graph is sampled, then
/// optionally pruned or accumulated according to `stage`.
///
/// # Safety
///
/// `parameters` and `result` must be valid for their types. The opaque graph
/// and conditioning job must be live and exclusively borrowed. Every other
/// non-empty input buffer must be readable for its stated length.
pub unsafe extern "C" fn shapeit_hmm_run_phase_job_v1(
    parameters: *const HmmPhaseJobV1,
    result: *mut HmmJobResultV1,
) -> u32 {
    if parameters.is_null() || result.is_null() {
        return STATUS_NULL_POINTER;
    }
    let parameters = &*parameters;
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size as usize != mem::size_of::<HmmPhaseJobV1>()
        || parameters.stage > STAGE_MAIN
        || (parameters.stage == STAGE_PRUNE
            && (!parameters.prune_threshold.is_finite()
                || parameters.prune_threshold < 0.0
                || parameters.prune_threshold > 1.0))
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if parameters.graph.is_null() || parameters.conditioning_job.is_null() {
        return STATUS_NULL_POINTER;
    }
    if !(*parameters.graph).is_built() {
        return STATUS_INVALID_DIMENSIONS;
    }
    let initially_requires_double = (*parameters.graph).requires_double_precision();
    let (_, transition_count, missing_count) = (*parameters.graph).hmm_dimensions();
    let missing_probability_count = match missing_count.checked_mul(HAPLOTYPES) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };

    let conditioning_job = &mut *parameters.conditioning_job;
    let mut transition_probabilities = mem::take(&mut conditioning_job.transition_probabilities);
    let mut missing_probabilities = mem::take(&mut conditioning_job.missing_probabilities);
    if transition_probabilities.len() < transition_count {
        transition_probabilities.resize(transition_count, 0.0);
    }
    if missing_probabilities.len() < missing_probability_count {
        missing_probabilities.resize(missing_probability_count, 0.0);
    }

    let hmm_parameters = HmmJobV1 {
        abi_version: parameters.abi_version,
        struct_size: mem::size_of::<HmmJobV1>() as u32,
        graph: parameters.graph,
        conditioning_job: parameters.conditioning_job,
        haplotypes: parameters.haplotypes,
        haplotypes_length: parameters.haplotypes_length,
        haplotype_stride: parameters.haplotype_stride,
        centimorgans: parameters.centimorgans,
        centimorgans_length: parameters.centimorgans_length,
        recombination: parameters.recombination,
        recombination_length: parameters.recombination_length,
        rare_alleles: parameters.rare_alleles,
        rare_alleles_length: parameters.rare_alleles_length,
        effective_population_size: parameters.effective_population_size,
        total_haplotypes: parameters.total_haplotypes,
        emission_match: parameters.emission_match,
        emission_mismatch: parameters.emission_mismatch,
        transition_probabilities: transition_probabilities.as_mut_ptr(),
        transition_probabilities_length: transition_probabilities.len(),
        missing_probabilities: missing_probabilities.as_mut_ptr(),
        missing_probabilities_length: missing_probabilities.len(),
    };
    let mut local_result = HmmJobResultV1::default();
    let mut double_validation = None;
    let mut hmm_status = run_hmm_job_v1_cached(
        &hmm_parameters,
        &mut local_result,
        false,
        &mut double_validation,
    );
    let mut sample_result = if hmm_status == STATUS_OK && local_result.fatal_outcome == 0 {
        sample_complete_hmm_output(
            &mut *parameters.graph,
            &transition_probabilities[..transition_count],
            &missing_probabilities[..missing_probability_count],
            parameters.sample_seed,
            parameters.sample_domain,
            parameters.sample_iteration,
            parameters.sample_item,
        )
    } else {
        Err(SampleError::Status(hmm_status))
    };

    // A zero-mass conditional row means the single-precision windows did not
    // retain a usable path. Recompute the complete job in double precision so
    // every transition block belongs to the same precision regime.
    if sample_result == Err(SampleError::DegenerateDistribution) && !initially_requires_double {
        (*parameters.graph).require_double_precision();
        let mut retry_result = HmmJobResultV1::default();
        hmm_status = run_hmm_job_v1_cached(
            &hmm_parameters,
            &mut retry_result,
            false,
            &mut double_validation,
        );
        if hmm_status == STATUS_OK && retry_result.fatal_outcome == 0 {
            retry_result.underflow_recovered_precision =
                match retry_result.underflow_recovered_precision.checked_add(1) {
                    Some(value) => value,
                    None => return STATUS_INTEGER_OVERFLOW,
                };
            local_result = retry_result;
            sample_result = sample_complete_hmm_output(
                &mut *parameters.graph,
                &transition_probabilities[..transition_count],
                &missing_probabilities[..missing_probability_count],
                parameters.sample_seed,
                parameters.sample_domain,
                parameters.sample_iteration,
                parameters.sample_item,
            );
        } else {
            local_result = retry_result;
            sample_result = Err(SampleError::Status(hmm_status));
        }
    }

    // f64 can retain a finite total while still emitting an unusable
    // conditional row or missing probability. If complete validation or the
    // sampler finds that case, replay every window in the log semiring once.
    if sample_result == Err(SampleError::DegenerateDistribution)
        && hmm_status == STATUS_OK
        && local_result.fatal_outcome == 0
    {
        (*parameters.graph).require_double_precision();
        let prior_precision_recoveries = local_result.underflow_recovered_precision;
        let mut log_result = HmmJobResultV1::default();
        hmm_status = run_hmm_job_v1_cached(
            &hmm_parameters,
            &mut log_result,
            true,
            &mut double_validation,
        );
        if hmm_status == STATUS_OK && log_result.fatal_outcome == 0 {
            log_result.underflow_recovered_precision =
                match prior_precision_recoveries.checked_add(1) {
                    Some(value) => value,
                    None => return STATUS_INTEGER_OVERFLOW,
                };
            local_result = log_result;
            sample_result = sample_complete_hmm_output(
                &mut *parameters.graph,
                &transition_probabilities[..transition_count],
                &missing_probabilities[..missing_probability_count],
                parameters.sample_seed,
                parameters.sample_domain,
                parameters.sample_iteration,
                parameters.sample_item,
            );
        } else {
            local_result = log_result;
            sample_result = Err(SampleError::Status(hmm_status));
        }
    }

    let operation_status = if hmm_status != STATUS_OK || local_result.fatal_outcome != 0 {
        hmm_status
    } else {
        match sample_result {
            Err(SampleError::Status(status)) => status,
            Err(SampleError::DegenerateDistribution) => STATUS_INVALID_DIMENSIONS,
            Ok(()) => match parameters.stage {
                STAGE_BURN => STATUS_OK,
                STAGE_PRUNE => shapeit_genotype_graph_prune_v1(
                    parameters.graph,
                    transition_probabilities.as_ptr(),
                    transition_count,
                    parameters.prune_threshold,
                ),
                STAGE_MAIN => shapeit_genotype_graph_store_v1(
                    parameters.graph,
                    transition_probabilities.as_ptr(),
                    transition_count,
                    missing_probabilities.as_ptr(),
                    missing_probability_count,
                ),
                _ => STATUS_INVALID_DIMENSIONS,
            },
        }
    };

    let conditioning_job = &mut *parameters.conditioning_job;
    conditioning_job.transition_probabilities = transition_probabilities;
    conditioning_job.missing_probabilities = missing_probabilities;
    if operation_status == STATUS_OK {
        *result = local_result;
    }
    operation_status
}

#[no_mangle]
/// Build conditioning state and execute one complete common-phasing sample job.
///
/// This is the coarse production boundary: the opaque conditioning job is
/// rebuilt in place and immediately consumed by the Rust HMM and MCMC stage.
///
/// # Safety
///
/// `parameters`, `job`, and `result` must be valid. `job` must contain null or
/// a live conditioning job. All nested buffers follow the safety contracts of
/// the conditioning-build and phase-job entry points.
pub unsafe extern "C" fn shapeit_common_phase_job_run_v1(
    parameters: *const CommonPhaseJobV1,
    job: *mut *mut ConditioningJobV1,
    result: *mut HmmJobResultV1,
) -> u32 {
    if parameters.is_null() || job.is_null() || result.is_null() {
        return STATUS_NULL_POINTER;
    }
    common_phase_job_run(&*parameters, job, &mut *result, None)
}

unsafe fn common_phase_job_run(
    parameters: &CommonPhaseJobV1,
    job: *mut *mut ConditioningJobV1,
    result: &mut HmmJobResultV1,
    shared_validation: Option<&ConditioningSharedLayout>,
) -> u32 {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<CommonPhaseJobV1>()
        || parameters.conditioning.graph.cast_mut() != parameters.phase.graph
        || parameters.conditioning.haplotypes != parameters.phase.haplotypes
        || parameters.conditioning.haplotypes_length != parameters.phase.haplotypes_length
        || parameters.conditioning.haplotype_stride != parameters.phase.haplotype_stride
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let conditioning_status = match shared_validation {
        Some(validation) => {
            conditioning_graph_job_build_prevalidated_v1(&parameters.conditioning, job, validation)
        }
        None => shapeit_conditioning_graph_job_build_v1(&parameters.conditioning, job),
    };
    if conditioning_status != STATUS_OK {
        return conditioning_status;
    }
    let mut phase = parameters.phase;
    phase.conditioning_job = *job;
    shapeit_hmm_run_phase_job_v1(&phase, result)
}

fn record_common_failure(shared: &CommonIterationShared<'_>, status: u32, sample: usize) {
    if shared
        .status
        .compare_exchange(STATUS_OK, status, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        shared.failed_sample.store(sample, Ordering::SeqCst);
    }
}

fn run_common_worker(worker: &mut CommonWorkerV1, shared: &CommonIterationShared<'_>) {
    worker.reset_iteration();
    loop {
        if shared.status.load(Ordering::SeqCst) != STATUS_OK
            || shared.fatal_outcome.load(Ordering::SeqCst) != 0
        {
            break;
        }
        let sample = shared.next_sample.fetch_add(1, Ordering::SeqCst);
        if sample >= shared.graph_addresses.len() {
            break;
        }

        let mut parameters = shared.template.0;
        let graph = shared.graph_addresses[sample] as *mut GenotypeGraphV1;
        parameters.conditioning.graph = graph;
        parameters.conditioning.target_individual = sample;
        parameters.conditioning.target_individual_count = shared.graph_addresses.len();
        parameters.conditioning.haploid_individuals = shared.haploid_individuals.as_ptr();
        parameters.conditioning.haploid_individuals_length = shared.haploid_individuals.len();
        parameters.conditioning.window_item = sample as u64;
        parameters.conditioning.fallback_item = sample as u64;
        parameters.phase.graph = graph;
        parameters.phase.conditioning_job = core::ptr::null_mut();
        parameters.phase.sample_item = sample as u64;

        let mut job_result = HmmJobResultV1::default();
        let status = unsafe {
            common_phase_job_run(
                &parameters,
                &mut worker.conditioning_job,
                &mut job_result,
                Some(&shared.conditioning_validation),
            )
        };
        if status != STATUS_OK {
            record_common_failure(shared, status, sample);
            break;
        }
        if job_result.fatal_outcome != 0 {
            if shared
                .fatal_outcome
                .compare_exchange(
                    0,
                    job_result.fatal_outcome,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                shared.failed_sample.store(sample, Ordering::SeqCst);
            }
            break;
        }

        worker.iteration.underflow_recovered_summing +=
            job_result.underflow_recovered_summing as u64;
        worker.iteration.underflow_recovered_precision +=
            u64::from(job_result.underflow_recovered_precision);
        let conditioning = unsafe { &*worker.conditioning_job };
        for (window_index, ((window, states), &used_fallback)) in conditioning
            .windows
            .iter()
            .zip(conditioning.states.iter())
            .zip(conditioning.used_fallback.iter())
            .enumerate()
        {
            let start_locus = match usize::try_from(window.start_locus) {
                Ok(value) => value,
                Err(_) => {
                    record_common_failure(shared, STATUS_OUT_OF_BOUNDS, sample);
                    return;
                }
            };
            let stop_locus = match usize::try_from(window.stop_locus) {
                Ok(value) => value,
                Err(_) => {
                    record_common_failure(shared, STATUS_OUT_OF_BOUNDS, sample);
                    return;
                }
            };
            if start_locus > stop_locus || stop_locus >= shared.base_pair_positions.len() {
                record_common_failure(shared, STATUS_OUT_OF_BOUNDS, sample);
                return;
            }
            worker
                .iteration
                .conditioning_states
                .push(states.len() as f64);
            let width = i64::from(shared.base_pair_positions[stop_locus])
                - i64::from(shared.base_pair_positions[start_locus]);
            worker
                .iteration
                .window_megabases
                .push(width as f64 * 1.0e-6);
            if used_fallback {
                worker.iteration.fallbacks.push(CommonFallbackV1 {
                    sample,
                    window: window_index,
                    states: states.len(),
                });
            }
        }

        {
            let _guard = match shared.serialized_output.lock() {
                Ok(value) => value,
                Err(_) => {
                    record_common_failure(shared, STATUS_THREAD_FAILURE, sample);
                    return;
                }
            };
            let registry = unsafe { &mut *(shared.ibd2_registry_address as *mut Ibd2TracksV1) };
            if let Err(status) = registry.push(sample, &conditioning.tracks) {
                record_common_failure(shared, status, sample);
                return;
            }
            let completed = shared.completed_samples.fetch_add(1, Ordering::SeqCst) + 1;
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
}

#[no_mangle]
/// Create a persistent common-phase worker pool and copy its stable graph and
/// ploidy table. Per-worker conditioning and HMM workspaces are retained across
/// every iteration run.
///
/// # Safety
///
/// Every graph must remain live until the pool is freed. `graphs`,
/// `haploid_individuals`, and `workers` must be valid for their stated access.
pub unsafe extern "C" fn shapeit_common_workers_create_v1(
    worker_count: usize,
    graphs: *const *mut GenotypeGraphV1,
    graph_count: usize,
    haploid_individuals: *const u8,
    haploid_individuals_length: usize,
    workers: *mut *mut CommonWorkersV1,
) -> u32 {
    if workers.is_null() {
        return STATUS_NULL_POINTER;
    }
    if worker_count == 0 || graph_count == 0 || haploid_individuals_length != graph_count {
        return STATUS_INVALID_DIMENSIONS;
    }
    if let Err(status) = require_const_pointer(graphs, graph_count) {
        return status;
    }
    if let Err(status) = require_const_pointer(haploid_individuals, haploid_individuals_length) {
        return status;
    }
    let graphs = const_slice(graphs, graph_count);
    let haploid_individuals = const_slice(haploid_individuals, haploid_individuals_length);
    if haploid_individuals.iter().any(|&value| value > 1) {
        return STATUS_INVALID_DIMENSIONS;
    }
    let mut graph_addresses = Vec::with_capacity(graph_count);
    let mut variant_count = None;
    for &graph in graphs {
        if graph.is_null() {
            return STATUS_NULL_POINTER;
        }
        let graph_ref = &*graph;
        if !graph_ref.is_built() {
            return STATUS_INVALID_DIMENSIONS;
        }
        let current_variant_count = graph_ref.hmm_dimensions().0;
        if variant_count.is_some_and(|expected| expected != current_variant_count) {
            return STATUS_INVALID_DIMENSIONS;
        }
        variant_count = Some(current_variant_count);
        graph_addresses.push(graph as usize);
    }
    let worker_count = core::cmp::min(worker_count, graph_count);
    let value = CommonWorkersV1 {
        workers: (0..worker_count).map(|_| CommonWorkerV1::new()).collect(),
        graph_addresses,
        haploid_individuals: haploid_individuals.to_vec(),
        variant_count: variant_count.unwrap_or(0),
        fallbacks: Vec::new(),
    };
    *workers = Box::into_raw(Box::new(value));
    STATUS_OK
}

#[no_mangle]
/// Free a common-phase worker pool. Null is accepted.
///
/// # Safety
///
/// `workers` must be null or a live pool returned by the create function, and
/// it must be freed at most once after all iteration calls have returned.
pub unsafe extern "C" fn shapeit_common_workers_free_v1(workers: *mut CommonWorkersV1) {
    if !workers.is_null() {
        drop(Box::from_raw(workers));
    }
}

#[no_mangle]
/// Schedule and run a complete common-phasing iteration in Rust.
///
/// Each graph is mutably accessed by exactly one scoped worker. Shared PBWT,
/// haplotype, map, and model buffers are read-only for the duration of the
/// call. Detected IBD2 tracks and progress callbacks are serialized.
///
/// # Safety
///
/// `workers`, `parameters`, and `result` must be valid and exclusively used by
/// this call. All buffers in the sample template must remain live and immutable
/// until the call returns. The IBD2 registry must be live and must describe the
/// same target individuals as the worker pool.
pub unsafe extern "C" fn shapeit_common_workers_run_iteration_v1(
    workers: *mut CommonWorkersV1,
    parameters: *const CommonIterationV1,
    result: *mut CommonIterationResultV1,
) -> u32 {
    if workers.is_null() || parameters.is_null() || result.is_null() {
        return STATUS_NULL_POINTER;
    }
    let workers = &mut *workers;
    let parameters = &*parameters;
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<CommonIterationV1>()
        || parameters.sample_template.abi_version != ABI_VERSION
        || parameters.sample_template.struct_size < mem::size_of::<CommonPhaseJobV1>()
        || parameters.ibd2_registry.is_null()
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    if let Err(status) = require_const_pointer(
        parameters.base_pair_positions,
        parameters.base_pair_positions_length,
    ) {
        return status;
    }
    if parameters.base_pair_positions_length < workers.variant_count
        || (*parameters.ibd2_registry).individual_count() != workers.graph_addresses.len()
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    let base_pair_positions = const_slice(
        parameters.base_pair_positions,
        parameters.base_pair_positions_length,
    );
    let mut effective_conditioning = parameters.sample_template.conditioning;
    effective_conditioning.target_individual = 0;
    effective_conditioning.target_individual_count = workers.graph_addresses.len();
    effective_conditioning.haploid_individuals = workers.haploid_individuals.as_ptr();
    effective_conditioning.haploid_individuals_length = workers.haploid_individuals.len();
    let conditioning_validation = match validate_conditioning_graph_job_shared_v1(
        &effective_conditioning,
        workers.variant_count,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let shared = CommonIterationShared {
        template: SharedCommonTemplate(parameters.sample_template),
        conditioning_validation,
        graph_addresses: &workers.graph_addresses,
        haploid_individuals: &workers.haploid_individuals,
        base_pair_positions,
        ibd2_registry_address: parameters.ibd2_registry as usize,
        progress: parameters.progress,
        progress_context_address: parameters.progress_context as usize,
        next_sample: AtomicUsize::new(0),
        completed_samples: AtomicUsize::new(0),
        status: AtomicU32::new(STATUS_OK),
        failed_sample: AtomicUsize::new(usize::MAX),
        fatal_outcome: AtomicI32::new(0),
        serialized_output: Mutex::new(()),
    };

    if workers.workers.len() == 1 {
        run_common_worker(&mut workers.workers[0], &shared);
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers.workers.len());
            for worker in &mut workers.workers {
                let shared = &shared;
                match thread::Builder::new().spawn_scoped(scope, move || {
                    run_common_worker(worker, shared);
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => record_common_failure(shared, STATUS_THREAD_FAILURE, usize::MAX),
                }
            }
            for handle in handles {
                if handle.join().is_err() {
                    record_common_failure(&shared, STATUS_THREAD_FAILURE, usize::MAX);
                }
            }
        });
    }

    let mut conditioning_states = CommonStats::default();
    let mut window_megabases = CommonStats::default();
    let mut underflow_recovered_summing = 0u64;
    let mut underflow_recovered_precision = 0u64;
    workers.fallbacks.clear();
    for worker in &workers.workers {
        underflow_recovered_summing = match underflow_recovered_summing
            .checked_add(worker.iteration.underflow_recovered_summing)
        {
            Some(value) => value,
            None => {
                record_common_failure(&shared, STATUS_INTEGER_OVERFLOW, usize::MAX);
                0
            }
        };
        underflow_recovered_precision = match underflow_recovered_precision
            .checked_add(worker.iteration.underflow_recovered_precision)
        {
            Some(value) => value,
            None => {
                record_common_failure(&shared, STATUS_INTEGER_OVERFLOW, usize::MAX);
                0
            }
        };
        conditioning_states.merge(worker.iteration.conditioning_states);
        window_megabases.merge(worker.iteration.window_megabases);
        workers
            .fallbacks
            .extend(worker.iteration.fallbacks.iter().copied());
    }
    workers
        .fallbacks
        .sort_unstable_by_key(|fallback| (fallback.sample, fallback.window));

    *result = CommonIterationResultV1 {
        underflow_recovered_summing,
        underflow_recovered_precision,
        fatal_outcome: shared.fatal_outcome.load(Ordering::SeqCst),
        failed_sample: shared.failed_sample.load(Ordering::SeqCst),
        windows: conditioning_states.count,
        conditioning_states_mean: conditioning_states.mean(),
        conditioning_states_sd: conditioning_states.standard_deviation(),
        window_megabases_mean: window_megabases.mean(),
        window_megabases_sd: window_megabases.standard_deviation(),
    };
    shared.status.load(Ordering::SeqCst)
}

fn record_pbwt_failure(shared: &CommonPbwtShared<'_>, status: u32, chunk: usize) {
    if shared
        .status
        .compare_exchange(STATUS_OK, status, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        shared.failed_chunk.store(chunk, Ordering::SeqCst);
    }
}

fn run_pbwt_worker(shared: &CommonPbwtShared<'_>) {
    loop {
        if shared.status.load(Ordering::SeqCst) != STATUS_OK {
            break;
        }
        let chunk = shared.next_chunk.fetch_add(1, Ordering::SeqCst);
        if chunk >= shared.parameters.chunk_count {
            break;
        }
        let parameters = shared.parameters;
        let status =
            unsafe { pbwt_select_chunk_prevalidated_v1(&shared.job, &shared.validation, chunk) };
        if status != STATUS_OK {
            record_pbwt_failure(shared, status, chunk);
            break;
        }
        let _guard = match shared.serialized_progress.lock() {
            Ok(value) => value,
            Err(_) => {
                record_pbwt_failure(shared, STATUS_THREAD_FAILURE, chunk);
                break;
            }
        };
        let completed = shared.completed_chunks.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(progress) = parameters.progress {
            unsafe {
                progress(
                    completed,
                    parameters.chunk_count,
                    parameters.progress_context,
                );
            }
        }
    }
}

unsafe fn run_common_pbwt_selection(
    worker_count: usize,
    parameters: &CommonPbwtSelectionV1,
    failed_chunk: &mut usize,
) -> u32 {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<CommonPbwtSelectionV1>()
        || parameters.chunk_count == 0
        || parameters.ibd2_registry.is_null()
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    for pointer_status in [
        require_const_pointer(parameters.chunk_starts, parameters.chunk_count),
        require_const_pointer(parameters.haplotypes, parameters.haplotypes_length),
        require_const_pointer(
            parameters.evaluated_sites,
            parameters.evaluated_sites_length,
        ),
        require_const_pointer(parameters.site_groups, parameters.site_groups_length),
        require_const_pointer(parameters.site_chunks, parameters.site_chunks_length),
        require_const_pointer(parameters.selected_sites, parameters.selected_sites_length),
        require_const_pointer(parameters.neighbors, parameters.neighbors_length),
    ] {
        if let Err(status) = pointer_status {
            return status;
        }
    }
    let chunk_starts = const_slice(parameters.chunk_starts, parameters.chunk_count);
    if chunk_starts.iter().any(|&start| start < 0) {
        return STATUS_OUT_OF_BOUNDS;
    }
    let job = PbwtSelectJobV1 {
        haplotypes: parameters.haplotypes,
        haplotypes_length: parameters.haplotypes_length,
        haplotype_stride: parameters.haplotype_stride,
        site_count: parameters.site_count,
        haplotype_count: parameters.haplotype_count,
        target_individual_count: parameters.target_individual_count,
        evaluated_sites: parameters.evaluated_sites,
        evaluated_sites_length: parameters.evaluated_sites_length,
        selected_sites: parameters.selected_sites,
        selected_sites_length: parameters.selected_sites_length,
        site_groups: parameters.site_groups,
        site_groups_length: parameters.site_groups_length,
        group_count: parameters.group_count,
        site_chunks: parameters.site_chunks,
        site_chunks_length: parameters.site_chunks_length,
        chunk_starts: parameters.chunk_starts,
        chunk_count: parameters.chunk_count,
        depth: parameters.depth,
        ibd2: parameters.ibd2_registry,
        neighbors: parameters.neighbors,
        neighbors_length: parameters.neighbors_length,
    };
    let validation = match validate_pbwt_select_job_v1(&job) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status = shapeit_pbwt_select_sites_v1(
        parameters.evaluated_sites,
        parameters.evaluated_sites_length,
        parameters.site_groups,
        parameters.site_groups_length,
        parameters.group_count,
        parameters.seed,
        parameters.domain,
        parameters.iteration,
        parameters.selected_sites,
        parameters.selected_sites_length,
    );
    if status != STATUS_OK {
        return status;
    }
    slice::from_raw_parts_mut(parameters.neighbors, parameters.neighbors_length).fill(-1);

    let shared = CommonPbwtShared {
        parameters,
        job,
        validation,
        next_chunk: AtomicUsize::new(0),
        completed_chunks: AtomicUsize::new(0),
        status: AtomicU32::new(STATUS_OK),
        failed_chunk: AtomicUsize::new(usize::MAX),
        serialized_progress: Mutex::new(()),
    };
    let execution_threads = core::cmp::min(worker_count, parameters.chunk_count);
    if execution_threads == 1 {
        run_pbwt_worker(&shared);
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(execution_threads);
            for _ in 0..execution_threads {
                let shared = &shared;
                match thread::Builder::new().spawn_scoped(scope, move || {
                    run_pbwt_worker(shared);
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(_) => record_pbwt_failure(shared, STATUS_THREAD_FAILURE, usize::MAX),
                }
            }
            for handle in handles {
                if handle.join().is_err() {
                    record_pbwt_failure(&shared, STATUS_THREAD_FAILURE, usize::MAX);
                }
            }
        });
    }
    let status = shared.status.load(Ordering::SeqCst);
    *failed_chunk = shared.failed_chunk.load(Ordering::SeqCst);
    if status != STATUS_OK {
        return status;
    }
    shapeit_pbwt_transpose_neighbors_v1(
        parameters.neighbors,
        parameters.neighbors_length,
        parameters.target_individual_count * 2,
        parameters.group_count,
        parameters.depth,
    )
}

#[no_mangle]
/// Execute the complete mutable core of one common-phasing iteration.
///
/// The transaction selects PBWT neighbours, phases every sample, collapses
/// IBD2 tracks, refreshes sampled target haplotypes, and transposes them for the
/// next PBWT pass. Caller-owned bitmatrices are updated in place.
///
/// # Safety
///
/// `workers`, `parameters`, and `result` must be valid and exclusively used for
/// the duration of this call. All nested buffers must satisfy their individual
/// PBWT, common-iteration, and bitmatrix ABI contracts and must not overlap
/// except where the same read-only buffer is intentionally repeated.
pub unsafe extern "C" fn shapeit_common_workers_run_full_iteration_v1(
    workers: *mut CommonWorkersV1,
    parameters: *const CommonFullIterationV1,
    result: *mut CommonFullIterationResultV1,
) -> u32 {
    if workers.is_null() || parameters.is_null() || result.is_null() {
        return STATUS_NULL_POINTER;
    }
    let parameters = &*parameters;
    let worker_count = (*workers).workers.len();
    let graph_count = (*workers).graph_addresses.len();
    let variant_count = (*workers).variant_count;
    let mut local_result = CommonFullIterationResultV1 {
        failed_pbwt_chunk: usize::MAX,
        ..CommonFullIterationResultV1::default()
    };
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<CommonFullIterationV1>()
        || parameters.pbwt.site_count != variant_count
        || parameters.pbwt.target_individual_count != graph_count
        || parameters.pbwt.ibd2_registry.cast_mut() != parameters.phase.ibd2_registry
        || parameters.pbwt.haplotypes != parameters.variant_major
        || parameters.pbwt.haplotypes_length != parameters.variant_major_length
        || parameters.pbwt.haplotype_stride != parameters.variant_major_stride
        || parameters.phase.sample_template.conditioning.selected_sites
            != parameters.pbwt.selected_sites
        || parameters.phase.sample_template.conditioning.pbwt_neighbors != parameters.pbwt.neighbors
        || parameters.phase.sample_template.conditioning.haplotypes != parameters.haplotype_major
        || parameters.phase.sample_template.phase.haplotypes != parameters.haplotype_major
    {
        *result = local_result;
        return STATUS_INVALID_DIMENSIONS;
    }

    let started = Instant::now();
    let status = run_common_pbwt_selection(
        worker_count,
        &parameters.pbwt,
        &mut local_result.failed_pbwt_chunk,
    );
    local_result.pbwt_seconds = started.elapsed().as_secs_f64();
    if status != STATUS_OK {
        *result = local_result;
        return status;
    }

    let started = Instant::now();
    let status = shapeit_common_workers_run_iteration_v1(
        workers,
        &parameters.phase,
        &mut local_result.phase,
    );
    local_result.hmm_seconds = started.elapsed().as_secs_f64();
    if status != STATUS_OK || local_result.phase.fatal_outcome != 0 {
        *result = local_result;
        return status;
    }

    let started = Instant::now();
    local_result.ibd2 = (*parameters.phase.ibd2_registry).collapse();
    local_result.ibd2_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let workers_ref = &*workers;
    let mut variant_views = Vec::with_capacity(workers_ref.graph_addresses.len());
    let mut variants_length = None;
    for &address in &workers_ref.graph_addresses {
        let graph = &*(address as *const GenotypeGraphV1);
        let variants = graph.hmm_arrays().0;
        if variants_length.is_some_and(|expected| expected != variants.len()) {
            *result = local_result;
            return STATUS_INVALID_DIMENSIONS;
        }
        variants_length = Some(variants.len());
        variant_views.push(variants.as_ptr());
    }
    let status = shapeit_bitmatrix_refresh_haplotypes_v1(
        variant_views.as_ptr(),
        variant_views.len(),
        variants_length.unwrap_or(0),
        variant_count,
        0,
        parameters.haplotype_major,
        parameters.haplotype_major_length,
        parameters.haplotype_major_rows,
        parameters.haplotype_major_stride,
    );
    local_result.haplotype_refresh_seconds = started.elapsed().as_secs_f64();
    if status != STATUS_OK {
        *result = local_result;
        return status;
    }

    let started = Instant::now();
    let target_haplotype_count = match graph_count.checked_mul(2) {
        Some(value) => value,
        None => {
            *result = local_result;
            return STATUS_INTEGER_OVERFLOW;
        }
    };
    let max_rows = match target_haplotype_count.checked_add(7) {
        Some(value) => value & !7,
        None => {
            *result = local_result;
            return STATUS_INTEGER_OVERFLOW;
        }
    };
    let max_cols = match variant_count.checked_add(7) {
        Some(value) => value & !7,
        None => {
            *result = local_result;
            return STATUS_INTEGER_OVERFLOW;
        }
    };
    let status = shapeit_bitmatrix_transpose_v1(
        parameters.haplotype_major,
        parameters.haplotype_major_length,
        parameters.haplotype_major_rows,
        parameters.haplotype_major_stride,
        max_rows,
        max_cols,
        parameters.variant_major,
        parameters.variant_major_length,
        parameters.variant_major_stride,
    );
    local_result.transpose_seconds = started.elapsed().as_secs_f64();
    *result = local_result;
    status
}

#[no_mangle]
/// Return the number of fallback events retained by the most recent iteration.
///
/// # Safety
///
/// `workers` must be null or point to a live, idle worker pool.
pub unsafe extern "C" fn shapeit_common_workers_fallback_count_v1(
    workers: *const CommonWorkersV1,
) -> usize {
    if workers.is_null() {
        0
    } else {
        (*workers).fallbacks.len()
    }
}

#[no_mangle]
/// Borrow one fallback event from the most recent iteration.
///
/// # Safety
///
/// `workers` and `fallback` must be valid and the pool must be idle.
pub unsafe extern "C" fn shapeit_common_workers_fallback_v1(
    workers: *const CommonWorkersV1,
    index: usize,
    fallback: *mut CommonFallbackV1,
) -> u32 {
    if workers.is_null() || fallback.is_null() {
        return STATUS_NULL_POINTER;
    }
    let workers = &*workers;
    if index >= workers.fallbacks.len() {
        return STATUS_OUT_OF_BOUNDS;
    }
    *fallback = workers.fallbacks[index];
    STATUS_OK
}

#[no_mangle]
pub extern "C" fn shapeit_hmm_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Return the caller-owned f64 workspace required by one double-precision window.
///
/// # Safety
///
/// `scratch_length` must point to writable storage for one `size_t`. Invalid
/// dimensions are reported without writing it.
pub unsafe extern "C" fn shapeit_hmm_double_scratch_len_v1(
    conditioning_haplotypes: usize,
    segment_count: usize,
    missing_count: usize,
    scratch_length: *mut usize,
) -> u32 {
    if scratch_length.is_null() {
        return STATUS_NULL_POINTER;
    }
    let layout = match scratch_layout(conditioning_haplotypes, segment_count, missing_count) {
        Ok(value) => value,
        Err(status) => return status,
    };
    *scratch_length = layout.total;
    STATUS_OK
}

unsafe fn run_segment_double_prevalidated(
    parameters: &HmmSegmentDoubleV1,
    layout: ValidatedLayout,
) -> i32 {
    let variants = const_slice(parameters.variants, parameters.variants_length);
    let ambiguous = const_slice(parameters.ambiguous, parameters.ambiguous_length);
    let segment_lengths = const_slice(
        parameters.segment_lengths,
        parameters.segment_lengths_length,
    );
    let diplotypes = const_slice(parameters.diplotypes, parameters.diplotypes_length);
    let haplotypes = const_slice(parameters.haplotypes, parameters.haplotypes_length);
    let centimorgans = const_slice(parameters.centimorgans, parameters.centimorgans_length);
    let recombination = const_slice(parameters.recombination, parameters.recombination_length);
    let rare_alleles = const_slice(parameters.rare_alleles, parameters.rare_alleles_length);
    let transition_probabilities = mut_slice(
        parameters.transition_probabilities,
        parameters.transition_probabilities_length,
    );
    let missing_probabilities = mut_slice(
        parameters.missing_probabilities,
        parameters.missing_probabilities_length,
    );
    let scratch = mut_slice(parameters.scratch, layout.scratch.total);
    let alpha_locus = mut_slice(parameters.alpha_locus_scratch, layout.scratch.segment_count);
    scratch.fill(0.0);
    alpha_locus.fill(0);

    let (prob, scratch) = scratch.split_at_mut(layout.scratch.states);
    let (prob_sum_k, scratch) = scratch.split_at_mut(parameters.conditioning_haplotypes);
    let alpha_length = layout.scratch.segment_count * layout.scratch.states;
    let (alpha, scratch) = scratch.split_at_mut(alpha_length);
    let alpha_sum_length = layout.scratch.segment_count * HAPLOTYPES;
    let (alpha_sum, scratch) = scratch.split_at_mut(alpha_sum_length);
    let (alpha_sum_sum, scratch) = scratch.split_at_mut(layout.scratch.segment_count);
    let alpha_missing_length = layout.scratch.missing_count * layout.scratch.states;
    let (alpha_missing, scratch) = scratch.split_at_mut(alpha_missing_length);
    let alpha_sum_missing_length = layout.scratch.missing_count * HAPLOTYPES;
    let (alpha_sum_missing, remainder) = scratch.split_at_mut(alpha_sum_missing_length);
    debug_assert!(remainder.is_empty());

    let mut engine = DoubleEngine {
        variants,
        ambiguous,
        segment_lengths,
        diplotypes,
        haplotypes,
        haplotype_stride: parameters.haplotype_stride,
        conditioning_haplotypes: parameters.conditioning_haplotypes,
        locus_offset: parameters.locus_offset as usize,
        centimorgans,
        recombination,
        rare_alleles,
        effective_population_size: parameters.effective_population_size,
        total_haplotypes: parameters.total_haplotypes,
        mismatch: parameters.emission_mismatch / parameters.emission_match,
        segment_first: layout.segment_first,
        segment_last: layout.segment_last,
        locus_first: layout.locus_first,
        locus_last: layout.locus_last,
        ambiguous_first: layout.ambiguous_first,
        missing_first: layout.missing_first,
        transition_last: layout.transition_last,
        prob,
        prob_sum_k,
        alpha,
        alpha_sum,
        alpha_sum_sum,
        alpha_missing,
        alpha_sum_missing,
        alpha_locus,
        transition_probabilities,
        missing_probabilities,
        prob_sum_t: 0.0,
        prob_sum_h: [0.0; HAPLOTYPES],
        sum_h_probs: 0.0,
        sum_d_probs: 0.0,
        h_probs: [0.0; HAPLOTYPES * HAPLOTYPES],
        d_probs: [0.0; HAPLOTYPES * HAPLOTYPES * HAPLOTYPES * HAPLOTYPES],
    };
    engine.run()
}

#[no_mangle]
/// Run one complete double-precision common-phasing HMM window.
///
/// # Safety
///
/// `parameters` and `outcome` must be valid for their types. Every non-empty
/// buffer in `parameters` must be valid for its stated length. Mutable buffers
/// must not overlap each other or any input buffer. Invalid layouts are
/// reported before any caller-owned output buffer is written.
pub unsafe extern "C" fn shapeit_hmm_run_segment_double_v1(
    parameters: *const HmmSegmentDoubleV1,
    outcome: *mut i32,
) -> u32 {
    if parameters.is_null() || outcome.is_null() {
        return STATUS_NULL_POINTER;
    }
    let parameters = &*parameters;
    for result in [
        require_const_pointer(parameters.variants, parameters.variants_length),
        require_const_pointer(parameters.ambiguous, parameters.ambiguous_length),
        require_const_pointer(
            parameters.segment_lengths,
            parameters.segment_lengths_length,
        ),
        require_const_pointer(parameters.diplotypes, parameters.diplotypes_length),
        require_const_pointer(parameters.haplotypes, parameters.haplotypes_length),
        require_const_pointer(parameters.centimorgans, parameters.centimorgans_length),
        require_const_pointer(parameters.recombination, parameters.recombination_length),
        require_const_pointer(parameters.rare_alleles, parameters.rare_alleles_length),
        require_mut_pointer(
            parameters.transition_probabilities,
            parameters.transition_probabilities_length,
        ),
        require_mut_pointer(
            parameters.missing_probabilities,
            parameters.missing_probabilities_length,
        ),
        require_mut_pointer(parameters.scratch, parameters.scratch_length),
        require_mut_pointer(
            parameters.alpha_locus_scratch,
            parameters.alpha_locus_scratch_length,
        ),
    ] {
        if let Err(status) = result {
            return status;
        }
    }

    let variants = const_slice(parameters.variants, parameters.variants_length);
    let segment_lengths = const_slice(
        parameters.segment_lengths,
        parameters.segment_lengths_length,
    );
    let diplotypes = const_slice(parameters.diplotypes, parameters.diplotypes_length);
    let layout = match validate(parameters, variants, segment_lengths, diplotypes) {
        Ok(value) => value,
        Err(status) => return status,
    };
    *outcome = run_segment_double_prevalidated(parameters, layout);
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    fn reset_double_graph_validation_scans() {
        DOUBLE_GRAPH_VALIDATION_SCANS.with(|scans| scans.set(0));
    }

    fn double_graph_validation_scans() -> usize {
        DOUBLE_GRAPH_VALIDATION_SCANS.with(core::cell::Cell::get)
    }

    #[test]
    fn whole_job_runs_owned_windows_and_retains_workspace() {
        let variants = [0u8];
        let mut graph = ptr::null_mut();
        let graph_status = unsafe {
            crate::genotype::shapeit_genotype_graph_create_v1(
                variants.as_ptr(),
                variants.len(),
                1,
                &mut graph,
            )
        };
        assert_eq!(graph_status, STATUS_OK);

        let mut conditioning_job = ConditioningJobV1::default();
        conditioning_job.windows.push(GenotypeWindowV1 {
            start_locus: 0,
            start_segment: 0,
            start_ambiguous: 0,
            start_missing: 0,
            start_transition: 64,
            stop_locus: 0,
            stop_segment: 0,
            stop_ambiguous: -1,
            stop_missing: -1,
            stop_transition: 63,
        });
        conditioning_job.states.push((0..8).collect());

        let haplotypes = [0u8; 8];
        let centimorgans = [0.0f32];
        let rare_alleles = [-1i8];
        let mut transitions = [0.0f64; 64];
        let parameters = HmmJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmJobV1>() as u32,
            graph,
            conditioning_job: &mut conditioning_job,
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: ptr::null(),
            recombination_length: 0,
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
        };
        let mut result = HmmJobResultV1::default();
        reset_double_graph_validation_scans();
        let status = unsafe { shapeit_hmm_run_job_v1(&parameters, &mut result) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(result.fatal_outcome, 0);
        assert_eq!(result.windows_completed, 1);
        assert_eq!(result.underflow_recovered_precision, 0);
        assert!((transitions.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(!conditioning_job.subset_haplotypes.is_empty());
        assert!(!conditioning_job.single_scratch.is_empty());
        assert_eq!(double_graph_validation_scans(), 0);

        transitions.fill(0.0);
        transitions[0] = f64::NAN;
        let mut forced_log_result = HmmJobResultV1::default();
        let forced_log_status =
            unsafe { run_hmm_job_v1(&parameters, &mut forced_log_result, true) };
        assert_eq!(forced_log_status, STATUS_OK);
        assert_eq!(forced_log_result.fatal_outcome, 0);
        assert_eq!(forced_log_result.windows_completed, 1);
        assert!((transitions.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let phase_parameters = HmmPhaseJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmPhaseJobV1>() as u32,
            graph,
            conditioning_job: &mut conditioning_job,
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: ptr::null(),
            recombination_length: 0,
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            stage: STAGE_BURN,
            prune_threshold: 0.999,
            sample_seed: 15_052_011,
            sample_domain: 2,
            sample_iteration: 3,
            sample_item: 4,
        };
        let mut phase_result = HmmJobResultV1::default();
        let phase_status =
            unsafe { shapeit_hmm_run_phase_job_v1(&phase_parameters, &mut phase_result) };
        assert_eq!(phase_status, STATUS_OK);
        assert_eq!(phase_result.fatal_outcome, 0);
        assert_eq!(phase_result.windows_completed, 1);
        assert_eq!(conditioning_job.transition_probabilities.len(), 64);

        let repeated_window = conditioning_job.windows[0];
        let repeated_states = conditioning_job.states[0].clone();
        conditioning_job.windows.push(repeated_window);
        conditioning_job.states.push(repeated_states);
        unsafe { (*graph).require_double_precision() };
        reset_double_graph_validation_scans();
        let mut double_result = HmmJobResultV1::default();
        let double_status = unsafe { shapeit_hmm_run_job_v1(&parameters, &mut double_result) };
        assert_eq!(double_status, STATUS_OK);
        assert_eq!(double_result.fatal_outcome, 0);
        assert_eq!(double_result.windows_completed, 2);
        assert_eq!(double_graph_validation_scans(), 1);

        unsafe { crate::genotype::shapeit_genotype_graph_free_v1(graph) };
    }

    #[test]
    fn common_workers_validate_conditioning_shared_arrays_once_per_iteration() {
        let variants = [0u8];
        let mut graph0 = ptr::null_mut();
        let mut graph1 = ptr::null_mut();
        for graph in [&mut graph0, &mut graph1] {
            let status = unsafe {
                crate::genotype::shapeit_genotype_graph_create_v1(
                    variants.as_ptr(),
                    variants.len(),
                    1,
                    graph,
                )
            };
            assert_eq!(status, STATUS_OK);
        }
        let graphs = [graph0, graph1];
        let haploid_individuals = [0u8; 2];
        let mut workers = ptr::null_mut();
        let status = unsafe {
            shapeit_common_workers_create_v1(
                1,
                graphs.as_ptr(),
                graphs.len(),
                haploid_individuals.as_ptr(),
                haploid_individuals.len(),
                &mut workers,
            )
        };
        assert_eq!(status, STATUS_OK);

        let centimorgans_f64 = [0.0f64];
        let centimorgans_f32 = [0.0f32];
        let base_pair_positions = [1i32];
        let mut selected_sites = [0u8];
        let site_grouping = [0i32];
        let pbwt_neighbors = [-1i32; 4];
        let haplotypes = [0u8; 4];
        let rare_alleles = [-1i8];
        let mut registry = Ibd2TracksV1::new(2, &centimorgans_f32).unwrap();
        let sample_template = CommonPhaseJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<CommonPhaseJobV1>(),
            conditioning: ConditioningGraphBuildV1 {
                abi_version: ABI_VERSION,
                struct_size: mem::size_of::<ConditioningGraphBuildV1>(),
                graph: ptr::null(),
                centimorgans: centimorgans_f64.as_ptr(),
                centimorgans_length: centimorgans_f64.len(),
                minimum_window_centimorgans: 1.0,
                selected_sites: selected_sites.as_ptr(),
                selected_sites_length: selected_sites.len(),
                site_grouping: site_grouping.as_ptr(),
                site_grouping_length: site_grouping.len(),
                pbwt_neighbors: pbwt_neighbors.as_ptr(),
                pbwt_neighbors_length: pbwt_neighbors.len(),
                pbwt_depth: 1,
                pbwt_group_count: 1,
                target_individual: 0,
                target_individual_count: graphs.len(),
                haplotype_count: haplotypes.len(),
                haploid_individuals: ptr::null(),
                haploid_individuals_length: 0,
                haplotypes: haplotypes.as_ptr(),
                haplotypes_length: haplotypes.len(),
                haplotype_stride: 1,
                maximum_heterozygote_mismatch: 0.75,
                window_seed: 15_052_011,
                window_domain: 2,
                window_iteration: 3,
                window_item: 0,
                fallback_seed: 15_052_011,
                fallback_domain: 8,
                fallback_iteration: 3,
                fallback_item: 0,
            },
            phase: HmmPhaseJobV1 {
                abi_version: ABI_VERSION,
                struct_size: mem::size_of::<HmmPhaseJobV1>() as u32,
                graph: ptr::null_mut(),
                conditioning_job: ptr::null_mut(),
                haplotypes: haplotypes.as_ptr(),
                haplotypes_length: haplotypes.len(),
                haplotype_stride: 1,
                centimorgans: centimorgans_f32.as_ptr(),
                centimorgans_length: centimorgans_f32.len(),
                recombination: ptr::null(),
                recombination_length: 0,
                rare_alleles: rare_alleles.as_ptr(),
                rare_alleles_length: rare_alleles.len(),
                effective_population_size: 15_000,
                total_haplotypes: haplotypes.len() as i32,
                emission_match: f64::from(0.9999f32),
                emission_mismatch: f64::from(0.0001f32),
                stage: STAGE_BURN,
                prune_threshold: 0.999,
                sample_seed: 15_052_011,
                sample_domain: 2,
                sample_iteration: 3,
                sample_item: 0,
            },
        };
        let iteration = CommonIterationV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<CommonIterationV1>(),
            sample_template,
            base_pair_positions: base_pair_positions.as_ptr(),
            base_pair_positions_length: base_pair_positions.len(),
            ibd2_registry: &mut registry,
            progress: None,
            progress_context: ptr::null_mut(),
        };

        crate::conditioning::reset_conditioning_shared_validation_scans();
        for expected_scans in [1, 2] {
            let mut result = CommonIterationResultV1::default();
            let status = unsafe {
                shapeit_common_workers_run_iteration_v1(workers, &iteration, &mut result)
            };
            assert_eq!(status, STATUS_OK);
            assert_eq!(result.failed_sample, usize::MAX);
            assert_eq!(
                crate::conditioning::conditioning_shared_validation_scans(),
                expected_scans
            );
        }

        selected_sites[0] = 2;
        assert_eq!(selected_sites[0], 2);
        let mut rejected_result = CommonIterationResultV1::default();
        let status = unsafe {
            shapeit_common_workers_run_iteration_v1(workers, &iteration, &mut rejected_result)
        };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(
            crate::conditioning::conditioning_shared_validation_scans(),
            3
        );

        unsafe {
            shapeit_common_workers_free_v1(workers);
            crate::genotype::shapeit_genotype_graph_free_v1(graph0);
            crate::genotype::shapeit_genotype_graph_free_v1(graph1);
        }
    }

    #[test]
    fn common_pbwt_validates_once_and_rejects_out_of_range_tail_before_writes() {
        let haplotypes = [0x30u8, 0x50, 0xa0, 0xc0];
        let evaluated_sites = [1u8; 4];
        let mut selected_sites = [0u8; 4];
        let site_groups = [0i32, 0, 1, 1];
        let mut site_chunks = [0i32, 0, 1, 1];
        let chunk_starts = [0i32, 0];
        let mut ibd2 = Ibd2TracksV1::new(2, &[0.0, 1.0, 2.0, 3.0]).unwrap();
        let mut neighbors = [-1i32; 16];
        let parameters = CommonPbwtSelectionV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<CommonPbwtSelectionV1>(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            site_count: 4,
            haplotype_count: 4,
            target_individual_count: 2,
            evaluated_sites: evaluated_sites.as_ptr(),
            evaluated_sites_length: evaluated_sites.len(),
            selected_sites: selected_sites.as_mut_ptr(),
            selected_sites_length: selected_sites.len(),
            site_groups: site_groups.as_ptr(),
            site_groups_length: site_groups.len(),
            group_count: 2,
            site_chunks: site_chunks.as_ptr(),
            site_chunks_length: site_chunks.len(),
            chunk_starts: chunk_starts.as_ptr(),
            chunk_count: chunk_starts.len(),
            depth: 1,
            ibd2_registry: &mut ibd2,
            neighbors: neighbors.as_mut_ptr(),
            neighbors_length: neighbors.len(),
            seed: 15_052_011,
            domain: 1,
            iteration: 3,
            progress: None,
            progress_context: ptr::null_mut(),
        };

        let mut reference_selected = [0u8; 4];
        let status = unsafe {
            shapeit_pbwt_select_sites_v1(
                evaluated_sites.as_ptr(),
                evaluated_sites.len(),
                site_groups.as_ptr(),
                site_groups.len(),
                2,
                parameters.seed,
                parameters.domain,
                parameters.iteration,
                reference_selected.as_mut_ptr(),
                reference_selected.len(),
            )
        };
        assert_eq!(status, STATUS_OK);
        let mut reference_neighbors = [-1i32; 16];
        for chunk in 0..2 {
            let status = unsafe {
                crate::pbwt::shapeit_pbwt_select_chunk_v1(
                    haplotypes.as_ptr(),
                    haplotypes.len(),
                    1,
                    4,
                    4,
                    2,
                    evaluated_sites.as_ptr(),
                    evaluated_sites.len(),
                    reference_selected.as_ptr(),
                    reference_selected.len(),
                    site_groups.as_ptr(),
                    site_groups.len(),
                    2,
                    site_chunks.as_ptr(),
                    site_chunks.len(),
                    chunk,
                    chunk_starts[chunk] as usize,
                    1,
                    &ibd2,
                    reference_neighbors.as_mut_ptr(),
                    reference_neighbors.len(),
                )
            };
            assert_eq!(status, STATUS_OK);
        }
        let status = unsafe {
            shapeit_pbwt_transpose_neighbors_v1(
                reference_neighbors.as_mut_ptr(),
                reference_neighbors.len(),
                4,
                2,
                1,
            )
        };
        assert_eq!(status, STATUS_OK);

        crate::pbwt::reset_pbwt_select_validation_scans();
        let mut failed_chunk = usize::MAX;
        let status = unsafe { run_common_pbwt_selection(1, &parameters, &mut failed_chunk) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(failed_chunk, usize::MAX);
        assert_eq!(crate::pbwt::pbwt_select_validation_scans(), 1);
        assert_eq!(selected_sites, reference_selected);
        assert_eq!(neighbors, reference_neighbors);
        let expected_selected = selected_sites;
        let expected_neighbors = neighbors;

        let status = unsafe { run_common_pbwt_selection(1, &parameters, &mut failed_chunk) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(selected_sites, expected_selected);
        assert_eq!(neighbors, expected_neighbors);
        assert_eq!(crate::pbwt::pbwt_select_validation_scans(), 2);

        site_chunks[3] = 2;
        assert_eq!(site_chunks[3], 2);
        selected_sites.fill(0xa5);
        neighbors.fill(77);
        let selected_before = selected_sites;
        let neighbors_before = neighbors;
        failed_chunk = 123;
        let status = unsafe { run_common_pbwt_selection(1, &parameters, &mut failed_chunk) };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(selected_sites, selected_before);
        assert_eq!(neighbors, neighbors_before);
        assert_eq!(failed_chunk, 123);
        assert_eq!(crate::pbwt::pbwt_select_validation_scans(), 3);
    }

    #[test]
    fn fp64_graph_validation_is_reused_for_noninitial_windows() {
        let variants = [0u8; 2];
        let segment_lengths = [1u16; 3];
        let diplotypes = [1u64; 3];
        let haplotypes = [0u8; 8];
        let centimorgans = [0.0f32, 0.01, 0.02];
        let recombination = [0.01f32; 2];
        let rare_alleles = [-1i8; 3];
        let mut transitions = [7.0f64; 3];
        let job = HmmJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmJobV1>() as u32,
            graph: ptr::null_mut(),
            conditioning_job: ptr::null_mut(),
            haplotypes: ptr::null(),
            haplotypes_length: 0,
            haplotype_stride: 1,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: recombination.as_ptr(),
            recombination_length: recombination.len(),
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
        };
        let first = JobWindowInputs {
            job: &job,
            variants: &variants,
            ambiguous: &[],
            segment_lengths: &segment_lengths,
            diplotypes: &diplotypes,
            subset_haplotypes: &haplotypes,
            subset_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            window: GenotypeWindowV1 {
                start_locus: 0,
                start_segment: 0,
                start_ambiguous: 0,
                start_missing: 0,
                start_transition: 1,
                stop_locus: 1,
                stop_segment: 1,
                stop_ambiguous: -1,
                stop_missing: -1,
                stop_transition: 1,
            },
        };
        let second = JobWindowInputs {
            job: &job,
            variants: &variants,
            ambiguous: &[],
            segment_lengths: &segment_lengths,
            diplotypes: &diplotypes,
            subset_haplotypes: &haplotypes,
            subset_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 1,
            window: GenotypeWindowV1 {
                start_locus: 1,
                start_segment: 1,
                start_ambiguous: 0,
                start_missing: 0,
                start_transition: 2,
                stop_locus: 2,
                stop_segment: 2,
                stop_ambiguous: -1,
                stop_missing: -1,
                stop_transition: 2,
            },
        };
        let mut scratch = Vec::new();
        let mut alpha_locus = Vec::new();
        let mut validation = None;

        reset_double_graph_validation_scans();
        assert_eq!(
            unsafe {
                run_job_window_double(&first, &mut scratch, &mut alpha_locus, &mut validation)
            },
            Ok(0)
        );
        assert_eq!(
            unsafe {
                run_job_window_double(&second, &mut scratch, &mut alpha_locus, &mut validation)
            },
            Ok(0)
        );
        assert_eq!(transitions, [1.0; 3]);
        assert_eq!(double_graph_validation_scans(), 1);
    }

    #[test]
    fn scratch_size_checks_overflow() {
        assert_eq!(
            scratch_layout(8, 2, 1).unwrap().total,
            8 * 8 + 8 + 2 * 8 * 8 + 2 * 8 + 2 + 8 * 8 + 8
        );
        assert_eq!(scratch_layout(0, 2, 1), Err(STATUS_INVALID_DIMENSIONS));
        assert_eq!(
            scratch_layout(usize::MAX, 2, 1),
            Err(STATUS_INTEGER_OVERFLOW)
        );
    }

    #[test]
    fn complete_hmm_output_validation_rejects_invalid_probabilities() {
        assert!(transition_probabilities_are_valid(&[0.0, 0.25, 0.75]));
        assert!(!transition_probabilities_are_valid(&[f64::NAN, 0.25, 0.75]));
        assert!(!transition_probabilities_are_valid(&[
            0.25,
            f64::INFINITY,
            0.75
        ]));
        assert!(!transition_probabilities_are_valid(&[
            0.25,
            -0.0 - 1e-12,
            0.75
        ]));
        assert!(!transition_probabilities_are_valid(&[0.25, 1.000_001]));
        assert!(missing_probabilities_are_valid(&[0.0, 0.25, 1.0]));
        assert!(!missing_probabilities_are_valid(&[0.0, f32::NAN, 1.0]));
        assert!(!missing_probabilities_are_valid(&[0.0, -0.01, 1.0]));
        assert!(!missing_probabilities_are_valid(&[0.0, 0.25, 1.01]));
    }

    #[test]
    fn double_precision_initial_diplotypes_preserve_tiny_mass() {
        let variants = [0u8];
        let lengths = [1u16];
        let diplotypes = [(1u64 << 0) | (1u64 << 9)];
        let haplotypes = [0b0101_0101u8; 8];
        let centimorgans = [0.0f32];
        let rare_alleles = [-1i8];
        let mut transitions = [0.0f64; 2];
        let layout = scratch_layout(8, 1, 0).unwrap();
        let mut scratch = vec![0.0; layout.total];
        let mut alpha_locus = [0i32; 1];
        let mut parameters = HmmSegmentDoubleV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmSegmentDoubleV1>() as u32,
            variants: variants.as_ptr(),
            variants_length: variants.len(),
            ambiguous: ptr::null(),
            ambiguous_length: 0,
            segment_lengths: lengths.as_ptr(),
            segment_lengths_length: lengths.len(),
            diplotypes: diplotypes.as_ptr(),
            diplotypes_length: diplotypes.len(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: ptr::null(),
            recombination_length: 0,
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            segment_first: 0,
            segment_last: 0,
            locus_first: 0,
            locus_last: 0,
            ambiguous_first: 0,
            ambiguous_last: -1,
            missing_first: 0,
            missing_last: -1,
            transition_first: 2,
            transition_last: 1,
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
            scratch: scratch.as_mut_ptr(),
            scratch_length: scratch.len(),
            alpha_locus_scratch: alpha_locus.as_mut_ptr(),
            alpha_locus_scratch_length: alpha_locus.len(),
        };
        let mut outcome = i32::MIN;
        let status = unsafe { shapeit_hmm_run_segment_double_v1(&parameters, &mut outcome) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(outcome, 0);
        assert_eq!(transitions, [0.5, 0.5]);

        // Twenty ordinary 1e-4 mismatches drive the permitted haplotype lane
        // below f32 range while remaining representable in f64. This is the
        // numerical condition that must be recoverable by a double HMM pass.
        let zero_support_variants = [0x22u8; 10];
        let zero_support_ambiguous = [0b1111_0000u8; 20];
        let zero_support_diplotypes = [1u64 << 36];
        let zero_support_lengths = [20u16];
        let zero_support_haplotypes = [0u8; 20];
        let zero_support_centimorgans = [0.0f32; 20];
        let zero_support_recombination = [0.0f32; 19];
        let zero_support_rare_alleles = [-1i8; 20];
        let mut zero_support_transitions = [7.0f64];
        parameters.variants = zero_support_variants.as_ptr();
        parameters.variants_length = zero_support_variants.len();
        parameters.ambiguous = zero_support_ambiguous.as_ptr();
        parameters.ambiguous_length = zero_support_ambiguous.len();
        parameters.segment_lengths = zero_support_lengths.as_ptr();
        parameters.diplotypes = zero_support_diplotypes.as_ptr();
        parameters.diplotypes_length = zero_support_diplotypes.len();
        parameters.haplotypes = zero_support_haplotypes.as_ptr();
        parameters.haplotypes_length = zero_support_haplotypes.len();
        parameters.centimorgans = zero_support_centimorgans.as_ptr();
        parameters.centimorgans_length = zero_support_centimorgans.len();
        parameters.recombination = zero_support_recombination.as_ptr();
        parameters.recombination_length = zero_support_recombination.len();
        parameters.rare_alleles = zero_support_rare_alleles.as_ptr();
        parameters.rare_alleles_length = zero_support_rare_alleles.len();
        parameters.emission_match = 0.9999;
        parameters.emission_mismatch = 0.0001;
        parameters.locus_last = 19;
        parameters.ambiguous_first = 0;
        parameters.ambiguous_last = 19;
        parameters.transition_first = 1;
        parameters.transition_last = 0;
        parameters.transition_probabilities = zero_support_transitions.as_mut_ptr();
        parameters.transition_probabilities_length = zero_support_transitions.len();
        let mut zero_support_outcome = i32::MIN;
        let zero_support_status =
            unsafe { shapeit_hmm_run_segment_double_v1(&parameters, &mut zero_support_outcome) };
        assert_eq!(zero_support_status, STATUS_OK);
        assert_eq!(zero_support_outcome, 0);
        assert_eq!(zero_support_transitions, [1.0]);
    }

    #[test]
    fn log_replay_recovers_permitted_mass_beyond_f64_range() {
        // The only permitted diplotype uses two lanes that each incur 100
        // 1e-4 mismatches. Its mass is mathematically nonzero but far below
        // the f64 range after the diplotype contraction.
        let variants = [0x22u8; 50];
        let ambiguous = [0b1111_0000u8; 100];
        let segment_lengths = [100u16];
        let diplotypes = [1u64 << 36];
        let haplotypes = [0u8; 100];
        let centimorgans = [0.0f32; 100];
        let recombination = [0.0f32; 99];
        let rare_alleles = [-1i8; 100];
        let mut transitions = [7.0f64];
        let job = HmmJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmJobV1>() as u32,
            graph: ptr::null_mut(),
            conditioning_job: ptr::null_mut(),
            haplotypes: ptr::null(),
            haplotypes_length: 0,
            haplotype_stride: 1,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: recombination.as_ptr(),
            recombination_length: recombination.len(),
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: 0.9999,
            emission_mismatch: 0.0001,
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
        };
        let inputs = JobWindowInputs {
            job: &job,
            variants: &variants,
            ambiguous: &ambiguous,
            segment_lengths: &segment_lengths,
            diplotypes: &diplotypes,
            subset_haplotypes: &haplotypes,
            subset_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            window: GenotypeWindowV1 {
                start_locus: 0,
                start_segment: 0,
                start_ambiguous: 0,
                start_missing: 0,
                start_transition: 1,
                stop_locus: 99,
                stop_segment: 0,
                stop_ambiguous: 99,
                stop_missing: -1,
                stop_transition: 0,
            },
        };
        let mut scratch = Vec::new();
        let mut alpha_locus = Vec::new();
        let mut double_validation = None;

        let double_outcome = unsafe {
            run_job_window_double(
                &inputs,
                &mut scratch,
                &mut alpha_locus,
                &mut double_validation,
            )
        }
        .unwrap();
        assert_eq!(double_outcome, -2);

        let recovered = unsafe {
            run_job_window_extended_precision(
                &inputs,
                &mut scratch,
                &mut alpha_locus,
                &mut double_validation,
            )
        }
        .unwrap();
        assert_eq!(
            recovered,
            ExtendedPrecisionOutcome {
                outcome: 0,
                used_log_domain: true,
            }
        );
        assert_eq!(transitions, [1.0]);
    }

    #[test]
    fn log_replay_matches_double_on_ordinary_window() {
        let variants = [0x12u8, 0x20, 0x01];
        let ambiguous = [0b1010_1010u8, 0b1100_1100];
        let segment_lengths = [3u16, 3];
        let diplotypes = [(1u64 << 0) | (1u64 << 9), (1u64 << 9) | (1u64 << 18)];
        let haplotypes = [
            0b0101_1010u8,
            0b0011_1100,
            0b1111_0000,
            0b1001_0110,
            0b0110_1001,
            0b1100_0011,
        ];
        let centimorgans = [0.0f32, 0.01, 0.02, 0.04, 0.07, 0.11];
        let recombination = [0.01f32; 5];
        let rare_alleles = [-1i8; 6];
        let mut double_transitions = [0.0f64; 6];
        let mut double_missing = [0.0f32; 16];
        let mut log_transitions = [0.0f64; 6];
        let mut log_missing = [0.0f32; 16];
        let mut job = HmmJobV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmJobV1>() as u32,
            graph: ptr::null_mut(),
            conditioning_job: ptr::null_mut(),
            haplotypes: ptr::null(),
            haplotypes_length: 0,
            haplotype_stride: 1,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: recombination.as_ptr(),
            recombination_length: recombination.len(),
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            transition_probabilities: double_transitions.as_mut_ptr(),
            transition_probabilities_length: double_transitions.len(),
            missing_probabilities: double_missing.as_mut_ptr(),
            missing_probabilities_length: double_missing.len(),
        };
        let window = GenotypeWindowV1 {
            start_locus: 0,
            start_segment: 0,
            start_ambiguous: 0,
            start_missing: 0,
            start_transition: 2,
            stop_locus: 5,
            stop_segment: 1,
            stop_ambiguous: 1,
            stop_missing: 1,
            stop_transition: 5,
        };
        let mut scratch = Vec::new();
        let mut alpha_locus = Vec::new();
        let mut double_validation = None;
        let double_inputs = JobWindowInputs {
            job: &job,
            variants: &variants,
            ambiguous: &ambiguous,
            segment_lengths: &segment_lengths,
            diplotypes: &diplotypes,
            subset_haplotypes: &haplotypes,
            subset_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            window,
        };
        assert_eq!(
            unsafe {
                run_job_window_double(
                    &double_inputs,
                    &mut scratch,
                    &mut alpha_locus,
                    &mut double_validation,
                )
            },
            Ok(0)
        );

        job.transition_probabilities = log_transitions.as_mut_ptr();
        job.missing_probabilities = log_missing.as_mut_ptr();
        let log_inputs = JobWindowInputs {
            job: &job,
            variants: &variants,
            ambiguous: &ambiguous,
            segment_lengths: &segment_lengths,
            diplotypes: &diplotypes,
            subset_haplotypes: &haplotypes,
            subset_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            window,
        };
        assert_eq!(
            unsafe { log::run_job_window_log(&log_inputs, &mut scratch, &mut alpha_locus) },
            Ok(0)
        );

        for (&double, &logged) in double_transitions.iter().zip(log_transitions.iter()) {
            assert!((double - logged).abs() < 1e-12, "{double} != {logged}");
        }
        for (&double, &logged) in double_missing.iter().zip(log_missing.iter()) {
            assert!((double - logged).abs() < 1e-6, "{double} != {logged}");
        }
        assert!(unsafe { window_hmm_output_is_valid(&log_inputs) });
        log_missing[3] = f32::NAN;
        assert!(!unsafe { window_hmm_output_is_valid(&log_inputs) });
        log_missing[3] = double_missing[3];
        log_transitions[4] = -1.0;
        assert!(!unsafe { window_hmm_output_is_valid(&log_inputs) });
    }

    #[test]
    fn invalid_layout_does_not_write_outputs() {
        let variants = [0u8];
        let lengths = [1u16];
        let diplotypes = [1u64];
        let haplotypes = [0u8; 8];
        let centimorgans = [0.0f32];
        let rare_alleles = [-1i8];
        let mut transitions = [0xa5a5_a5a5_a5a5_a5a5u64 as f64];
        let before = transitions;
        let mut scratch = [0.0f64; 82];
        let mut alpha_locus = [0i32; 1];
        let parameters = HmmSegmentDoubleV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmSegmentDoubleV1>() as u32,
            variants: variants.as_ptr(),
            variants_length: variants.len(),
            ambiguous: ptr::null(),
            ambiguous_length: 0,
            segment_lengths: lengths.as_ptr(),
            segment_lengths_length: lengths.len(),
            diplotypes: diplotypes.as_ptr(),
            diplotypes_length: diplotypes.len(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: ptr::null(),
            recombination_length: 0,
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            segment_first: 0,
            segment_last: 0,
            locus_first: 0,
            locus_last: 1,
            ambiguous_first: 0,
            ambiguous_last: -1,
            missing_first: 0,
            missing_last: -1,
            transition_first: 1,
            transition_last: 0,
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
            scratch: scratch.as_mut_ptr(),
            scratch_length: scratch.len(),
            alpha_locus_scratch: alpha_locus.as_mut_ptr(),
            alpha_locus_scratch_length: alpha_locus.len(),
        };
        let mut outcome = 77;
        let status = unsafe { shapeit_hmm_run_segment_double_v1(&parameters, &mut outcome) };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(outcome, 77);
        assert_eq!(transitions, before);
    }

    #[test]
    fn public_double_abi_revalidates_unselected_graph_tail() {
        let mut variants = [0u8];
        let lengths = [1u16, 1];
        let diplotypes = [1u64, 1];
        let haplotypes = [0u8];
        let centimorgans = [0.0f32, 0.01];
        let recombination = [0.01f32];
        let rare_alleles = [-1i8; 2];
        let mut transitions = [0.0f64; 2];
        let layout = scratch_layout(8, 1, 0).unwrap();
        let mut scratch = vec![0.0f64; layout.total];
        let mut alpha_locus = [0i32; 1];
        let parameters = HmmSegmentDoubleV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmSegmentDoubleV1>() as u32,
            variants: variants.as_ptr(),
            variants_length: variants.len(),
            ambiguous: ptr::null(),
            ambiguous_length: 0,
            segment_lengths: lengths.as_ptr(),
            segment_lengths_length: lengths.len(),
            diplotypes: diplotypes.as_ptr(),
            diplotypes_length: diplotypes.len(),
            haplotypes: haplotypes.as_ptr(),
            haplotypes_length: haplotypes.len(),
            haplotype_stride: 1,
            conditioning_haplotypes: 8,
            locus_offset: 0,
            centimorgans: centimorgans.as_ptr(),
            centimorgans_length: centimorgans.len(),
            recombination: recombination.as_ptr(),
            recombination_length: recombination.len(),
            rare_alleles: rare_alleles.as_ptr(),
            rare_alleles_length: rare_alleles.len(),
            effective_population_size: 15_000,
            total_haplotypes: 16,
            emission_match: f64::from(0.9999f32),
            emission_mismatch: f64::from(0.0001f32),
            segment_first: 0,
            segment_last: 0,
            locus_first: 0,
            locus_last: 0,
            ambiguous_first: 0,
            ambiguous_last: -1,
            missing_first: 0,
            missing_last: -1,
            transition_first: 1,
            transition_last: 0,
            transition_probabilities: transitions.as_mut_ptr(),
            transition_probabilities_length: transitions.len(),
            missing_probabilities: ptr::null_mut(),
            missing_probabilities_length: 0,
            scratch: scratch.as_mut_ptr(),
            scratch_length: scratch.len(),
            alpha_locus_scratch: alpha_locus.as_mut_ptr(),
            alpha_locus_scratch_length: alpha_locus.len(),
        };

        reset_double_graph_validation_scans();
        let mut outcome = i32::MIN;
        let status = unsafe { shapeit_hmm_run_segment_double_v1(&parameters, &mut outcome) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(outcome, 0);
        assert_eq!(double_graph_validation_scans(), 1);

        // Turn the unselected second locus into a missing genotype. A cached
        // prefix would overlook the newly required missing-probability block.
        variants[0] = 0x10;
        assert_eq!(variants[0], 0x10);
        transitions.copy_from_slice(&[7.0, 8.0]);
        let before = transitions;
        outcome = 77;
        let status = unsafe { shapeit_hmm_run_segment_double_v1(&parameters, &mut outcome) };
        assert_eq!(status, STATUS_OUT_OF_BOUNDS);
        assert_eq!(outcome, 77);
        assert_eq!(transitions, before);
        assert_eq!(double_graph_validation_scans(), 2);
    }
}

mod log;
mod single;
