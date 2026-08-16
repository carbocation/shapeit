#![allow(clippy::needless_range_loop)]

use core::{mem, slice};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

use self::single::{shapeit_hmm_run_segment_single_prevalidated_v1, HmmSegmentSingleV1};
use crate::bitmatrix::shapeit_bitmatrix_subset_transpose_v1;
use crate::conditioning::{
    shapeit_conditioning_graph_job_build_v1, ConditioningGraphBuildV1, ConditioningJobV1,
};
use crate::genotype::{
    shapeit_genotype_graph_prune_v1, shapeit_genotype_graph_sample_current_v1,
    shapeit_genotype_graph_store_v1, GenotypeGraphV1, GenotypeWindowV1,
};
use crate::ibd2::Ibd2TracksV1;

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

fn validate(
    parameters: &HmmSegmentDoubleV1,
    variants: &[u8],
    segment_lengths: &[u16],
    diplotypes: &[u64],
) -> Result<ValidatedLayout, u32> {
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size as usize != mem::size_of::<HmmSegmentDoubleV1>()
        || parameters.effective_population_size <= 0
        || parameters.total_haplotypes <= 0
        || !parameters.emission_match.is_finite()
        || !parameters.emission_mismatch.is_finite()
        || parameters.emission_match == 0.0
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let segment_first = usize_coordinate(parameters.segment_first)?;
    let segment_last = usize_coordinate(parameters.segment_last)?;
    let locus_first = usize_coordinate(parameters.locus_first)?;
    let locus_last = usize_coordinate(parameters.locus_last)?;
    if segment_first > segment_last
        || segment_last >= segment_lengths.len()
        || segment_lengths.len() > diplotypes.len()
        || locus_first > locus_last
    {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    if segment_lengths.contains(&0) || diplotypes[..segment_lengths.len()].contains(&0) {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let segment_count = segment_last - segment_first + 1;
    let mut total_variants = 0usize;
    let mut total_ambiguous = 0usize;
    let mut total_missing = 0usize;
    let mut transition_total = 0usize;
    let mut previous_diplotypes = 1usize;
    let mut expected_locus_first = 0usize;
    let mut expected_locus_last = 0usize;
    let mut expected_ambiguous_first = 0usize;
    let mut expected_ambiguous_stop = -1i32;
    let mut expected_missing_first = 0usize;
    let mut expected_missing_stop = -1i32;
    let mut expected_transition_first = 0usize;
    let mut expected_transition_last = 0usize;

    let required_variant_bytes = segment_lengths
        .iter()
        .try_fold(0usize, |sum, &length| sum.checked_add(length as usize))
        .ok_or(STATUS_INTEGER_OVERFLOW)?
        .checked_add(1)
        .ok_or(STATUS_INTEGER_OVERFLOW)?
        >> 1;
    if variants.len() < required_variant_bytes {
        return Err(STATUS_OUT_OF_BOUNDS);
    }

    for (segment, (&length, &diplotype_mask)) in
        segment_lengths.iter().zip(diplotypes.iter()).enumerate()
    {
        if segment == segment_first {
            expected_locus_first = total_variants;
            expected_ambiguous_first = total_ambiguous;
            expected_missing_first = total_missing;
        }
        let segment_stop = total_variants
            .checked_add(length as usize)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        for locus in total_variants..segment_stop {
            match variant_code(variants, locus) {
                1 => {
                    total_missing = total_missing
                        .checked_add(1)
                        .ok_or(STATUS_INTEGER_OVERFLOW)?
                }
                2 | 3 => {
                    total_ambiguous = total_ambiguous
                        .checked_add(1)
                        .ok_or(STATUS_INTEGER_OVERFLOW)?
                }
                _ => {}
            }
        }
        total_variants = segment_stop;

        let current_diplotypes = diplotype_count(diplotype_mask);
        let transition_count = previous_diplotypes
            .checked_mul(current_diplotypes)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        transition_total = transition_total
            .checked_add(transition_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if segment == segment_first {
            expected_transition_first = transition_total;
        }
        if segment == segment_last {
            expected_locus_last = total_variants - 1;
            expected_ambiguous_stop = signed_stop(total_ambiguous)?;
            expected_missing_stop = signed_stop(total_missing)?;
            expected_transition_last = transition_total - 1;
        }
        previous_diplotypes = current_diplotypes;
    }

    if locus_first != expected_locus_first
        || locus_last != expected_locus_last
        || parameters.ambiguous_first
            != i32::try_from(expected_ambiguous_first).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.ambiguous_last != expected_ambiguous_stop
        || parameters.missing_first
            != i32::try_from(expected_missing_first).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.missing_last != expected_missing_stop
        || parameters.transition_first
            != i32::try_from(expected_transition_first).map_err(|_| STATUS_INTEGER_OVERFLOW)?
        || parameters.transition_last
            != i32::try_from(expected_transition_last).map_err(|_| STATUS_INTEGER_OVERFLOW)?
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }

    let window_missing_end =
        usize::try_from(expected_missing_stop + 1).map_err(|_| STATUS_INTEGER_OVERFLOW)?;
    let window_missing_count = window_missing_end - expected_missing_first;
    let scratch = scratch_layout(
        parameters.conditioning_haplotypes,
        segment_count,
        window_missing_count,
    )?;
    if parameters.scratch_length < scratch.total
        || parameters.alpha_locus_scratch_length < segment_count
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
    if required_haplotype_length > parameters.haplotypes_length
        || total_ambiguous > parameters.ambiguous_length
        || total_variants > parameters.centimorgans_length
        || total_variants > parameters.rare_alleles_length
        || total_variants.saturating_sub(1) > parameters.recombination_length
        || transition_total > parameters.transition_probabilities_length
        || total_missing
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
        ambiguous_first: expected_ambiguous_first,
        missing_first: expected_missing_first,
        transition_last: expected_transition_last,
    })
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

    fn set_first_transitions(&mut self) {
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
        let scale_diplotype = 1.0 / total;
        for (target, &value) in self.transition_probabilities[..count]
            .iter_mut()
            .zip(probabilities.iter())
        {
            *target = value * scale_diplotype;
        }
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
                self.set_first_transitions();
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
    let mut outcome = 0;
    let status = shapeit_hmm_run_segment_double_v1(&parameters, &mut outcome);
    if status == STATUS_OK {
        Ok(outcome)
    } else {
        Err(status)
    }
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

#[no_mangle]
/// Run every HMM window in one Rust-owned conditioning job.
///
/// The function retains subset-transpose and HMM scratch capacity in the
/// worker-local conditioning job. It also owns the established single-to-double
/// fallback decision and persists that decision in the genotype graph.
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
    let parameters = &*parameters;
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
        let outcome = if require_double_precision {
            match run_job_window_double(&window_inputs, double_scratch, alpha_locus_scratch) {
                Ok(value) => value,
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
                let double_outcome = match run_job_window_double(
                    &window_inputs,
                    double_scratch,
                    alpha_locus_scratch,
                ) {
                    Ok(value) => value,
                    Err(status) => return status,
                };
                require_double_precision = true;
                local_result.underflow_recovered_precision += 1;
                double_outcome
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
    let graph = &*parameters.graph;
    if !graph.is_built() {
        return STATUS_INVALID_DIMENSIONS;
    }
    let (_, transition_count, missing_count) = graph.hmm_dimensions();
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
    let hmm_status = shapeit_hmm_run_job_v1(&hmm_parameters, &mut local_result);
    let operation_status = if hmm_status != STATUS_OK || local_result.fatal_outcome != 0 {
        hmm_status
    } else {
        let sample_status = shapeit_genotype_graph_sample_current_v1(
            parameters.graph,
            transition_probabilities.as_ptr(),
            transition_probabilities.len(),
            missing_probabilities.as_ptr(),
            missing_probabilities.len(),
            parameters.sample_seed,
            parameters.sample_domain,
            parameters.sample_iteration,
            parameters.sample_item,
        );
        if sample_status != STATUS_OK {
            sample_status
        } else {
            match parameters.stage {
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
            }
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
    let parameters = &*parameters;
    if parameters.abi_version != ABI_VERSION
        || parameters.struct_size < mem::size_of::<CommonPhaseJobV1>()
        || parameters.conditioning.graph.cast_mut() != parameters.phase.graph
        || parameters.conditioning.haplotypes != parameters.phase.haplotypes
        || parameters.conditioning.haplotypes_length != parameters.phase.haplotypes_length
        || parameters.conditioning.haplotype_stride != parameters.phase.haplotype_stride
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let conditioning_status =
        shapeit_conditioning_graph_job_build_v1(&parameters.conditioning, job);
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
            shapeit_common_phase_job_run_v1(
                &parameters,
                &mut worker.conditioning_job,
                &mut job_result,
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
    let shared = CommonIterationShared {
        template: SharedCommonTemplate(parameters.sample_template),
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
    let ambiguous = const_slice(parameters.ambiguous, parameters.ambiguous_length);
    let segment_lengths = const_slice(
        parameters.segment_lengths,
        parameters.segment_lengths_length,
    );
    let diplotypes = const_slice(parameters.diplotypes, parameters.diplotypes_length);
    let layout = match validate(parameters, variants, segment_lengths, diplotypes) {
        Ok(value) => value,
        Err(status) => return status,
    };

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
    *outcome = engine.run();
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

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
            start_transition: 1,
            stop_locus: 0,
            stop_segment: 0,
            stop_ambiguous: -1,
            stop_missing: -1,
            stop_transition: 0,
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
        let status = unsafe { shapeit_hmm_run_job_v1(&parameters, &mut result) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(result.fatal_outcome, 0);
        assert_eq!(result.windows_completed, 1);
        assert_eq!(result.underflow_recovered_precision, 0);
        assert!((transitions.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(!conditioning_job.subset_haplotypes.is_empty());
        assert!(!conditioning_job.single_scratch.is_empty());

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

        unsafe { crate::genotype::shapeit_genotype_graph_free_v1(graph) };
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
    fn single_locus_segment_normalizes_first_diplotypes() {
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
}

mod single;
