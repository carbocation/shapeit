#![allow(clippy::needless_range_loop)]

use core::{mem, slice};

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
const HAPLOTYPES: usize = 8;

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
