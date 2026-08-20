use super::*;

#[cfg(target_arch = "x86_64")]
use core::arch::asm;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __cpuid, __m128, __m256, _mm256_add_ps, _mm256_and_si256, _mm256_blendv_ps,
    _mm256_castps128_ps256, _mm256_castsi256_ps, _mm256_cvtps_pd, _mm256_fmadd_ps,
    _mm256_insertf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setr_epi32, _mm256_setr_ps, _mm256_setzero_ps, _mm256_slli_epi32, _mm256_srlv_epi32,
    _mm256_storeu_ps, _mm256_xor_si256, _mm_loadu_ps, _mm_set1_ps, _mm_setr_ps, _mm_storeu_ps,
};

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "x86_64")]
#[inline]
fn intel_family_model(eax: u32) -> (u32, u32) {
    let base_family = (eax >> 8) & 0x0f;
    let family = if base_family == 0x0f {
        base_family + ((eax >> 20) & 0xff)
    } else {
        base_family
    };
    let base_model = (eax >> 4) & 0x0f;
    let model = if matches!(base_family, 0x06 | 0x0f) {
        base_model | (((eax >> 16) & 0x0f) << 4)
    } else {
        base_model
    };
    (family, model)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn intel_avx512_full_model_allowed(eax: u32) -> bool {
    let (family, model) = intel_family_model(eax);
    family == 6 && matches!(model, 0x8f | 0xcf)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx512_full_kernel_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(detect_avx512_full_kernel)
}

#[cfg(target_arch = "x86_64")]
fn detect_avx512_full_kernel() -> bool {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return false;
    }
    // The paired-ZMM kernel loses on Cascade Lake but wins on measured
    // Sapphire Rapids (model 0x8f) and Emerald Rapids (model 0xcf).
    // SAFETY: CPUID is available in every x86-64 execution environment.
    let vendor = unsafe { __cpuid(0) };
    if [vendor.ebx, vendor.edx, vendor.ecx]
        != [
            u32::from_le_bytes(*b"Genu"),
            u32::from_le_bytes(*b"ineI"),
            u32::from_le_bytes(*b"ntel"),
        ]
    {
        return false;
    }
    // SAFETY: Basic CPUID leaf 1 is available on every x86-64 processor.
    let version = unsafe { __cpuid(1) };
    intel_avx512_full_model_allowed(version.eax)
}

#[repr(C)]
pub struct HmmSegmentSingleV1 {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,

    pub(super) variants: *const u8,
    pub(super) variants_length: usize,
    pub(super) ambiguous: *const u8,
    pub(super) ambiguous_length: usize,
    pub(super) segment_lengths: *const u16,
    pub(super) segment_lengths_length: usize,
    pub(super) diplotypes: *const u64,
    pub(super) diplotypes_length: usize,

    pub(super) haplotypes: *const u8,
    pub(super) haplotypes_length: usize,
    pub(super) haplotype_stride: usize,
    pub(super) conditioning_haplotypes: usize,
    pub(super) locus_offset: u32,

    pub(super) centimorgans: *const f32,
    pub(super) centimorgans_length: usize,
    pub(super) recombination: *const f32,
    pub(super) recombination_length: usize,
    pub(super) rare_alleles: *const i8,
    pub(super) rare_alleles_length: usize,
    pub(super) effective_population_size: i32,
    pub(super) total_haplotypes: i32,
    pub(super) emission_match: f32,
    pub(super) emission_mismatch: f32,

    pub(super) segment_first: i32,
    pub(super) segment_last: i32,
    pub(super) locus_first: i32,
    pub(super) locus_last: i32,
    pub(super) ambiguous_first: i32,
    pub(super) ambiguous_last: i32,
    pub(super) missing_first: i32,
    pub(super) missing_last: i32,
    pub(super) transition_first: i32,
    pub(super) transition_last: i32,

    pub(super) transition_probabilities: *mut f64,
    pub(super) transition_probabilities_length: usize,
    pub(super) missing_probabilities: *mut f32,
    pub(super) missing_probabilities_length: usize,

    pub(super) scratch: *mut f32,
    pub(super) scratch_length: usize,
    pub(super) alpha_locus_scratch: *mut i32,
    pub(super) alpha_locus_scratch_length: usize,
    pub(super) index_scratch: *mut usize,
    pub(super) index_scratch_length: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SingleScratchLayout {
    states: usize,
    segment_count: usize,
    missing_count: usize,
    alpha_values: usize,
    float_total: usize,
    index_total: usize,
}
fn single_validation_shadow(parameters: &HmmSegmentSingleV1) -> HmmSegmentDoubleV1 {
    HmmSegmentDoubleV1 {
        abi_version: parameters.abi_version,
        struct_size: mem::size_of::<HmmSegmentDoubleV1>() as u32,
        variants: parameters.variants,
        variants_length: parameters.variants_length,
        ambiguous: parameters.ambiguous,
        ambiguous_length: parameters.ambiguous_length,
        segment_lengths: parameters.segment_lengths,
        segment_lengths_length: parameters.segment_lengths_length,
        diplotypes: parameters.diplotypes,
        diplotypes_length: parameters.diplotypes_length,
        haplotypes: parameters.haplotypes,
        haplotypes_length: parameters.haplotypes_length,
        haplotype_stride: parameters.haplotype_stride,
        conditioning_haplotypes: parameters.conditioning_haplotypes,
        locus_offset: parameters.locus_offset,
        centimorgans: parameters.centimorgans,
        centimorgans_length: parameters.centimorgans_length,
        recombination: parameters.recombination,
        recombination_length: parameters.recombination_length,
        rare_alleles: parameters.rare_alleles,
        rare_alleles_length: parameters.rare_alleles_length,
        effective_population_size: parameters.effective_population_size,
        total_haplotypes: parameters.total_haplotypes,
        emission_match: f64::from(parameters.emission_match),
        emission_mismatch: f64::from(parameters.emission_mismatch),
        segment_first: parameters.segment_first,
        segment_last: parameters.segment_last,
        locus_first: parameters.locus_first,
        locus_last: parameters.locus_last,
        ambiguous_first: parameters.ambiguous_first,
        ambiguous_last: parameters.ambiguous_last,
        missing_first: parameters.missing_first,
        missing_last: parameters.missing_last,
        transition_first: parameters.transition_first,
        transition_last: parameters.transition_last,
        transition_probabilities: parameters.transition_probabilities,
        transition_probabilities_length: parameters.transition_probabilities_length,
        missing_probabilities: parameters.missing_probabilities,
        missing_probabilities_length: parameters.missing_probabilities_length,
        scratch: core::ptr::null_mut(),
        scratch_length: usize::MAX,
        alpha_locus_scratch: core::ptr::null_mut(),
        alpha_locus_scratch_length: usize::MAX,
    }
}

fn single_prevalidated_layout(parameters: &HmmSegmentSingleV1) -> Result<ValidatedLayout, u32> {
    let segment_first = usize_coordinate(parameters.segment_first)?;
    let segment_last = usize_coordinate(parameters.segment_last)?;
    let locus_first = usize_coordinate(parameters.locus_first)?;
    let locus_last = usize_coordinate(parameters.locus_last)?;
    let ambiguous_first = usize_coordinate(parameters.ambiguous_first)?;
    let missing_first = usize_coordinate(parameters.missing_first)?;
    let transition_last = usize_coordinate(parameters.transition_last)?;
    if segment_last < segment_first || locus_last < locus_first {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let segment_count = segment_last - segment_first + 1;
    let missing_count = if parameters.missing_last < parameters.missing_first {
        0
    } else {
        usize::try_from(
            i64::from(parameters.missing_last) - i64::from(parameters.missing_first) + 1,
        )
        .map_err(|_| STATUS_INTEGER_OVERFLOW)?
    };
    let scratch = scratch_layout(
        parameters.conditioning_haplotypes,
        segment_count,
        missing_count,
    )?;
    Ok(ValidatedLayout {
        scratch,
        segment_first,
        segment_last,
        locus_first,
        locus_last,
        ambiguous_first,
        missing_first,
        transition_last,
    })
}

fn single_scratch_layout(
    parameters: &HmmSegmentSingleV1,
    variants: &[u8],
    ambiguous: &[u8],
    segment_lengths: &[u16],
    validated: ValidatedLayout,
    mut indexes: Option<&mut [usize]>,
) -> Result<SingleScratchLayout, u32> {
    let segment_count = validated.segment_last - validated.segment_first + 1;
    let states = parameters
        .conditioning_haplotypes
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let index_total = segment_count
        .checked_mul(4)
        .and_then(|value| value.checked_add(1))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if let Some(indexes) = indexes.as_ref() {
        if indexes.len() < index_total {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
    }

    let mut locus = validated.locus_first;
    let mut ambiguous_index = validated.ambiguous_first;
    let mut alpha_values = 0usize;
    for relative_segment in 0..segment_count {
        let segment = validated.segment_first + relative_segment;
        let segment_length = segment_lengths[segment] as usize;
        let mut haplotypes = 1usize;
        let mut first_ambiguous = segment_length;
        let mut last_ambiguous = 0usize;
        for relative_locus in 0..segment_length {
            if variant_code(variants, locus) > 1 {
                first_ambiguous = first_ambiguous.min(relative_locus);
                last_ambiguous = relative_locus;
                let code = ambiguous[ambiguous_index];
                ambiguous_index += 1;
                while haplotypes < HAPLOTYPES {
                    let periodic = (haplotypes..HAPLOTYPES).all(|haplotype| {
                        ((code >> haplotype) & 1) == ((code >> (haplotype % haplotypes)) & 1)
                    });
                    if periodic {
                        break;
                    }
                    haplotypes <<= 1;
                }
            }
            locus += 1;
        }
        let alpha_start = alpha_values;
        alpha_values = alpha_values
            .checked_add(
                parameters
                    .conditioning_haplotypes
                    .checked_mul(haplotypes)
                    .ok_or(STATUS_INTEGER_OVERFLOW)?,
            )
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if let Some(indexes) = &mut indexes {
            indexes[relative_segment] = haplotypes;
            indexes[segment_count + relative_segment] = first_ambiguous;
            indexes[2 * segment_count + relative_segment] = last_ambiguous;
            indexes[3 * segment_count + relative_segment] = alpha_start;
        }
    }
    if let Some(indexes) = indexes {
        indexes[4 * segment_count] = alpha_values;
    }

    let alpha_sum = segment_count
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha_missing = validated
        .scratch
        .missing_count
        .checked_mul(states)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let alpha_sum_missing = validated
        .scratch
        .missing_count
        .checked_mul(HAPLOTYPES)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let float_total = states
        .checked_add(parameters.conditioning_haplotypes)
        .and_then(|value| value.checked_add(alpha_values))
        .and_then(|value| value.checked_add(alpha_sum))
        .and_then(|value| value.checked_add(segment_count))
        .and_then(|value| value.checked_add(alpha_missing))
        .and_then(|value| value.checked_add(alpha_sum_missing))
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    Ok(SingleScratchLayout {
        states,
        segment_count,
        missing_count: validated.scratch.missing_count,
        alpha_values,
        float_total,
        index_total,
    })
}
struct SingleEngine<'a> {
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
    mismatch: f32,
    #[cfg(target_arch = "x86_64")]
    avx512: bool,

    segment_first: usize,
    segment_last: usize,
    locus_first: usize,
    locus_last: usize,
    ambiguous_first: usize,
    ambiguous_last: isize,
    missing_first: usize,
    missing_last: isize,
    transition_last: usize,

    segment_hap_count: &'a [usize],
    segment_first_ambiguous: &'a [usize],
    segment_last_ambiguous: &'a [usize],
    alpha_offsets: &'a [usize],

    prob: &'a mut [f32],
    prob_sum_k: &'a mut [f32],
    alpha: &'a mut [f32],
    alpha_sum: &'a mut [f32],
    alpha_sum_sum: &'a mut [f32],
    alpha_missing: &'a mut [f32],
    alpha_sum_missing: &'a mut [f32],
    alpha_locus: &'a mut [i32],
    transition_probabilities: &'a mut [f64],
    missing_probabilities: &'a mut [f32],

    prob_haps: usize,
    prob_sum_t: f32,
    prob_sum_h: [f32; HAPLOTYPES],
    sum_h_probs: f32,
    sum_h_probs_double: f64,
    sum_d_probs: f64,
    h_probs: [f32; HAPLOTYPES * HAPLOTYPES],
    h_probs_double: [f64; HAPLOTYPES * HAPLOTYPES],
    d_probs: [f64; HAPLOTYPES * HAPLOTYPES * HAPLOTYPES * HAPLOTYPES],
}

impl SingleEngine<'_> {
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

    #[inline(always)]
    fn transition_probability(&self, previous: usize, current: usize) -> f32 {
        debug_assert_ne!(previous, current);
        if previous.abs_diff(current) == 1 {
            return self.recombination[previous.min(current)];
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
        -(unsafe { expm1f(argument) })
    }

    #[inline]
    fn update_total(&mut self, unique_sums: &[f32], haplotypes: usize) {
        for h in 0..HAPLOTYPES {
            self.prob_sum_h[h] = unique_sums[h % haplotypes];
        }
        self.prob_sum_t = self.prob_sum_h[0]
            + self.prob_sum_h[1]
            + self.prob_sum_h[2]
            + self.prob_sum_h[3]
            + self.prob_sum_h[4]
            + self.prob_sum_h[5]
            + self.prob_sum_h[6]
            + self.prob_sum_h[7];
    }

    fn init_hom(&mut self, locus: usize, relative_locus: usize) {
        let genotype_allele = self.hap0(locus);
        let mut sums = [0.0f32; HAPLOTYPES];
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
        self.update_total(&sums, HAPLOTYPES);
    }

    fn init_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize) {
        let code = self.ambiguous[ambiguous_index];
        let mut sums = [0.0f32; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let graph_haplotype = ((code >> h) & 1) != 0;
                let emission = if graph_haplotype != allele {
                    self.mismatch
                } else {
                    1.0
                };
                self.prob[start + h] = emission;
                sums[h] += emission;
            }
        }
        self.update_total(&sums, HAPLOTYPES);
    }

    fn init_missing(&mut self) {
        let probability = 1.0f32 / (HAPLOTYPES as f32 * self.conditioning_haplotypes as f32);
        self.prob.fill(probability);
        self.prob_sum_h.fill(1.0f32 / HAPLOTYPES as f32);
        self.prob_sum_t = 1.0;
    }

    fn reshape_haplotypes(&mut self, haplotypes: usize) {
        debug_assert!(matches!(haplotypes, 1 | 2 | 4 | HAPLOTYPES));
        if self.prob_haps == haplotypes {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // The enclosing SHAPEIT common-phasing binary requires AVX2/FMA.
            unsafe {
                self.reshape_haplotypes_avx2(haplotypes);
            }
            return;
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            match (self.prob_haps, haplotypes) {
                (8, 1) | (8, 2) | (8, 4) => {
                    for k in 1..self.conditioning_haplotypes {
                        for h in 0..haplotypes {
                            self.prob[k * haplotypes + h] = self.prob[k * HAPLOTYPES + h];
                        }
                    }
                }
                (1, 2) | (1, 4) | (1, 8) => {
                    for k in (0..self.conditioning_haplotypes).rev() {
                        let value = self.prob[k];
                        for h in 0..haplotypes {
                            self.prob[k * haplotypes + h] = value;
                        }
                    }
                }
                (2, 4) | (2, 8) => {
                    for k in (0..self.conditioning_haplotypes).rev() {
                        let values = [self.prob[k * 2], self.prob[k * 2 + 1]];
                        for h in 0..haplotypes {
                            self.prob[k * haplotypes + h] = values[h & 1];
                        }
                    }
                }
                (4, 8) => {
                    for k in (0..self.conditioning_haplotypes).rev() {
                        let values = [
                            self.prob[k * 4],
                            self.prob[k * 4 + 1],
                            self.prob[k * 4 + 2],
                            self.prob[k * 4 + 3],
                        ];
                        for h in 0..HAPLOTYPES {
                            self.prob[k * HAPLOTYPES + h] = values[h & 3];
                        }
                    }
                }
                _ => unreachable!("invalid compressed HMM lane transition"),
            }
            self.prob_haps = haplotypes;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn reshape_haplotypes_avx2(&mut self, haplotypes: usize) {
        let probability = self.prob.as_mut_ptr();
        match (self.prob_haps, haplotypes) {
            (8, 1) => {
                for k in 1..self.conditioning_haplotypes {
                    *probability.add(k) = *probability.add(k * HAPLOTYPES);
                }
            }
            (8, 2) => {
                for k in 1..self.conditioning_haplotypes {
                    *probability.add(k * 2) = *probability.add(k * HAPLOTYPES);
                    *probability.add(k * 2 + 1) = *probability.add(k * HAPLOTYPES + 1);
                }
            }
            (8, 4) => {
                for k in 1..self.conditioning_haplotypes {
                    _mm_storeu_ps(
                        probability.add(k * 4),
                        _mm_loadu_ps(probability.add(k * HAPLOTYPES)),
                    );
                }
            }
            (1, 2) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    let value = *probability.add(k);
                    *probability.add(k * 2) = value;
                    *probability.add(k * 2 + 1) = value;
                }
            }
            (1, 4) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    _mm_storeu_ps(probability.add(k * 4), _mm_set1_ps(*probability.add(k)));
                }
            }
            (1, 8) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    _mm256_storeu_ps(
                        probability.add(k * HAPLOTYPES),
                        _mm256_set1_ps(*probability.add(k)),
                    );
                }
            }
            (2, 4) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    let value0 = *probability.add(k * 2);
                    let value1 = *probability.add(k * 2 + 1);
                    _mm_storeu_ps(
                        probability.add(k * 4),
                        _mm_setr_ps(value0, value1, value0, value1),
                    );
                }
            }
            (2, 8) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    let value0 = *probability.add(k * 2);
                    let value1 = *probability.add(k * 2 + 1);
                    let value = _mm_setr_ps(value0, value1, value0, value1);
                    let mut row = _mm256_castps128_ps256(value);
                    row = _mm256_insertf128_ps(row, value, 1);
                    _mm256_storeu_ps(probability.add(k * HAPLOTYPES), row);
                }
            }
            (4, 8) => {
                for k in (0..self.conditioning_haplotypes).rev() {
                    let value = _mm_loadu_ps(probability.add(k * 4));
                    _mm_storeu_ps(probability.add(k * HAPLOTYPES), value);
                    _mm_storeu_ps(probability.add(k * HAPLOTYPES + 4), value);
                }
            }
            _ => unreachable!("invalid compressed HMM lane transition"),
        }
        self.prob_haps = haplotypes;
    }

    fn run_hom(&mut self, locus: usize, relative_locus: usize, transition: f32) -> bool {
        let genotype_allele = self.hap0(locus);
        let rare_allele = self.rare_alleles[locus];
        if rare_allele >= 0 && genotype_allele != (rare_allele != 0) {
            return false;
        }
        self.run_reduced::<false>(relative_locus, genotype_allele, 0, transition);
        true
    }

    fn run_reduced<const AMBIGUOUS: bool>(
        &mut self,
        relative_locus: usize,
        genotype_allele: bool,
        ambiguous_code: u8,
        transition: f32,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            // The enclosing SHAPEIT common-phasing binary already requires AVX2 and FMA.
            unsafe {
                match self.prob_haps {
                    1 => self.run_compressed_avx2::<1, AMBIGUOUS>(
                        relative_locus,
                        genotype_allele,
                        ambiguous_code,
                        transition,
                    ),
                    2 => self.run_compressed_avx2::<2, AMBIGUOUS>(
                        relative_locus,
                        genotype_allele,
                        ambiguous_code,
                        transition,
                    ),
                    4 => self.run_compressed_avx2::<4, AMBIGUOUS>(
                        relative_locus,
                        genotype_allele,
                        ambiguous_code,
                        transition,
                    ),
                    HAPLOTYPES => self.run_full_avx2::<AMBIGUOUS>(
                        relative_locus,
                        genotype_allele,
                        ambiguous_code,
                        transition,
                    ),
                    _ => unreachable!("invalid compressed HMM lane count"),
                }
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            let factor = transition / (self.conditioning_haplotypes as f32 * self.prob_sum_t);
            let stay_factor = (1.0 - transition) / self.prob_sum_t;
            let row = relative_locus + self.locus_offset;
            let allele_bytes = unsafe { self.haplotypes.as_ptr().add(row * self.haplotype_stride) };
            let sums = unsafe {
                match self.prob_haps {
                    1 => neon::run_compressed::<1, AMBIGUOUS>(
                        self.prob.as_mut_ptr(),
                        self.conditioning_haplotypes,
                        self.prob_sum_h.as_ptr(),
                        allele_bytes,
                        factor,
                        stay_factor,
                        self.mismatch,
                        genotype_allele,
                        ambiguous_code,
                    ),
                    2 => neon::run_compressed::<2, AMBIGUOUS>(
                        self.prob.as_mut_ptr(),
                        self.conditioning_haplotypes,
                        self.prob_sum_h.as_ptr(),
                        allele_bytes,
                        factor,
                        stay_factor,
                        self.mismatch,
                        genotype_allele,
                        ambiguous_code,
                    ),
                    4 => neon::run_four::<AMBIGUOUS>(
                        self.prob.as_mut_ptr(),
                        self.conditioning_haplotypes,
                        self.prob_sum_h.as_ptr(),
                        allele_bytes,
                        factor,
                        stay_factor,
                        self.mismatch,
                        genotype_allele,
                        ambiguous_code,
                    ),
                    HAPLOTYPES => neon::run_full::<AMBIGUOUS>(
                        self.prob.as_mut_ptr(),
                        self.conditioning_haplotypes,
                        self.prob_sum_h.as_ptr(),
                        allele_bytes,
                        factor,
                        stay_factor,
                        self.mismatch,
                        genotype_allele,
                        ambiguous_code,
                    ),
                    _ => unreachable!("invalid compressed HMM lane count"),
                }
            };
            self.update_total(&sums, self.prob_haps);
            return;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let haplotypes = self.prob_haps;
            let factor = transition / (self.conditioning_haplotypes as f32 * self.prob_sum_t);
            let stay_factor = (1.0 - transition) / self.prob_sum_t;
            let mut sums = [0.0f32; HAPLOTYPES];
            for k in 0..self.conditioning_haplotypes {
                let allele = self.allele(relative_locus, k);
                let start = k * haplotypes;
                for h in 0..haplotypes {
                    let graph_haplotype = if AMBIGUOUS {
                        ((ambiguous_code >> h) & 1) != 0
                    } else {
                        genotype_allele
                    };
                    let emission = if graph_haplotype != allele {
                        self.mismatch
                    } else {
                        1.0
                    };
                    let value = self.prob[start + h]
                        .mul_add(stay_factor, self.prob_sum_h[h] * factor)
                        * emission;
                    self.prob[start + h] = value;
                    sums[h] += value;
                }
            }
            self.update_total(&sums, haplotypes);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_compressed_avx2<const N: usize, const AMBIGUOUS: bool>(
        &mut self,
        relative_locus: usize,
        genotype_allele: bool,
        ambiguous_code: u8,
        transition: f32,
    ) {
        debug_assert!(matches!(N, 1 | 2 | 4));
        debug_assert_eq!(self.prob_haps, N);

        let factor = transition / (self.conditioning_haplotypes as f32 * self.prob_sum_t);
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let frequencies = if N == 1 {
            _mm256_set1_ps(self.prob_sum_h[0])
        } else if N == 2 {
            _mm256_setr_ps(
                self.prob_sum_h[0],
                self.prob_sum_h[1],
                self.prob_sum_h[0],
                self.prob_sum_h[1],
                self.prob_sum_h[0],
                self.prob_sum_h[1],
                self.prob_sum_h[0],
                self.prob_sum_h[1],
            )
        } else {
            _mm256_setr_ps(
                self.prob_sum_h[0],
                self.prob_sum_h[1],
                self.prob_sum_h[2],
                self.prob_sum_h[3],
                self.prob_sum_h[0],
                self.prob_sum_h[1],
                self.prob_sum_h[2],
                self.prob_sum_h[3],
            )
        };
        let transferred = _mm256_mul_ps(frequencies, _mm256_set1_ps(factor));
        let stay = _mm256_set1_ps(stay_factor);
        let mismatch = _mm256_set1_ps(self.mismatch);
        let ones = _mm256_set1_ps(1.0);
        let mut vector_sums: [__m256; HAPLOTYPES] = [
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        ];
        let probability = self.prob.as_mut_ptr();
        let row = relative_locus + self.locus_offset;
        let allele_bytes = self.haplotypes.as_ptr().add(row * self.haplotype_stride);
        let mut k = 0usize;
        let mut probability_index = 0usize;

        if N == 1 {
            let shifts = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let graph = if AMBIGUOUS {
                i32::from((ambiguous_code & 1) != 0)
            } else {
                i32::from(genotype_allele)
            };
            let graph_bits = _mm256_set1_epi32(graph);
            let one = _mm256_set1_epi32(1);
            while k + 7 < self.conditioning_haplotypes {
                let packed_byte = *allele_bytes.add(k >> 3);
                let packed = _mm256_set1_epi32(i32::from(packed_byte));
                let mask = _mm256_slli_epi32(
                    _mm256_xor_si256(
                        _mm256_and_si256(_mm256_srlv_epi32(packed, shifts), one),
                        graph_bits,
                    ),
                    31,
                );
                let mut value = _mm256_fmadd_ps(
                    _mm256_loadu_ps(probability.add(probability_index)),
                    stay,
                    transferred,
                );
                value = _mm256_mul_ps(
                    value,
                    _mm256_blendv_ps(ones, mismatch, _mm256_castsi256_ps(mask)),
                );
                vector_sums[0] = _mm256_add_ps(vector_sums[0], value);
                _mm256_storeu_ps(probability.add(probability_index), value);
                probability_index += HAPLOTYPES;
                k += HAPLOTYPES;
            }
        } else if N == 2 {
            let shifts03 = _mm256_setr_epi32(7, 7, 6, 6, 5, 5, 4, 4);
            let shifts47 = _mm256_setr_epi32(3, 3, 2, 2, 1, 1, 0, 0);
            let graph0 = if AMBIGUOUS {
                i32::from((ambiguous_code & 1) != 0)
            } else {
                i32::from(genotype_allele)
            };
            let graph1 = if AMBIGUOUS {
                i32::from((ambiguous_code & 2) != 0)
            } else {
                i32::from(genotype_allele)
            };
            let graph_bits = _mm256_setr_epi32(
                graph0, graph1, graph0, graph1, graph0, graph1, graph0, graph1,
            );
            let one = _mm256_set1_epi32(1);
            while k + 7 < self.conditioning_haplotypes {
                let packed_byte = *allele_bytes.add(k >> 3);
                let packed = _mm256_set1_epi32(i32::from(packed_byte));
                let mut value03 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(probability.add(probability_index)),
                    stay,
                    transferred,
                );
                let mut value47 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(probability.add(probability_index + HAPLOTYPES)),
                    stay,
                    transferred,
                );
                let mask03 = _mm256_slli_epi32(
                    _mm256_xor_si256(
                        _mm256_and_si256(_mm256_srlv_epi32(packed, shifts03), one),
                        graph_bits,
                    ),
                    31,
                );
                let mask47 = _mm256_slli_epi32(
                    _mm256_xor_si256(
                        _mm256_and_si256(_mm256_srlv_epi32(packed, shifts47), one),
                        graph_bits,
                    ),
                    31,
                );
                value03 = _mm256_mul_ps(
                    value03,
                    _mm256_blendv_ps(ones, mismatch, _mm256_castsi256_ps(mask03)),
                );
                value47 = _mm256_mul_ps(
                    value47,
                    _mm256_blendv_ps(ones, mismatch, _mm256_castsi256_ps(mask47)),
                );
                vector_sums[0] = _mm256_add_ps(vector_sums[0], value03);
                vector_sums[1] = _mm256_add_ps(vector_sums[1], value47);
                _mm256_storeu_ps(probability.add(probability_index), value03);
                _mm256_storeu_ps(probability.add(probability_index + HAPLOTYPES), value47);
                probability_index += 2 * HAPLOTYPES;
                k += HAPLOTYPES;
            }
        } else {
            let mut emission_zero_lanes = [1.0f32; 4];
            let mut emission_one_lanes = [1.0f32; 4];
            for haplotype in 0..4 {
                let graph_haplotype = if AMBIGUOUS {
                    ((ambiguous_code >> haplotype) & 1) != 0
                } else {
                    genotype_allele
                };
                if graph_haplotype {
                    emission_zero_lanes[haplotype] = self.mismatch;
                } else {
                    emission_one_lanes[haplotype] = self.mismatch;
                }
            }
            let emission_zero = _mm_loadu_ps(emission_zero_lanes.as_ptr());
            let emission_one = _mm_loadu_ps(emission_one_lanes.as_ptr());
            while k + 7 < self.conditioning_haplotypes {
                let packed = *allele_bytes.add(k >> 3);
                for pair in 0..4 {
                    let allele0 = ((packed >> (7 - 2 * pair)) & 1) != 0;
                    let allele1 = ((packed >> (6 - 2 * pair)) & 1) != 0;
                    let mut emission =
                        _mm256_castps128_ps256(if allele0 { emission_one } else { emission_zero });
                    emission = _mm256_insertf128_ps(
                        emission,
                        if allele1 { emission_one } else { emission_zero },
                        1,
                    );
                    let value = _mm256_mul_ps(
                        _mm256_fmadd_ps(
                            _mm256_loadu_ps(probability.add(probability_index)),
                            stay,
                            transferred,
                        ),
                        emission,
                    );
                    vector_sums[pair] = _mm256_add_ps(vector_sums[pair], value);
                    _mm256_storeu_ps(probability.add(probability_index), value);
                    probability_index += HAPLOTYPES;
                }
                k += HAPLOTYPES;
            }
        }

        let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        for vector_index in 0..N {
            _mm256_storeu_ps(
                sum_lanes.as_mut_ptr().add(vector_index * HAPLOTYPES),
                vector_sums[vector_index],
            );
        }
        while k < self.conditioning_haplotypes {
            let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
            for haplotype in 0..N {
                let graph_haplotype = if AMBIGUOUS {
                    ((ambiguous_code >> haplotype) & 1) != 0
                } else {
                    genotype_allele
                };
                let mut value = self.prob[probability_index + haplotype]
                    .mul_add(stay_factor, self.prob_sum_h[haplotype] * factor);
                if graph_haplotype != conditioning_allele {
                    value *= self.mismatch;
                }
                sum_lanes[haplotype] += value;
                self.prob[probability_index + haplotype] = value;
            }
            probability_index += N;
            k += 1;
        }

        let mut unique_sums = [0.0f32; HAPLOTYPES];
        for haplotype in 0..N {
            let sum01 = sum_lanes[haplotype] + sum_lanes[N + haplotype];
            let sum23 = sum_lanes[2 * N + haplotype] + sum_lanes[3 * N + haplotype];
            let sum45 = sum_lanes[4 * N + haplotype] + sum_lanes[5 * N + haplotype];
            let sum67 = sum_lanes[6 * N + haplotype] + sum_lanes[7 * N + haplotype];
            unique_sums[haplotype] = (sum01 + sum23) + (sum45 + sum67);
        }
        self.update_total(&unique_sums, N);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_full_avx2<const AMBIGUOUS: bool>(
        &mut self,
        relative_locus: usize,
        genotype_allele: bool,
        ambiguous_code: u8,
        transition: f32,
    ) {
        debug_assert_eq!(self.prob_haps, HAPLOTYPES);
        let factor = transition / (self.conditioning_haplotypes as f32 * self.prob_sum_t);
        let transferred = _mm256_mul_ps(
            _mm256_loadu_ps(self.prob_sum_h.as_ptr()),
            _mm256_set1_ps(factor),
        );
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let stay = _mm256_set1_ps(stay_factor);
        let mismatch = _mm256_set1_ps(self.mismatch);
        let mut vector_sums: [__m256; HAPLOTYPES];
        let probability = self.prob.as_mut_ptr();
        let row = relative_locus + self.locus_offset;
        let allele_bytes = self.haplotypes.as_ptr().add(row * self.haplotype_stride);
        let mut k: usize;
        let mut probability_index: usize;

        if AMBIGUOUS {
            let code = ambiguous_code;
            let mut emission_zero = [1.0f32; HAPLOTYPES];
            let mut emission_one = [1.0f32; HAPLOTYPES];
            for haplotype in 0..HAPLOTYPES {
                if ((code >> haplotype) & 1) != 0 {
                    emission_zero[haplotype] = self.mismatch;
                } else {
                    emission_one[haplotype] = self.mismatch;
                }
            }
            let emission_zero = _mm256_loadu_ps(emission_zero.as_ptr());
            let emission_one = _mm256_loadu_ps(emission_one.as_ptr());
            let block_count = self.conditioning_haplotypes / HAPLOTYPES;
            vector_sums = if self.avx512 {
                Self::run_full_ambiguous_blocks_avx512(
                    probability,
                    allele_bytes,
                    block_count,
                    stay,
                    transferred,
                    code,
                    self.mismatch,
                )
            } else {
                Self::run_full_ambiguous_blocks_avx2(
                    probability,
                    allele_bytes,
                    block_count,
                    stay,
                    transferred,
                    emission_zero,
                    emission_one,
                )
            };
            k = block_count * HAPLOTYPES;
            probability_index = block_count * HAPLOTYPES * HAPLOTYPES;
            while k < self.conditioning_haplotypes {
                let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
                let emission = if conditioning_allele {
                    emission_one
                } else {
                    emission_zero
                };
                let value = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        _mm256_loadu_ps(probability.add(probability_index)),
                        stay,
                        transferred,
                    ),
                    emission,
                );
                vector_sums[0] = _mm256_add_ps(vector_sums[0], value);
                _mm256_storeu_ps(probability.add(probability_index), value);
                probability_index += HAPLOTYPES;
                k += 1;
            }
        } else {
            const FULL_BLOCK_FLOATS: usize = HAPLOTYPES * HAPLOTYPES;
            let block_count = self.conditioning_haplotypes / HAPLOTYPES;
            vector_sums = if self.avx512 {
                Self::run_full_hom_blocks_avx512(
                    probability,
                    allele_bytes,
                    block_count,
                    stay,
                    transferred,
                    self.mismatch,
                    genotype_allele,
                )
            } else {
                Self::run_full_hom_blocks_avx2(
                    probability,
                    allele_bytes,
                    block_count,
                    stay,
                    transferred,
                    mismatch,
                    genotype_allele,
                )
            };
            k = block_count * HAPLOTYPES;
            probability_index = block_count * FULL_BLOCK_FLOATS;
            while k < self.conditioning_haplotypes {
                let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
                let emission = if genotype_allele != conditioning_allele {
                    mismatch
                } else {
                    _mm256_set1_ps(1.0)
                };
                let value = _mm256_mul_ps(
                    _mm256_fmadd_ps(
                        _mm256_loadu_ps(probability.add(probability_index)),
                        stay,
                        transferred,
                    ),
                    emission,
                );
                vector_sums[0] = _mm256_add_ps(vector_sums[0], value);
                _mm256_storeu_ps(probability.add(probability_index), value);
                probability_index += HAPLOTYPES;
                k += 1;
            }
        }

        let sum01 = _mm256_add_ps(vector_sums[0], vector_sums[1]);
        let sum23 = _mm256_add_ps(vector_sums[2], vector_sums[3]);
        let sum45 = _mm256_add_ps(vector_sums[4], vector_sums[5]);
        let sum67 = _mm256_add_ps(vector_sums[6], vector_sums[7]);
        let total = _mm256_add_ps(_mm256_add_ps(sum01, sum23), _mm256_add_ps(sum45, sum67));
        let mut sums = [0.0f32; HAPLOTYPES];
        _mm256_storeu_ps(sums.as_mut_ptr(), total);
        self.update_total(&sums, HAPLOTYPES);
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_full_hom_blocks_avx2(
        probability: *mut f32,
        allele_bytes: *const u8,
        block_count: usize,
        stay: __m256,
        transferred: __m256,
        mismatch: __m256,
        genotype_allele: bool,
    ) -> [__m256; HAPLOTYPES] {
        let probability_cursor = probability;
        let allele_cursor = allele_bytes;
        let blocks = block_count;
        let genotype_mask = if genotype_allele { u8::MAX as usize } else { 0 };
        let emissions = [_mm256_set1_ps(1.0), mismatch];
        let emission_base = emissions.as_ptr();
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        let mut sum5 = _mm256_setzero_ps();
        let mut sum6 = _mm256_setzero_ps();
        let mut sum7 = _mm256_setzero_ps();

        // Keep the eight reduction lanes in registers and consume each state
        // before loading the next. LLVM otherwise schedules all eight updated
        // values together and spills accumulators in this dominant loop.
        asm!(
            "test {blocks}, {blocks}",
            "jz 8f",
            "2:",
            "movzx {packed:e}, byte ptr [{alleles}]",
            "xor {packed}, {genotype_mask}",
            "test {packed:e}, {packed:e}",
            "jz 4f",
            "shl {packed:e}, 24",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 0]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum0}, {sum0}, {value}",
            "vmovups ymmword ptr [{probability} + 0], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 32]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum1}, {sum1}, {value}",
            "vmovups ymmword ptr [{probability} + 32], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 64]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum2}, {sum2}, {value}",
            "vmovups ymmword ptr [{probability} + 64], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 96]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum3}, {sum3}, {value}",
            "vmovups ymmword ptr [{probability} + 96], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 128]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum4}, {sum4}, {value}",
            "vmovups ymmword ptr [{probability} + 128], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 160]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum5}, {sum5}, {value}",
            "vmovups ymmword ptr [{probability} + 160], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 192]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum6}, {sum6}, {value}",
            "vmovups ymmword ptr [{probability} + 192], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 224]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum7}, {sum7}, {value}",
            "vmovups ymmword ptr [{probability} + 224], {value}",
            "jmp 6f",

            "4:",
            "vmovups {value}, ymmword ptr [{probability} + 0]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum0}, {sum0}, {value}",
            "vmovups ymmword ptr [{probability} + 0], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 32]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum1}, {sum1}, {value}",
            "vmovups ymmword ptr [{probability} + 32], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 64]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum2}, {sum2}, {value}",
            "vmovups ymmword ptr [{probability} + 64], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 96]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum3}, {sum3}, {value}",
            "vmovups ymmword ptr [{probability} + 96], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 128]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum4}, {sum4}, {value}",
            "vmovups ymmword ptr [{probability} + 128], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 160]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum5}, {sum5}, {value}",
            "vmovups ymmword ptr [{probability} + 160], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 192]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum6}, {sum6}, {value}",
            "vmovups ymmword ptr [{probability} + 192], {value}",
            "vmovups {value}, ymmword ptr [{probability} + 224]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vaddps {sum7}, {sum7}, {value}",
            "vmovups ymmword ptr [{probability} + 224], {value}",

            "6:",
            "add {probability}, 256",
            "inc {alleles}",
            "dec {blocks}",
            "jnz 2b",
            "8:",
            probability = inout(reg) probability_cursor => _,
            alleles = inout(reg) allele_cursor => _,
            blocks = inout(reg) blocks => _,
            genotype_mask = in(reg) genotype_mask,
            emissions = in(reg) emission_base,
            stay = in(ymm_reg) stay,
            transferred = in(ymm_reg) transferred,
            sum0 = inout(ymm_reg) sum0,
            sum1 = inout(ymm_reg) sum1,
            sum2 = inout(ymm_reg) sum2,
            sum3 = inout(ymm_reg) sum3,
            sum4 = inout(ymm_reg) sum4,
            sum5 = inout(ymm_reg) sum5,
            sum6 = inout(ymm_reg) sum6,
            sum7 = inout(ymm_reg) sum7,
            value = out(ymm_reg) _,
            packed = out(reg) _,
            bit = out(reg) _,
            options(nostack),
        );
        [sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7]
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    // ZMM and opmask instructions are confined to inline assembly. Runtime
    // dispatch guards them; stable Rust only names the AVX2/FMA intrinsics.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_full_hom_blocks_avx512(
        probability: *mut f32,
        allele_bytes: *const u8,
        block_count: usize,
        stay: __m256,
        transferred: __m256,
        mismatch: f32,
        genotype_allele: bool,
    ) -> [__m256; HAPLOTYPES] {
        let mut stay_pair = [0.0f32; 2 * HAPLOTYPES];
        let mut transferred_pair = [0.0f32; 2 * HAPLOTYPES];
        _mm256_storeu_ps(stay_pair.as_mut_ptr(), stay);
        _mm256_storeu_ps(stay_pair.as_mut_ptr().add(HAPLOTYPES), stay);
        _mm256_storeu_ps(transferred_pair.as_mut_ptr(), transferred);
        _mm256_storeu_ps(transferred_pair.as_mut_ptr().add(HAPLOTYPES), transferred);

        // Each two-bit code selects the conditioning-state halves that need
        // the mismatch multiplier. Opmasks replace a 64-byte emission-vector
        // lookup while retaining the exact independent AVX2 reduction lanes.
        let mismatch_masks = [0x0000u16, 0xff00, 0x00ff, 0xffff];
        let mismatch_pair = [mismatch; 2 * HAPLOTYPES];

        let probability_cursor = probability;
        let allele_cursor = allele_bytes;
        let blocks = block_count;
        let genotype_mask = if genotype_allele { u8::MAX as usize } else { 0 };
        let stay_pair_ptr = stay_pair.as_ptr();
        let transferred_pair_ptr = transferred_pair.as_ptr();
        let mismatch_masks_ptr = mismatch_masks.as_ptr();
        let mismatch_pair_ptr = mismatch_pair.as_ptr();
        let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let sum_lanes_ptr = sum_lanes.as_mut_ptr();

        asm!(
            "vmovups zmm5, zmmword ptr [{stay_pair}]",
            "vmovups zmm6, zmmword ptr [{transferred_pair}]",
            "vmovups zmm7, zmmword ptr [{mismatch_pair}]",
            "vpxord zmm0, zmm0, zmm0",
            "vpxord zmm1, zmm1, zmm1",
            "vpxord zmm2, zmm2, zmm2",
            "vpxord zmm3, zmm3, zmm3",
            "test {blocks}, {blocks}",
            "jz 8f",
            "2:",
            "movzx {packed:e}, byte ptr [{alleles}]",
            "xor {packed}, {genotype_mask}",
            "test {packed:e}, {packed:e}",
            "jz 4f",

            "mov {code}, {packed}",
            "shr {code}, 6",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 0]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm0, zmm0, zmm4",
            "vmovups zmmword ptr [{probability} + 0], zmm4",

            "mov {code}, {packed}",
            "shr {code}, 4",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 64]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm1, zmm1, zmm4",
            "vmovups zmmword ptr [{probability} + 64], zmm4",

            "mov {code}, {packed}",
            "shr {code}, 2",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 128]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm2, zmm2, zmm4",
            "vmovups zmmword ptr [{probability} + 128], zmm4",

            "mov {code}, {packed}",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 192]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm3, zmm3, zmm4",
            "vmovups zmmword ptr [{probability} + 192], zmm4",
            "jmp 6f",

            "4:",
            "vmovups zmm4, zmmword ptr [{probability} + 0]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vaddps zmm0, zmm0, zmm4",
            "vmovups zmmword ptr [{probability} + 0], zmm4",
            "vmovups zmm4, zmmword ptr [{probability} + 64]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vaddps zmm1, zmm1, zmm4",
            "vmovups zmmword ptr [{probability} + 64], zmm4",
            "vmovups zmm4, zmmword ptr [{probability} + 128]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vaddps zmm2, zmm2, zmm4",
            "vmovups zmmword ptr [{probability} + 128], zmm4",
            "vmovups zmm4, zmmword ptr [{probability} + 192]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vaddps zmm3, zmm3, zmm4",
            "vmovups zmmword ptr [{probability} + 192], zmm4",

            "6:",
            "add {probability}, 256",
            "inc {alleles}",
            "dec {blocks}",
            "jnz 2b",
            "8:",
            "vmovups zmmword ptr [{sums} + 0], zmm0",
            "vmovups zmmword ptr [{sums} + 64], zmm1",
            "vmovups zmmword ptr [{sums} + 128], zmm2",
            "vmovups zmmword ptr [{sums} + 192], zmm3",
            "vzeroupper",
            probability = inout(reg) probability_cursor => _,
            alleles = inout(reg) allele_cursor => _,
            blocks = inout(reg) blocks => _,
            genotype_mask = in(reg) genotype_mask,
            stay_pair = in(reg) stay_pair_ptr,
            transferred_pair = in(reg) transferred_pair_ptr,
            mismatch_masks = in(reg) mismatch_masks_ptr,
            mismatch_pair = in(reg) mismatch_pair_ptr,
            sums = in(reg) sum_lanes_ptr,
            packed = out(reg) _,
            code = out(reg) _,
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("k1") _,
            options(nostack),
        );

        [
            _mm256_loadu_ps(sum_lanes.as_ptr()),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(8)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(16)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(24)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(32)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(40)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(48)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(56)),
        ]
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    // See run_full_hom_blocks_avx512: the avx512f target feature remains
    // intentionally absent because it is unstable on supported Rust releases.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_full_ambiguous_blocks_avx512(
        probability: *mut f32,
        allele_bytes: *const u8,
        block_count: usize,
        stay: __m256,
        transferred: __m256,
        ambiguous_code: u8,
        mismatch: f32,
    ) -> [__m256; HAPLOTYPES] {
        let mut stay_pair = [0.0f32; 2 * HAPLOTYPES];
        let mut transferred_pair = [0.0f32; 2 * HAPLOTYPES];
        _mm256_storeu_ps(stay_pair.as_mut_ptr(), stay);
        _mm256_storeu_ps(stay_pair.as_mut_ptr().add(HAPLOTYPES), stay);
        _mm256_storeu_ps(transferred_pair.as_mut_ptr(), transferred);
        _mm256_storeu_ps(transferred_pair.as_mut_ptr().add(HAPLOTYPES), transferred);

        let zero_mask = u16::from(ambiguous_code);
        let one_mask = u16::from(!ambiguous_code);
        let mismatch_masks = [
            zero_mask | (zero_mask << 8),
            zero_mask | (one_mask << 8),
            one_mask | (zero_mask << 8),
            one_mask | (one_mask << 8),
        ];
        let mismatch_pair = [mismatch; 2 * HAPLOTYPES];

        let probability_cursor = probability;
        let allele_cursor = allele_bytes;
        let blocks = block_count;
        let stay_pair_ptr = stay_pair.as_ptr();
        let transferred_pair_ptr = transferred_pair.as_ptr();
        let mismatch_masks_ptr = mismatch_masks.as_ptr();
        let mismatch_pair_ptr = mismatch_pair.as_ptr();
        let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let sum_lanes_ptr = sum_lanes.as_mut_ptr();

        asm!(
            "vmovups zmm5, zmmword ptr [{stay_pair}]",
            "vmovups zmm6, zmmword ptr [{transferred_pair}]",
            "vmovups zmm7, zmmword ptr [{mismatch_pair}]",
            "vpxord zmm0, zmm0, zmm0",
            "vpxord zmm1, zmm1, zmm1",
            "vpxord zmm2, zmm2, zmm2",
            "vpxord zmm3, zmm3, zmm3",
            "test {blocks}, {blocks}",
            "jz 4f",
            "2:",
            "movzx {packed:e}, byte ptr [{alleles}]",

            "mov {code}, {packed}",
            "shr {code}, 6",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 0]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm0, zmm0, zmm4",
            "vmovups zmmword ptr [{probability} + 0], zmm4",

            "mov {code}, {packed}",
            "shr {code}, 4",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 64]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm1, zmm1, zmm4",
            "vmovups zmmword ptr [{probability} + 64], zmm4",

            "mov {code}, {packed}",
            "shr {code}, 2",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 128]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm2, zmm2, zmm4",
            "vmovups zmmword ptr [{probability} + 128], zmm4",

            "mov {code}, {packed}",
            "and {code}, 3",
            "movzx {code:e}, word ptr [{mismatch_masks} + {code}*2]",
            "kmovw k1, {code:e}",
            "vmovups zmm4, zmmword ptr [{probability} + 192]",
            "vfmadd132ps zmm4, zmm6, zmm5",
            "vmulps zmm4 {{k1}}, zmm4, zmm7",
            "vaddps zmm3, zmm3, zmm4",
            "vmovups zmmword ptr [{probability} + 192], zmm4",

            "add {probability}, 256",
            "inc {alleles}",
            "dec {blocks}",
            "jnz 2b",
            "4:",
            "vmovups zmmword ptr [{sums} + 0], zmm0",
            "vmovups zmmword ptr [{sums} + 64], zmm1",
            "vmovups zmmword ptr [{sums} + 128], zmm2",
            "vmovups zmmword ptr [{sums} + 192], zmm3",
            "vzeroupper",
            probability = inout(reg) probability_cursor => _,
            alleles = inout(reg) allele_cursor => _,
            blocks = inout(reg) blocks => _,
            stay_pair = in(reg) stay_pair_ptr,
            transferred_pair = in(reg) transferred_pair_ptr,
            mismatch_masks = in(reg) mismatch_masks_ptr,
            mismatch_pair = in(reg) mismatch_pair_ptr,
            sums = in(reg) sum_lanes_ptr,
            packed = out(reg) _,
            code = out(reg) _,
            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("k1") _,
            options(nostack),
        );

        [
            _mm256_loadu_ps(sum_lanes.as_ptr()),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(8)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(16)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(24)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(32)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(40)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(48)),
            _mm256_loadu_ps(sum_lanes.as_ptr().add(56)),
        ]
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_full_ambiguous_blocks_avx2(
        probability: *mut f32,
        allele_bytes: *const u8,
        block_count: usize,
        stay: __m256,
        transferred: __m256,
        emission_zero: __m256,
        emission_one: __m256,
    ) -> [__m256; HAPLOTYPES] {
        let probability_cursor = probability;
        let allele_cursor = allele_bytes;
        let blocks = block_count;
        let emissions = [emission_zero, emission_one];
        let emission_base = emissions.as_ptr();
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        let mut sum5 = _mm256_setzero_ps();
        let mut sum6 = _mm256_setzero_ps();
        let mut sum7 = _mm256_setzero_ps();

        // Keep every state lane and its running sum in registers. In the
        // intrinsic form LLVM spills the eight independent accumulators.
        asm!(
            "test {blocks}, {blocks}",
            "jz 4f",
            "2:",
            "movzx {packed:e}, byte ptr [{alleles}]",
            "shl {packed:e}, 24",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 0]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum0}, {sum0}, {value}",
            "vmovups ymmword ptr [{probability} + 0], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 32]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum1}, {sum1}, {value}",
            "vmovups ymmword ptr [{probability} + 32], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 64]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum2}, {sum2}, {value}",
            "vmovups ymmword ptr [{probability} + 64], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 96]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum3}, {sum3}, {value}",
            "vmovups ymmword ptr [{probability} + 96], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 128]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum4}, {sum4}, {value}",
            "vmovups ymmword ptr [{probability} + 128], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 160]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum5}, {sum5}, {value}",
            "vmovups ymmword ptr [{probability} + 160], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 192]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum6}, {sum6}, {value}",
            "vmovups ymmword ptr [{probability} + 192], {value}",

            "shl {packed:e}, 1",
            "sbb {bit}, {bit}",
            "and {bit}, 32",
            "vmovups {value}, ymmword ptr [{probability} + 224]",
            "vfmadd132ps {value}, {transferred}, {stay}",
            "vmulps {value}, {value}, ymmword ptr [{emissions} + {bit}]",
            "vaddps {sum7}, {sum7}, {value}",
            "vmovups ymmword ptr [{probability} + 224], {value}",

            "add {probability}, 256",
            "inc {alleles}",
            "dec {blocks}",
            "jnz 2b",
            "4:",
            probability = inout(reg) probability_cursor => _,
            alleles = inout(reg) allele_cursor => _,
            blocks = inout(reg) blocks => _,
            emissions = in(reg) emission_base,
            stay = in(ymm_reg) stay,
            transferred = in(ymm_reg) transferred,
            sum0 = inout(ymm_reg) sum0,
            sum1 = inout(ymm_reg) sum1,
            sum2 = inout(ymm_reg) sum2,
            sum3 = inout(ymm_reg) sum3,
            sum4 = inout(ymm_reg) sum4,
            sum5 = inout(ymm_reg) sum5,
            sum6 = inout(ymm_reg) sum6,
            sum7 = inout(ymm_reg) sum7,
            value = out(ymm_reg) _,
            packed = out(reg) _,
            bit = out(reg) _,
            options(nostack),
        );
        [sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7]
    }

    fn run_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize, transition: f32) {
        let code = self.ambiguous[ambiguous_index];
        if self.prob_haps == 1 {
            self.run_reduced::<false>(relative_locus, (code & 1) != 0, 0, transition);
        } else {
            self.run_reduced::<true>(relative_locus, false, code, transition);
        }
    }

    fn run_missing(&mut self, transition: f32) {
        self.reshape_haplotypes(HAPLOTYPES);
        let factor = transition / (self.conditioning_haplotypes as f32 * self.prob_sum_t);
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let mut sums = [0.0f32; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let value = self.prob[start + h].mul_add(stay_factor, self.prob_sum_h[h] * factor);
                self.prob[start + h] = value;
                sums[h] += value;
            }
        }
        self.update_total(&sums, HAPLOTYPES);
    }

    fn collapse_hom(&mut self, locus: usize, relative_locus: usize, transition: f32) {
        let genotype_allele = self.hap0(locus);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.collapse_avx2::<0>(relative_locus, genotype_allele, 0, transition);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.collapse_neon::<0>(relative_locus, genotype_allele, 0, transition);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.collapse(relative_locus, Some(genotype_allele), None, transition);
    }

    fn collapse_ambiguous(
        &mut self,
        relative_locus: usize,
        ambiguous_index: usize,
        transition: f32,
    ) {
        let ambiguous_code = self.ambiguous[ambiguous_index];
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.collapse_avx2::<1>(relative_locus, false, ambiguous_code, transition);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.collapse_neon::<1>(relative_locus, false, ambiguous_code, transition);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.collapse(relative_locus, None, Some(ambiguous_code), transition);
    }

    fn collapse_missing(&mut self, transition: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.collapse_avx2::<2>(0, false, 0, transition);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.collapse_neon::<2>(0, false, 0, transition);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        self.collapse(0, None, None, transition);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn collapse(
        &mut self,
        relative_locus: usize,
        genotype_allele: Option<bool>,
        ambiguous_code: Option<u8>,
        transition: f32,
    ) {
        debug_assert_eq!(self.prob_haps, HAPLOTYPES);
        let transferred = transition / self.conditioning_haplotypes as f32;
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let mut sums = [0.0f32; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = if genotype_allele.is_some() || ambiguous_code.is_some() {
                self.allele(relative_locus, k)
            } else {
                false
            };
            let base = self.prob_sum_k[k].mul_add(stay_factor, transferred);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let graph_haplotype =
                    genotype_allele.or_else(|| ambiguous_code.map(|code| ((code >> h) & 1) != 0));
                let emission = if graph_haplotype.is_some_and(|graph| graph != allele) {
                    self.mismatch
                } else {
                    1.0
                };
                let value = base * emission;
                self.prob[start + h] = value;
                sums[h] += value;
            }
        }
        self.update_total(&sums, HAPLOTYPES);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn collapse_neon<const KIND: u8>(
        &mut self,
        relative_locus: usize,
        genotype_allele: bool,
        ambiguous_code: u8,
        transition: f32,
    ) {
        debug_assert!(KIND <= 2);
        debug_assert_eq!(self.prob_haps, HAPLOTYPES);
        let transferred = transition / self.conditioning_haplotypes as f32;
        let stay_factor = (1.0 - transition) / self.prob_sum_t;
        let allele_bytes = if KIND < 2 {
            let row = relative_locus + self.locus_offset;
            self.haplotypes.as_ptr().add(row * self.haplotype_stride)
        } else {
            core::ptr::null()
        };
        let sums = neon::collapse::<KIND>(
            self.prob.as_mut_ptr(),
            self.prob_sum_k.as_ptr(),
            self.conditioning_haplotypes,
            allele_bytes,
            transferred,
            stay_factor,
            self.mismatch,
            genotype_allele,
            ambiguous_code,
        );
        self.update_total(&sums, HAPLOTYPES);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn collapse_avx2<const KIND: u8>(
        &mut self,
        relative_locus: usize,
        genotype_allele: bool,
        ambiguous_code: u8,
        transition: f32,
    ) {
        debug_assert!(KIND <= 2);
        debug_assert_eq!(self.prob_haps, HAPLOTYPES);
        let transferred = _mm256_set1_ps(transition / self.conditioning_haplotypes as f32);
        let stay = _mm256_set1_ps((1.0 - transition) / self.prob_sum_t);
        let ones = _mm256_set1_ps(1.0);
        let mismatch = _mm256_set1_ps(self.mismatch);
        let mut emission_zero_lanes = [1.0f32; HAPLOTYPES];
        let mut emission_one_lanes = [1.0f32; HAPLOTYPES];
        for haplotype in 0..HAPLOTYPES {
            if ((ambiguous_code >> haplotype) & 1) != 0 {
                emission_zero_lanes[haplotype] = self.mismatch;
            } else {
                emission_one_lanes[haplotype] = self.mismatch;
            }
        }
        let emission_zero = _mm256_loadu_ps(emission_zero_lanes.as_ptr());
        let emission_one = _mm256_loadu_ps(emission_one_lanes.as_ptr());
        let mut vector_sums: [__m256; HAPLOTYPES] = [
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        ];
        let probability = self.prob.as_mut_ptr();
        let row = relative_locus + self.locus_offset;
        let allele_bytes = self.haplotypes.as_ptr().add(row * self.haplotype_stride);
        let mut k = 0usize;
        let mut probability_index = 0usize;

        while k + 7 < self.conditioning_haplotypes {
            let packed = if KIND < 2 {
                *allele_bytes.add(k >> 3)
            } else {
                0
            };
            for lane in 0..HAPLOTYPES {
                let conditioning_allele = ((packed >> (7 - lane)) & 1) != 0;
                let mut value =
                    _mm256_fmadd_ps(_mm256_set1_ps(self.prob_sum_k[k + lane]), stay, transferred);
                if KIND == 0 {
                    let emission = if genotype_allele != conditioning_allele {
                        mismatch
                    } else {
                        ones
                    };
                    value = _mm256_mul_ps(value, emission);
                } else if KIND == 1 {
                    let emission = if conditioning_allele {
                        emission_one
                    } else {
                        emission_zero
                    };
                    value = _mm256_mul_ps(value, emission);
                }
                vector_sums[lane] = _mm256_add_ps(vector_sums[lane], value);
                _mm256_storeu_ps(probability.add(probability_index), value);
                probability_index += HAPLOTYPES;
            }
            k += HAPLOTYPES;
        }

        while k < self.conditioning_haplotypes {
            let conditioning_allele = if KIND < 2 {
                ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0
            } else {
                false
            };
            let mut value = _mm256_fmadd_ps(_mm256_set1_ps(self.prob_sum_k[k]), stay, transferred);
            if KIND == 0 {
                let emission = if genotype_allele != conditioning_allele {
                    mismatch
                } else {
                    ones
                };
                value = _mm256_mul_ps(value, emission);
            } else if KIND == 1 {
                let emission = if conditioning_allele {
                    emission_one
                } else {
                    emission_zero
                };
                value = _mm256_mul_ps(value, emission);
            }
            vector_sums[0] = _mm256_add_ps(vector_sums[0], value);
            _mm256_storeu_ps(probability.add(probability_index), value);
            probability_index += HAPLOTYPES;
            k += 1;
        }

        let sum01 = _mm256_add_ps(vector_sums[0], vector_sums[1]);
        let sum23 = _mm256_add_ps(vector_sums[2], vector_sums[3]);
        let sum45 = _mm256_add_ps(vector_sums[4], vector_sums[5]);
        let sum67 = _mm256_add_ps(vector_sums[6], vector_sums[7]);
        let total = _mm256_add_ps(_mm256_add_ps(sum01, sum23), _mm256_add_ps(sum45, sum67));
        let mut sums = [0.0f32; HAPLOTYPES];
        _mm256_storeu_ps(sums.as_mut_ptr(), total);
        self.update_total(&sums, HAPLOTYPES);
    }

    fn sum_conditioning_haplotypes(&mut self) {
        if self.prob_haps < HAPLOTYPES {
            let multiplicity = (HAPLOTYPES / self.prob_haps) as f32;
            for k in 0..self.conditioning_haplotypes {
                let start = k * self.prob_haps;
                let mut sum = self.prob[start];
                for h in 1..self.prob_haps {
                    sum += self.prob[start + h];
                }
                self.prob_sum_k[k] = sum * multiplicity;
            }
        } else {
            for k in 0..self.conditioning_haplotypes {
                let start = k * HAPLOTYPES;
                self.prob_sum_k[k] = self.prob[start]
                    + self.prob[start + 1]
                    + self.prob[start + 2]
                    + self.prob[start + 3]
                    + self.prob[start + 4]
                    + self.prob[start + 5]
                    + self.prob[start + 6]
                    + self.prob[start + 7];
            }
        }
    }

    fn save_alpha(&mut self, relative_segment: usize, previous_locus: usize) {
        let start = self.alpha_offsets[relative_segment];
        let stop = self.alpha_offsets[relative_segment + 1];
        self.alpha[start..stop].copy_from_slice(&self.prob[..stop - start]);
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

impl SingleEngine<'_> {
    fn transition_haplotypes(&mut self, relative_segment: usize, previous_locus: usize) -> bool {
        let alpha_segment = relative_segment - 1;
        let previous_haplotypes = self.segment_hap_count[alpha_segment];
        let current_haplotypes = self.segment_hap_count[relative_segment];
        let alpha_total = self.alpha_sum_sum[alpha_segment];
        let transition =
            self.transition_probability(self.alpha_locus[alpha_segment] as usize, previous_locus);
        let stay_factor = (1.0 - transition) / alpha_total;
        let alpha_start = self.alpha_offsets[alpha_segment];
        let alpha_sum_start = alpha_segment * HAPLOTYPES;

        if previous_haplotypes < HAPLOTYPES || current_haplotypes < HAPLOTYPES {
            for h1 in 0..previous_haplotypes {
                let transferred = (self.alpha_sum[alpha_sum_start + h1] / alpha_total) * transition
                    / self.conditioning_haplotypes as f32;
                let mut sums = [0.0f32; HAPLOTYPES];
                for k in 0..self.conditioning_haplotypes {
                    let alpha = self.alpha[alpha_start + k * previous_haplotypes + h1]
                        .mul_add(stay_factor, transferred);
                    for h2 in 0..current_haplotypes {
                        sums[h2] = alpha.mul_add(self.prob[k * current_haplotypes + h2], sums[h2]);
                    }
                }
                let row = h1 * HAPLOTYPES;
                self.h_probs[row..row + current_haplotypes]
                    .copy_from_slice(&sums[..current_haplotypes]);
            }
            let mut total = 0.0f32;
            for h1 in 0..HAPLOTYPES {
                for h2 in 0..HAPLOTYPES {
                    let value = self.h_probs
                        [(h1 % previous_haplotypes) * HAPLOTYPES + h2 % current_haplotypes];
                    self.h_probs[h1 * HAPLOTYPES + h2] = value;
                    total += value;
                }
            }
            self.sum_h_probs = total;
        } else {
            #[cfg(target_arch = "x86_64")]
            {
                // The enclosing SHAPEIT common-phasing binary already requires AVX2 and FMA.
                return unsafe {
                    self.transition_haplotypes_full_avx2(
                        alpha_start,
                        alpha_sum_start,
                        alpha_total,
                        transition,
                        stay_factor,
                    )
                };
            }

            #[cfg(target_arch = "aarch64")]
            {
                return unsafe {
                    self.transition_haplotypes_full_neon(
                        alpha_start,
                        alpha_sum_start,
                        alpha_total,
                        transition,
                        stay_factor,
                    )
                };
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let mut total = 0.0f32;
                for h1 in 0..HAPLOTYPES {
                    let transferred = (self.alpha_sum[alpha_sum_start + h1] / alpha_total)
                        * transition
                        / self.conditioning_haplotypes as f32;
                    let mut sums = [0.0f32; HAPLOTYPES];
                    for k in 0..self.conditioning_haplotypes {
                        let alpha = self.alpha[alpha_start + k * HAPLOTYPES + h1]
                            .mul_add(stay_factor, transferred);
                        for h2 in 0..HAPLOTYPES {
                            sums[h2] = alpha.mul_add(self.prob[k * HAPLOTYPES + h2], sums[h2]);
                        }
                    }
                    let row = h1 * HAPLOTYPES;
                    self.h_probs[row..row + HAPLOTYPES].copy_from_slice(&sums);
                    total += sums[0]
                        + sums[1]
                        + sums[2]
                        + sums[3]
                        + sums[4]
                        + sums[5]
                        + sums[6]
                        + sums[7];
                }
                self.sum_h_probs = total;
            }
        }
        self.sum_h_probs.is_nan()
            || self.sum_h_probs.is_infinite()
            || self.sum_h_probs < f32::MIN_POSITIVE
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn transition_haplotypes_full_neon(
        &mut self,
        alpha_start: usize,
        alpha_sum_start: usize,
        alpha_total: f32,
        transition: f32,
        stay_factor: f32,
    ) -> bool {
        let total = neon::transition_full(
            self.alpha.as_ptr().add(alpha_start),
            self.prob.as_ptr(),
            self.alpha_sum.as_ptr().add(alpha_sum_start),
            self.conditioning_haplotypes,
            alpha_total,
            transition,
            stay_factor,
            self.h_probs.as_mut_ptr(),
        );
        self.sum_h_probs = total;
        total.is_nan() || total.is_infinite() || total < f32::MIN_POSITIVE
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn transition_haplotypes_full_avx2(
        &mut self,
        alpha_start: usize,
        alpha_sum_start: usize,
        alpha_total: f32,
        transition: f32,
        stay_factor: f32,
    ) -> bool {
        let mut sums: [__m256; HAPLOTYPES] = [
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        ];
        let mut transferred = [0.0f32; HAPLOTYPES];
        for h1 in 0..HAPLOTYPES {
            transferred[h1] = (self.alpha_sum[alpha_sum_start + h1] / alpha_total) * transition
                / self.conditioning_haplotypes as f32;
        }
        let alpha = self.alpha.as_ptr().add(alpha_start);
        let beta = self.prob.as_ptr();
        for k in 0..self.conditioning_haplotypes {
            let beta8 = _mm256_loadu_ps(beta.add(k * HAPLOTYPES));
            for h1 in 0..HAPLOTYPES {
                let alpha_value =
                    (*alpha.add(k * HAPLOTYPES + h1)).mul_add(stay_factor, transferred[h1]);
                sums[h1] = _mm256_fmadd_ps(_mm256_set1_ps(alpha_value), beta8, sums[h1]);
            }
        }
        let mut total = 0.0f32;
        for h1 in 0..HAPLOTYPES {
            let row = h1 * HAPLOTYPES;
            _mm256_storeu_ps(self.h_probs.as_mut_ptr().add(row), sums[h1]);
            total += self.h_probs[row]
                + self.h_probs[row + 1]
                + self.h_probs[row + 2]
                + self.h_probs[row + 3]
                + self.h_probs[row + 4]
                + self.h_probs[row + 5]
                + self.h_probs[row + 6]
                + self.h_probs[row + 7];
        }
        self.sum_h_probs = total;
        total.is_nan() || total.is_infinite() || total < f32::MIN_POSITIVE
    }

    fn transition_haplotypes_double(
        &mut self,
        relative_segment: usize,
        previous_locus: usize,
    ) -> bool {
        let alpha_segment = relative_segment - 1;
        let previous_haplotypes = self.segment_hap_count[alpha_segment];
        let current_haplotypes = self.segment_hap_count[relative_segment];
        let alpha_total = f64::from(self.alpha_sum_sum[alpha_segment]);
        let transition =
            self.transition_probability(self.alpha_locus[alpha_segment] as usize, previous_locus);
        let stay_factor = f64::from(1.0 - transition) / alpha_total;
        let alpha_start = self.alpha_offsets[alpha_segment];
        let alpha_sum_start = alpha_segment * HAPLOTYPES;

        #[cfg(target_arch = "x86_64")]
        if previous_haplotypes == HAPLOTYPES && current_haplotypes == HAPLOTYPES {
            // The enclosing SHAPEIT common-phasing binary already requires AVX2 and FMA.
            return unsafe {
                self.transition_haplotypes_double_full_avx2(
                    alpha_start,
                    alpha_sum_start,
                    alpha_total,
                    transition,
                    stay_factor,
                )
            };
        }

        for h1 in 0..previous_haplotypes {
            let transferred = (f64::from(self.alpha_sum[alpha_sum_start + h1]) / alpha_total)
                * f64::from(transition)
                / self.conditioning_haplotypes as f64;
            for h2 in 0..current_haplotypes {
                let mut sum = 0.0f64;
                for k in 0..self.conditioning_haplotypes {
                    let alpha = f64::from(self.alpha[alpha_start + k * previous_haplotypes + h1])
                        .mul_add(stay_factor, transferred);
                    sum = alpha.mul_add(f64::from(self.prob[k * current_haplotypes + h2]), sum);
                }
                self.h_probs_double[h1 * HAPLOTYPES + h2] = sum;
            }
        }
        let mut total = 0.0f64;
        for h1 in 0..HAPLOTYPES {
            for h2 in 0..HAPLOTYPES {
                let value = self.h_probs_double
                    [(h1 % previous_haplotypes) * HAPLOTYPES + h2 % current_haplotypes];
                self.h_probs_double[h1 * HAPLOTYPES + h2] = value;
                total += value;
            }
        }
        self.sum_h_probs_double = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn transition_haplotypes_double_full_avx2(
        &mut self,
        alpha_start: usize,
        alpha_sum_start: usize,
        alpha_total: f64,
        transition: f32,
        stay_factor: f64,
    ) -> bool {
        let alpha = self.alpha.as_ptr().add(alpha_start);
        let beta = self.prob.as_ptr();
        let mut total = 0.0f64;
        for h1 in 0..HAPLOTYPES {
            let mut sum0 = _mm256_setzero_pd();
            let mut sum1 = _mm256_setzero_pd();
            let transferred = (f64::from(self.alpha_sum[alpha_sum_start + h1]) / alpha_total)
                * f64::from(transition)
                / self.conditioning_haplotypes as f64;
            for k in 0..self.conditioning_haplotypes {
                let alpha_value =
                    f64::from(*alpha.add(k * HAPLOTYPES + h1)).mul_add(stay_factor, transferred);
                let alpha4 = _mm256_set1_pd(alpha_value);
                let beta0: __m128 = _mm_loadu_ps(beta.add(k * HAPLOTYPES));
                let beta1: __m128 = _mm_loadu_ps(beta.add(k * HAPLOTYPES + 4));
                sum0 = _mm256_fmadd_pd(alpha4, _mm256_cvtps_pd(beta0), sum0);
                sum1 = _mm256_fmadd_pd(alpha4, _mm256_cvtps_pd(beta1), sum1);
            }
            let row = h1 * HAPLOTYPES;
            _mm256_storeu_pd(self.h_probs_double.as_mut_ptr().add(row), sum0);
            _mm256_storeu_pd(self.h_probs_double.as_mut_ptr().add(row + 4), sum1);
            for h2 in 0..HAPLOTYPES {
                total += self.h_probs_double[row + h2];
            }
        }
        self.sum_h_probs_double = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    fn transition_diplotypes_multiply(&mut self, previous: u64, current: u64) -> bool {
        let scaling = 1.0 / f64::from(self.sum_h_probs);
        let mut total = 0.0f64;
        let mut index = 0usize;
        let mut previous_active = previous;
        while previous_active != 0 {
            let previous_diplotype = previous_active.trailing_zeros() as usize;
            previous_active &= previous_active - 1;
            let mut current_active = current;
            while current_active != 0 {
                let current_diplotype = current_active.trailing_zeros() as usize;
                current_active &= current_active - 1;
                let first = f64::from(
                    self.h_probs[(previous_diplotype >> 3) * HAPLOTYPES + (current_diplotype >> 3)],
                ) * scaling;
                let second = f64::from(
                    self.h_probs[(previous_diplotype & 7) * HAPLOTYPES + (current_diplotype & 7)],
                ) * scaling;
                let value = first * second;
                self.d_probs[index] = value;
                total += value;
                index += 1;
            }
        }
        self.sum_d_probs = total;
        total.is_nan() || total.is_infinite() || total < f64::MIN_POSITIVE
    }

    fn transition_diplotypes_multiply_double(&mut self, previous: u64, current: u64) -> bool {
        let scaling = 1.0 / self.sum_h_probs_double;
        let mut total = 0.0f64;
        let mut index = 0usize;
        let mut previous_active = previous;
        while previous_active != 0 {
            let previous_diplotype = previous_active.trailing_zeros() as usize;
            previous_active &= previous_active - 1;
            let mut current_active = current;
            while current_active != 0 {
                let current_diplotype = current_active.trailing_zeros() as usize;
                current_active &= current_active - 1;
                let first = self.h_probs_double
                    [(previous_diplotype >> 3) * HAPLOTYPES + (current_diplotype >> 3)]
                    * scaling;
                let second = self.h_probs_double
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
        let scaling = 1.0 / f64::from(self.sum_h_probs);
        let mut total = 0.0f64;
        let mut index = 0usize;
        let mut previous_active = previous;
        while previous_active != 0 {
            let previous_diplotype = previous_active.trailing_zeros() as usize;
            previous_active &= previous_active - 1;
            let mut current_active = current;
            while current_active != 0 {
                let current_diplotype = current_active.trailing_zeros() as usize;
                current_active &= current_active - 1;
                let first = f64::from(
                    self.h_probs[(previous_diplotype >> 3) * HAPLOTYPES + (current_diplotype >> 3)],
                ) * scaling;
                let second = f64::from(
                    self.h_probs[(previous_diplotype & 7) * HAPLOTYPES + (current_diplotype & 7)],
                );
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
        if !self.prob_sum_t.is_finite() || self.prob_sum_t < f32::MIN_POSITIVE {
            return true;
        }
        let scale = f64::from(1.0f32 / self.prob_sum_t);
        let mut probabilities = [0.0f64; 64];
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut active = self.diplotypes[0];
        while active != 0 {
            let diplotype = active.trailing_zeros() as usize;
            active &= active - 1;
            let value = (f64::from(self.prob_sum_h[diplotype >> 3]) * scale)
                * (f64::from(self.prob_sum_h[diplotype & 7]) * scale);
            probabilities[count] = value;
            total += value;
            count += 1;
        }
        // The dense HMM can retain finite mass while every diplotype allowed
        // by the first graph segment has underflowed to zero. Signal this just
        // like the guarded inter-segment contractions so the window is rerun
        // in double precision before any invalid value is published.
        if !total.is_finite() || total < f64::MIN_POSITIVE {
            return true;
        }
        let scaling = 1.0 / total;
        for (target, &value) in self.transition_probabilities[..count]
            .iter_mut()
            .zip(probabilities.iter())
        {
            *target = value * scaling;
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
        let previous = self.diplotypes[segment - 1];
        let current = self.diplotypes[segment];
        let mut recovered = 0;
        if self.transition_haplotypes(relative_segment, previous_locus) {
            if self.transition_haplotypes_double(relative_segment, previous_locus)
                || self.transition_diplotypes_multiply_double(previous, current)
            {
                return -1;
            }
        } else if self.transition_diplotypes_multiply(previous, current)
            && (self.transition_haplotypes_double(relative_segment, previous_locus)
                || self.transition_diplotypes_multiply_double(previous, current))
        {
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
        let mut sums = [[0.0f32; HAPLOTYPES]; 2];
        for k in 0..self.conditioning_haplotypes {
            let allele = usize::from(self.allele(relative_locus, k));
            let conditioning_start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let reciprocal = 1.0 / self.alpha_sum_missing[sum_start + h];
                let value = (self.alpha_missing[state_start + conditioning_start + h] * reciprocal)
                    * self.prob[conditioning_start + h];
                sums[allele][h] += value;
            }
        }
        let output_start = absolute_missing * HAPLOTYPES;
        for h in 0..HAPLOTYPES {
            self.missing_probabilities[output_start + h] = sums[1][h] / (sums[0][h] + sums[1][h]);
        }
    }
}

impl SingleEngine<'_> {
    fn forward(&mut self) {
        let mut segment = self.segment_first;
        let mut segment_locus = 0usize;
        let mut ambiguous_index = self.ambiguous_first;
        let mut missing_index = self.missing_first;
        let mut previous_locus = self.locus_first;
        self.prob_haps = HAPLOTYPES;

        for locus in self.locus_first..=self.locus_last {
            let relative_locus = locus - self.locus_first;
            let relative_segment = segment - self.segment_first;
            let segment_haplotypes = self.segment_hap_count[relative_segment];
            let desired_haplotypes =
                if segment_locus < self.segment_first_ambiguous[relative_segment] {
                    1
                } else {
                    segment_haplotypes
                };
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
            if relative_locus != 0 && segment_locus != 0 {
                self.reshape_haplotypes(desired_haplotypes);
            }

            if relative_locus == 0 {
                self.prob_haps = HAPLOTYPES;
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
            } else {
                self.prob_haps = HAPLOTYPES;
                if homozygous {
                    self.collapse_hom(locus, relative_locus, transition);
                } else if ambiguous {
                    self.collapse_ambiguous(relative_locus, ambiguous_index, transition);
                } else {
                    self.collapse_missing(transition);
                }
            }
            if update_previous {
                previous_locus = locus;
            }

            if missing {
                self.save_missing(relative_missing);
                missing_index += 1;
            }
            if segment_locus + 1 == self.segment_lengths[segment] as usize {
                self.reshape_haplotypes(segment_haplotypes);
                self.sum_conditioning_haplotypes();
                self.save_alpha(relative_segment, previous_locus);
            } else {
                self.reshape_haplotypes(desired_haplotypes);
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
        let mut recovered = 0;
        let mut segment = self.segment_last;
        let mut segment_locus = self.segment_lengths[segment] as isize - 1;
        let mut ambiguous_index = self.ambiguous_last;
        let mut missing_index = self.missing_last;
        let mut transition_cursor = self.transition_last as isize;
        let mut previous_locus = self.locus_last;
        self.prob_haps = HAPLOTYPES;

        for locus in (self.locus_first..=self.locus_last).rev() {
            let relative_locus = locus - self.locus_first;
            let relative_segment = segment - self.segment_first;
            let segment_haplotypes = self.segment_hap_count[relative_segment];
            let desired_haplotypes =
                if segment_locus as usize > self.segment_last_ambiguous[relative_segment] {
                    1
                } else {
                    segment_haplotypes
                };
            let ambiguous = self.is_ambiguous(locus);
            let missing = self.is_missing(locus);
            let homozygous = !(ambiguous || missing);
            let transition = if locus == self.locus_last {
                0.0
            } else {
                self.transition_probability(previous_locus, locus)
            };
            let mut update_previous = true;
            if locus != self.locus_last
                && segment_locus + 1 != self.segment_lengths[segment] as isize
            {
                self.reshape_haplotypes(desired_haplotypes);
            }

            if locus == self.locus_last {
                self.prob_haps = HAPLOTYPES;
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
            } else {
                self.prob_haps = HAPLOTYPES;
                if homozygous {
                    self.collapse_hom(locus, relative_locus, transition);
                } else if ambiguous {
                    self.collapse_ambiguous(relative_locus, ambiguous_index as usize, transition);
                } else {
                    self.collapse_missing(transition);
                }
            }
            if update_previous {
                previous_locus = locus;
            }
            if missing {
                let relative_missing = (missing_index - self.missing_first as isize) as usize;
                self.impute(relative_locus, relative_missing, missing_index as usize);
                missing_index -= 1;
            }
            if segment_locus == 0 {
                self.reshape_haplotypes(segment_haplotypes);
                self.sum_conditioning_haplotypes();
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
                recovered += result;
            }
            if segment_locus != 0 {
                self.reshape_haplotypes(desired_haplotypes);
            }

            segment_locus -= 1;
            ambiguous_index -= isize::from(ambiguous);
            if segment_locus < 0 && segment > 0 {
                segment -= 1;
                segment_locus = self.segment_lengths[segment] as isize - 1;
            }
        }
        recovered
    }

    fn run(&mut self) -> i32 {
        self.forward();
        self.backward()
    }
}
#[no_mangle]
/// Return the caller-owned workspaces required by one single-precision window.
///
/// # Safety
///
/// `parameters` and all three output lengths must be valid for their types.
/// Every non-empty immutable buffer in `parameters` must be readable for its
/// stated length. No phasing output or scratch buffer is read or written.
pub unsafe extern "C" fn shapeit_hmm_single_scratch_len_v1(
    parameters: *const HmmSegmentSingleV1,
    float_scratch_length: *mut usize,
    alpha_locus_scratch_length: *mut usize,
    index_scratch_length: *mut usize,
) -> u32 {
    if parameters.is_null()
        || float_scratch_length.is_null()
        || alpha_locus_scratch_length.is_null()
        || index_scratch_length.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let parameters = &*parameters;
    if parameters.struct_size as usize != mem::size_of::<HmmSegmentSingleV1>() {
        return STATUS_INVALID_DIMENSIONS;
    }
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
    let shadow = single_validation_shadow(parameters);
    let validated = match validate(&shadow, variants, segment_lengths, diplotypes) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let layout = match single_scratch_layout(
        parameters,
        variants,
        ambiguous,
        segment_lengths,
        validated,
        None,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    *float_scratch_length = layout.float_total;
    *alpha_locus_scratch_length = layout.segment_count;
    *index_scratch_length = layout.index_total;
    STATUS_OK
}
unsafe fn run_segment_single_v1_impl(
    parameters: *const HmmSegmentSingleV1,
    outcome: *mut i32,
    prevalidated: bool,
) -> u32 {
    if parameters.is_null() || outcome.is_null() {
        return STATUS_NULL_POINTER;
    }
    let parameters = &*parameters;
    if parameters.struct_size as usize != mem::size_of::<HmmSegmentSingleV1>() {
        return STATUS_INVALID_DIMENSIONS;
    }
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
    let validated = match if prevalidated {
        single_prevalidated_layout(parameters)
    } else {
        let shadow = single_validation_shadow(parameters);
        validate(&shadow, variants, segment_lengths, diplotypes)
    } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let required_indexes = match validated
        .scratch
        .segment_count
        .checked_mul(4)
        .and_then(|value| value.checked_add(1))
    {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    for result in [
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
        require_mut_pointer(parameters.index_scratch, parameters.index_scratch_length),
    ] {
        if let Err(status) = result {
            return status;
        }
    }
    if parameters.alpha_locus_scratch_length < validated.scratch.segment_count
        || parameters.index_scratch_length < required_indexes
    {
        return STATUS_OUT_OF_BOUNDS;
    }

    let indexes = mut_slice(parameters.index_scratch, required_indexes);
    indexes.fill(0);
    let layout = match single_scratch_layout(
        parameters,
        variants,
        ambiguous,
        segment_lengths,
        validated,
        Some(indexes),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if parameters.scratch_length < layout.float_total {
        return STATUS_OUT_OF_BOUNDS;
    }

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
    let scratch = mut_slice(parameters.scratch, layout.float_total);
    let alpha_locus = mut_slice(parameters.alpha_locus_scratch, layout.segment_count);
    scratch.fill(0.0);
    alpha_locus.fill(0);

    let (prob, scratch) = scratch.split_at_mut(layout.states);
    let (prob_sum_k, scratch) = scratch.split_at_mut(parameters.conditioning_haplotypes);
    let (alpha, scratch) = scratch.split_at_mut(layout.alpha_values);
    let alpha_sum_length = layout.segment_count * HAPLOTYPES;
    let (alpha_sum, scratch) = scratch.split_at_mut(alpha_sum_length);
    let (alpha_sum_sum, scratch) = scratch.split_at_mut(layout.segment_count);
    let alpha_missing_length = layout.missing_count * layout.states;
    let (alpha_missing, scratch) = scratch.split_at_mut(alpha_missing_length);
    let alpha_sum_missing_length = layout.missing_count * HAPLOTYPES;
    let (alpha_sum_missing, remainder) = scratch.split_at_mut(alpha_sum_missing_length);
    debug_assert!(remainder.is_empty());

    let (segment_hap_count, indexes) = indexes.split_at(layout.segment_count);
    let (segment_first_ambiguous, indexes) = indexes.split_at(layout.segment_count);
    let (segment_last_ambiguous, alpha_offsets) = indexes.split_at(layout.segment_count);
    debug_assert_eq!(alpha_offsets.len(), layout.segment_count + 1);

    let mut engine = SingleEngine {
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
        #[cfg(target_arch = "x86_64")]
        avx512: avx512_full_kernel_available(),
        segment_first: validated.segment_first,
        segment_last: validated.segment_last,
        locus_first: validated.locus_first,
        locus_last: validated.locus_last,
        ambiguous_first: validated.ambiguous_first,
        ambiguous_last: parameters.ambiguous_last as isize,
        missing_first: validated.missing_first,
        missing_last: parameters.missing_last as isize,
        transition_last: validated.transition_last,
        segment_hap_count,
        segment_first_ambiguous,
        segment_last_ambiguous,
        alpha_offsets,
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
        prob_haps: HAPLOTYPES,
        prob_sum_t: 0.0,
        prob_sum_h: [0.0; HAPLOTYPES],
        sum_h_probs: 0.0,
        sum_h_probs_double: 0.0,
        sum_d_probs: 0.0,
        h_probs: [0.0; HAPLOTYPES * HAPLOTYPES],
        h_probs_double: [0.0; HAPLOTYPES * HAPLOTYPES],
        d_probs: [0.0; HAPLOTYPES * HAPLOTYPES * HAPLOTYPES * HAPLOTYPES],
    };
    *outcome = engine.run();
    STATUS_OK
}

#[no_mangle]
/// Run one complete single-precision common-phasing HMM window.
///
/// # Safety
///
/// `parameters` and `outcome` must be valid for their types. Every non-empty
/// buffer in `parameters` must be valid for its stated length. Mutable buffers
/// must not overlap each other or any input buffer. Invalid layouts are
/// reported before any caller-owned phasing output is written.
pub unsafe extern "C" fn shapeit_hmm_run_segment_single_v1(
    parameters: *const HmmSegmentSingleV1,
    outcome: *mut i32,
) -> u32 {
    run_segment_single_v1_impl(parameters, outcome, false)
}

#[no_mangle]
/// Run a single-precision HMM window whose layout was already validated.
///
/// # Safety
///
/// In addition to the requirements of `shapeit_hmm_run_segment_single_v1`,
/// every coordinate and buffer length must describe a valid, mutually
/// consistent HMM window. This entry point performs only constant-time pointer
/// and workspace checks before accessing caller-owned buffers.
pub unsafe extern "C" fn shapeit_hmm_run_segment_single_prevalidated_v1(
    parameters: *const HmmSegmentSingleV1,
    outcome: *mut i32,
) -> u32 {
    run_segment_single_v1_impl(parameters, outcome, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_full_kernel_model_allowlist_is_narrow() {
        assert_eq!(intel_family_model(0x0008_06f0), (6, 0x8f));
        assert_eq!(intel_family_model(0x000c_06f0), (6, 0xcf));
        assert!(intel_avx512_full_model_allowed(0x0008_06f0));
        assert!(intel_avx512_full_model_allowed(0x000c_06f0));
        assert!(!intel_avx512_full_model_allowed(0x0005_0650));
        assert!(!intel_avx512_full_model_allowed(0x0006_06a0));
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn assert_vector_sums_bitwise_equal(
        left: [__m256; HAPLOTYPES],
        right: [__m256; HAPLOTYPES],
    ) {
        let mut left_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let mut right_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        for vector in 0..HAPLOTYPES {
            _mm256_storeu_ps(
                left_lanes.as_mut_ptr().add(vector * HAPLOTYPES),
                left[vector],
            );
            _mm256_storeu_ps(
                right_lanes.as_mut_ptr().add(vector * HAPLOTYPES),
                right[vector],
            );
        }
        assert_eq!(left_lanes.map(f32::to_bits), right_lanes.map(f32::to_bits));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_full_blocks_are_bitwise_identical_to_avx2() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        const BLOCKS: usize = 5;
        let alleles = [0x00u8, 0xff, 0x96, 0x01, 0x80];
        let probability: Vec<f32> = (0..BLOCKS * HAPLOTYPES * HAPLOTYPES)
            .map(|index| (index as f32 + 1.0) * 0.000_031_25)
            .collect();
        let stay = unsafe { _mm256_set1_ps(0.9375) };
        let transferred = unsafe {
            _mm256_setr_ps(
                0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
            )
        };
        let mismatch_scalar = 0.000_100_01f32;
        let mismatch = unsafe { _mm256_set1_ps(mismatch_scalar) };

        for genotype_allele in [false, true] {
            let mut avx2_probability = probability.clone();
            let mut avx512_probability = probability.clone();
            let avx2_sums = unsafe {
                SingleEngine::run_full_hom_blocks_avx2(
                    avx2_probability.as_mut_ptr(),
                    alleles.as_ptr(),
                    BLOCKS,
                    stay,
                    transferred,
                    mismatch,
                    genotype_allele,
                )
            };
            let avx512_sums = unsafe {
                SingleEngine::run_full_hom_blocks_avx512(
                    avx512_probability.as_mut_ptr(),
                    alleles.as_ptr(),
                    BLOCKS,
                    stay,
                    transferred,
                    mismatch_scalar,
                    genotype_allele,
                )
            };
            assert_eq!(
                avx2_probability
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                avx512_probability
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            );
            unsafe { assert_vector_sums_bitwise_equal(avx2_sums, avx512_sums) };
        }

        let emission_zero =
            unsafe { _mm256_setr_ps(1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001) };
        let emission_one =
            unsafe { _mm256_setr_ps(0.0001, 1.0, 0.0001, 1.0, 0.0001, 1.0, 0.0001, 1.0) };
        let mut avx2_probability = probability.clone();
        let mut avx512_probability = probability;
        let avx2_sums = unsafe {
            SingleEngine::run_full_ambiguous_blocks_avx2(
                avx2_probability.as_mut_ptr(),
                alleles.as_ptr(),
                BLOCKS,
                stay,
                transferred,
                emission_zero,
                emission_one,
            )
        };
        let avx512_sums = unsafe {
            SingleEngine::run_full_ambiguous_blocks_avx512(
                avx512_probability.as_mut_ptr(),
                alleles.as_ptr(),
                BLOCKS,
                stay,
                transferred,
                0xaa,
                0.0001,
            )
        };
        assert_eq!(
            avx2_probability
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            avx512_probability
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        unsafe { assert_vector_sums_bitwise_equal(avx2_sums, avx512_sums) };
    }

    #[test]
    fn single_precision_initial_diplotypes_normalize_or_report_underflow() {
        let variants = [0u8];
        let lengths = [1u16];
        let diplotypes = [(1u64 << 0) | (1u64 << 9)];
        let haplotypes = [0b0101_0101u8; 8];
        let centimorgans = [0.0f32];
        let rare_alleles = [-1i8];
        let mut transitions = [0.0f64; 2];
        let mut parameters = HmmSegmentSingleV1 {
            abi_version: ABI_VERSION,
            struct_size: mem::size_of::<HmmSegmentSingleV1>() as u32,
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
            emission_match: 0.9999,
            emission_mismatch: 0.0001,
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
            scratch: ptr::null_mut(),
            scratch_length: 0,
            alpha_locus_scratch: ptr::null_mut(),
            alpha_locus_scratch_length: 0,
            index_scratch: ptr::null_mut(),
            index_scratch_length: 0,
        };
        let mut float_length = 0;
        let mut alpha_locus_length = 0;
        let mut index_length = 0;
        let status = unsafe {
            shapeit_hmm_single_scratch_len_v1(
                &parameters,
                &mut float_length,
                &mut alpha_locus_length,
                &mut index_length,
            )
        };
        assert_eq!(status, STATUS_OK);
        let mut scratch = vec![0.0f32; float_length];
        let mut alpha_locus = vec![0i32; alpha_locus_length];
        let mut indexes = vec![0usize; index_length];
        parameters.scratch = scratch.as_mut_ptr();
        parameters.scratch_length = scratch.len();
        parameters.alpha_locus_scratch = alpha_locus.as_mut_ptr();
        parameters.alpha_locus_scratch_length = alpha_locus.len();
        parameters.index_scratch = indexes.as_mut_ptr();
        parameters.index_scratch_length = indexes.len();

        let mut outcome = i32::MIN;
        let status = unsafe { shapeit_hmm_run_segment_single_v1(&parameters, &mut outcome) };
        assert_eq!(status, STATUS_OK);
        assert_eq!(outcome, 0);
        assert_eq!(transitions, [0.499_999_999_999_999_94; 2]);

        // Twenty ordinary 1e-4 mismatches drive the permitted haplotype lane
        // below f32 range while other dense lanes remain finite. This is the
        // initial-distribution underflow that previously escaped all guards.
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
        let mut zero_float_length = 0;
        let mut zero_alpha_locus_length = 0;
        let mut zero_index_length = 0;
        let zero_scratch_status = unsafe {
            shapeit_hmm_single_scratch_len_v1(
                &parameters,
                &mut zero_float_length,
                &mut zero_alpha_locus_length,
                &mut zero_index_length,
            )
        };
        assert_eq!(zero_scratch_status, STATUS_OK);
        scratch.resize(zero_float_length, 0.0);
        alpha_locus.resize(zero_alpha_locus_length, 0);
        indexes.resize(zero_index_length, 0);
        parameters.scratch = scratch.as_mut_ptr();
        parameters.scratch_length = scratch.len();
        parameters.alpha_locus_scratch = alpha_locus.as_mut_ptr();
        parameters.alpha_locus_scratch_length = alpha_locus.len();
        parameters.index_scratch = indexes.as_mut_ptr();
        parameters.index_scratch_length = indexes.len();
        let mut zero_support_outcome = i32::MIN;
        let zero_support_status =
            unsafe { shapeit_hmm_run_segment_single_v1(&parameters, &mut zero_support_outcome) };
        assert_eq!(zero_support_status, STATUS_OK);
        assert_eq!(zero_support_outcome, -2);
        assert_eq!(zero_support_transitions, [7.0]);
    }
}
