use super::{
    const_slice, inclusive_count, mut_slice, scratch_layout, variant_code, JobWindowInputs,
    HAPLOTYPES,
};

const LOG_ZERO: f64 = f64::NEG_INFINITY;

#[inline]
fn log_add(lhs: f64, rhs: f64) -> f64 {
    if lhs == LOG_ZERO {
        return rhs;
    }
    if rhs == LOG_ZERO {
        return lhs;
    }
    if lhs.is_nan() || rhs.is_nan() || lhs == f64::INFINITY || rhs == f64::INFINITY {
        return f64::NAN;
    }
    let maximum = lhs.max(rhs);
    maximum + (-(lhs - rhs).abs()).exp().ln_1p()
}

#[inline]
fn log_sum(values: &[f64]) -> f64 {
    values.iter().copied().fold(LOG_ZERO, log_add)
}

#[inline]
fn log_probability(probability: f64) -> f64 {
    if probability == 0.0 {
        LOG_ZERO
    } else if probability > 0.0 && probability.is_finite() {
        probability.ln()
    } else {
        f64::NAN
    }
}

#[inline]
fn valid_log_mass(value: f64) -> bool {
    value.is_finite()
}

fn write_normalized_logs(log_weights: &[f64], log_total: f64, target: &mut [f64]) -> bool {
    if log_weights.len() != target.len() || !valid_log_mass(log_total) {
        return true;
    }
    let mut represented_total = 0.0;
    for (&log_weight, probability) in log_weights.iter().zip(target.iter_mut()) {
        *probability = if log_weight == LOG_ZERO {
            0.0
        } else if log_weight.is_finite() {
            (log_weight - log_total).min(0.0).exp()
        } else {
            return true;
        };
        represented_total += *probability;
    }
    if !represented_total.is_finite() || represented_total <= 0.0 {
        return true;
    }
    let reciprocal = 1.0 / represented_total;
    for probability in target {
        *probability *= reciprocal;
    }
    false
}

struct LogEngine<'a> {
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
    log_mismatch: f64,

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

impl LogEngine<'_> {
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
        -(unsafe { super::expm1f(argument) } as f64)
    }

    #[inline]
    fn transition_logs(transition: f64) -> (f64, f64) {
        (
            log_probability(transition),
            log_probability(1.0 - transition),
        )
    }

    #[inline]
    fn update_total(&mut self, sums: [f64; HAPLOTYPES]) {
        self.prob_sum_h = sums;
        self.prob_sum_t = log_sum(&self.prob_sum_h);
    }

    fn init_hom(&mut self, locus: usize, relative_locus: usize) {
        let genotype_allele = self.hap0(locus);
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let emission = if genotype_allele != self.allele(relative_locus, k) {
                self.log_mismatch
            } else {
                0.0
            };
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                self.prob[start + h] = emission;
                sums[h] = log_add(sums[h], emission);
            }
        }
        self.update_total(sums);
    }

    fn init_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize) {
        let code = self.ambiguous[ambiguous_index];
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let hap = ((code >> h) & 1) != 0;
                let emission = if hap != allele {
                    self.log_mismatch
                } else {
                    0.0
                };
                self.prob[start + h] = emission;
                sums[h] = log_add(sums[h], emission);
            }
        }
        self.update_total(sums);
    }

    fn init_missing(&mut self) {
        let log_probability = -((HAPLOTYPES * self.conditioning_haplotypes) as f64).ln();
        self.prob.fill(log_probability);
        self.prob_sum_h.fill(-(HAPLOTYPES as f64).ln());
        self.prob_sum_t = 0.0;
    }

    fn run_hom(&mut self, locus: usize, relative_locus: usize, transition: f64) -> bool {
        let genotype_allele = self.hap0(locus);
        let rare_allele = self.rare_alleles[locus];
        if rare_allele >= 0 && genotype_allele != (rare_allele != 0) {
            return false;
        }
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_factor =
            log_transition - (self.conditioning_haplotypes as f64).ln() - self.prob_sum_t;
        let log_stay_factor = log_stay - self.prob_sum_t;
        let mut transferred = [LOG_ZERO; HAPLOTYPES];
        for h in 0..HAPLOTYPES {
            transferred[h] = self.prob_sum_h[h] + log_factor;
        }
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let emission = if genotype_allele != self.allele(relative_locus, k) {
                self.log_mismatch
            } else {
                0.0
            };
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let value =
                    log_add(self.prob[start + h] + log_stay_factor, transferred[h]) + emission;
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
            }
        }
        self.update_total(sums);
        true
    }

    fn run_ambiguous(&mut self, relative_locus: usize, ambiguous_index: usize, transition: f64) {
        let code = self.ambiguous[ambiguous_index];
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_factor =
            log_transition - (self.conditioning_haplotypes as f64).ln() - self.prob_sum_t;
        let log_stay_factor = log_stay - self.prob_sum_t;
        let mut transferred = [LOG_ZERO; HAPLOTYPES];
        for h in 0..HAPLOTYPES {
            transferred[h] = self.prob_sum_h[h] + log_factor;
        }
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let hap = ((code >> h) & 1) != 0;
                let emission = if hap != allele {
                    self.log_mismatch
                } else {
                    0.0
                };
                let value =
                    log_add(self.prob[start + h] + log_stay_factor, transferred[h]) + emission;
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
            }
        }
        self.update_total(sums);
    }

    fn run_missing(&mut self, transition: f64) {
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_factor =
            log_transition - (self.conditioning_haplotypes as f64).ln() - self.prob_sum_t;
        let log_stay_factor = log_stay - self.prob_sum_t;
        let mut transferred = [LOG_ZERO; HAPLOTYPES];
        for h in 0..HAPLOTYPES {
            transferred[h] = self.prob_sum_h[h] + log_factor;
        }
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let value = log_add(self.prob[start + h] + log_stay_factor, transferred[h]);
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
            }
        }
        self.update_total(sums);
    }

    fn collapse_hom(&mut self, locus: usize, relative_locus: usize, transition: f64) {
        let genotype_allele = self.hap0(locus);
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_stay_factor = log_stay - self.prob_sum_t;
        let log_transferred = log_transition - (self.conditioning_haplotypes as f64).ln();
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let emission = if genotype_allele != self.allele(relative_locus, k) {
                self.log_mismatch
            } else {
                0.0
            };
            let value = log_add(self.prob_sum_k[k] + log_stay_factor, log_transferred) + emission;
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
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
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_stay_factor = log_stay - self.prob_sum_t;
        let log_transferred = log_transition - (self.conditioning_haplotypes as f64).ln();
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let allele = self.allele(relative_locus, k);
            let base = log_add(self.prob_sum_k[k] + log_stay_factor, log_transferred);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                let hap = ((code >> h) & 1) != 0;
                let emission = if hap != allele {
                    self.log_mismatch
                } else {
                    0.0
                };
                let value = base + emission;
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
            }
        }
        self.update_total(sums);
    }

    fn collapse_missing(&mut self, transition: f64) {
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_stay_factor = log_stay - self.prob_sum_t;
        let log_transferred = log_transition - (self.conditioning_haplotypes as f64).ln();
        let mut sums = [LOG_ZERO; HAPLOTYPES];
        for k in 0..self.conditioning_haplotypes {
            let value = log_add(self.prob_sum_k[k] + log_stay_factor, log_transferred);
            let start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                self.prob[start + h] = value;
                sums[h] = log_add(sums[h], value);
            }
        }
        self.update_total(sums);
    }

    fn sum_conditioning_haplotypes(&mut self) {
        for k in 0..self.conditioning_haplotypes {
            let start = k * HAPLOTYPES;
            self.prob_sum_k[k] = log_sum(&self.prob[start..start + HAPLOTYPES]);
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

impl LogEngine<'_> {
    fn transition_haplotypes(&mut self, relative_segment: usize, previous_locus: usize) -> bool {
        let alpha_segment = relative_segment - 1;
        let alpha_sum_total = self.alpha_sum_sum[alpha_segment];
        if !valid_log_mass(alpha_sum_total) {
            return true;
        }
        let alpha_locus = self.alpha_locus[alpha_segment] as usize;
        let transition = self.transition_probability(alpha_locus, previous_locus);
        let (log_transition, log_stay) = Self::transition_logs(transition);
        let log_stay_factor = log_stay - alpha_sum_total;
        let alpha_state_start = alpha_segment * self.prob.len();
        let alpha_sum_start = alpha_segment * HAPLOTYPES;
        let mut total = LOG_ZERO;

        for h1 in 0..HAPLOTYPES {
            let transferred = self.alpha_sum[alpha_sum_start + h1] - alpha_sum_total
                + log_transition
                - (self.conditioning_haplotypes as f64).ln();
            let row_start = h1 * HAPLOTYPES;
            let mut sums = [LOG_ZERO; HAPLOTYPES];
            for k in 0..self.conditioning_haplotypes {
                let state_start = k * HAPLOTYPES;
                let alpha = log_add(
                    self.alpha[alpha_state_start + state_start + h1] + log_stay_factor,
                    transferred,
                );
                for h2 in 0..HAPLOTYPES {
                    sums[h2] = log_add(sums[h2], alpha + self.prob[state_start + h2]);
                }
            }
            self.h_probs[row_start..row_start + HAPLOTYPES].copy_from_slice(&sums);
            total = log_add(total, log_sum(&sums));
        }
        self.sum_h_probs = total;
        !valid_log_mass(total)
    }

    fn transition_diplotypes(&mut self, previous: u64, current: u64) -> bool {
        if !valid_log_mass(self.sum_h_probs) {
            return true;
        }
        let log_scaling = -self.sum_h_probs;
        let mut total = LOG_ZERO;
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
                    + log_scaling;
                let second = self.h_probs
                    [(previous_diplotype & 7) * HAPLOTYPES + (current_diplotype & 7)]
                    + log_scaling;
                let value = first + second;
                self.d_probs[index] = value;
                total = log_add(total, value);
                index += 1;
            }
        }
        self.sum_d_probs = total;
        !valid_log_mass(total)
    }

    fn set_first_transitions(&mut self) -> bool {
        if !valid_log_mass(self.prob_sum_t) {
            return true;
        }
        let mut probabilities = [LOG_ZERO; 64];
        let mut total = LOG_ZERO;
        let mut count = 0usize;
        let mut active = self.diplotypes[0];
        while active != 0 {
            let diplotype = active.trailing_zeros() as usize;
            active &= active - 1;
            let value = self.prob_sum_h[diplotype >> 3] - self.prob_sum_t
                + self.prob_sum_h[diplotype & 7]
                - self.prob_sum_t;
            probabilities[count] = value;
            total = log_add(total, value);
            count += 1;
        }
        write_normalized_logs(
            &probabilities[..count],
            total,
            &mut self.transition_probabilities[..count],
        )
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
        if self.transition_diplotypes(previous, current) {
            return -2;
        }

        let transition_count = previous.count_ones() as usize * current.count_ones() as usize;
        *transition_cursor -= transition_count as isize - 1;
        let start = *transition_cursor as usize;
        if write_normalized_logs(
            &self.d_probs[..transition_count],
            self.sum_d_probs,
            &mut self.transition_probabilities[start..start + transition_count],
        ) {
            return -2;
        }
        *transition_cursor -= 1;
        0
    }

    fn impute(
        &mut self,
        relative_locus: usize,
        relative_missing: usize,
        absolute_missing: usize,
    ) -> bool {
        let state_start = relative_missing * self.prob.len();
        let sum_start = relative_missing * HAPLOTYPES;
        let mut sums = [[LOG_ZERO; HAPLOTYPES]; 2];
        for k in 0..self.conditioning_haplotypes {
            let allele = usize::from(self.allele(relative_locus, k));
            let conditioning_start = k * HAPLOTYPES;
            for h in 0..HAPLOTYPES {
                // Preserve the established AVX denominator mapping for lanes 4..7.
                let denominator_index = if h < 4 { h } else { h - 3 };
                let denominator = self.alpha_sum_missing[sum_start + denominator_index];
                let alpha = self.alpha_missing[state_start + conditioning_start + h] - denominator;
                sums[allele][h] =
                    log_add(sums[allele][h], alpha + self.prob[conditioning_start + h]);
            }
        }
        let output_start = absolute_missing * HAPLOTYPES;
        for h in 0..HAPLOTYPES {
            let total = log_add(sums[0][h], sums[1][h]);
            if !valid_log_mass(total) {
                return true;
            }
            let probability = if sums[1][h] == LOG_ZERO {
                0.0
            } else {
                (sums[1][h] - total).min(0.0).exp()
            };
            if !probability.is_finite() {
                return true;
            }
            self.missing_probabilities[output_start + h] = probability as f32;
        }
        false
    }

    fn forward(&mut self) -> bool {
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
            if !valid_log_mass(self.prob_sum_t) {
                return true;
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
        false
    }

    fn backward(&mut self) -> i32 {
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
            if !valid_log_mass(self.prob_sum_t) {
                return -2;
            }
            if segment_locus == 0 {
                self.sum_conditioning_haplotypes();
            }
            if update_previous {
                previous_locus = locus;
            }

            if locus == 0 && self.set_first_transitions() {
                return -2;
            }
            if segment_locus == 0 && locus != self.locus_first {
                let result =
                    self.set_other_transitions(segment, previous_locus, &mut transition_cursor);
                if result < 0 {
                    return result;
                }
            }
            if missing {
                let relative_missing = (missing_index - self.missing_first as isize) as usize;
                if self.impute(relative_locus, relative_missing, missing_index as usize) {
                    return -2;
                }
                missing_index -= 1;
            }

            segment_locus -= 1;
            ambiguous_index -= isize::from(ambiguous);
            if segment_locus < 0 && segment > 0 {
                segment -= 1;
                segment_locus = self.segment_lengths[segment] as isize - 1;
            }
        }
        0
    }

    fn run(&mut self) -> i32 {
        if self.forward() {
            -2
        } else {
            self.backward()
        }
    }
}

pub(super) unsafe fn run_job_window_log(
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
    scratch.resize(layout.total, LOG_ZERO);
    scratch.fill(LOG_ZERO);
    alpha_locus.resize(segment_count, 0);
    alpha_locus.fill(0);

    let (prob, scratch) = scratch.split_at_mut(layout.states);
    let (prob_sum_k, scratch) = scratch.split_at_mut(inputs.conditioning_haplotypes);
    let alpha_length = layout.segment_count * layout.states;
    let (alpha, scratch) = scratch.split_at_mut(alpha_length);
    let alpha_sum_length = layout.segment_count * HAPLOTYPES;
    let (alpha_sum, scratch) = scratch.split_at_mut(alpha_sum_length);
    let (alpha_sum_sum, scratch) = scratch.split_at_mut(layout.segment_count);
    let alpha_missing_length = layout.missing_count * layout.states;
    let (alpha_missing, scratch) = scratch.split_at_mut(alpha_missing_length);
    let alpha_sum_missing_length = layout.missing_count * HAPLOTYPES;
    let (alpha_sum_missing, remainder) = scratch.split_at_mut(alpha_sum_missing_length);
    debug_assert!(remainder.is_empty());

    let centimorgans = const_slice(inputs.job.centimorgans, inputs.job.centimorgans_length);
    let recombination = const_slice(inputs.job.recombination, inputs.job.recombination_length);
    let rare_alleles = const_slice(inputs.job.rare_alleles, inputs.job.rare_alleles_length);
    let transition_probabilities = mut_slice(
        inputs.job.transition_probabilities,
        inputs.job.transition_probabilities_length,
    );
    let missing_probabilities = mut_slice(
        inputs.job.missing_probabilities,
        inputs.job.missing_probabilities_length,
    );

    let mut engine = LogEngine {
        variants: inputs.variants,
        ambiguous: inputs.ambiguous,
        segment_lengths: inputs.segment_lengths,
        diplotypes: inputs.diplotypes,
        haplotypes: inputs.subset_haplotypes,
        haplotype_stride: inputs.subset_stride,
        conditioning_haplotypes: inputs.conditioning_haplotypes,
        locus_offset: inputs.locus_offset as usize,
        centimorgans,
        recombination,
        rare_alleles,
        effective_population_size: inputs.job.effective_population_size,
        total_haplotypes: inputs.job.total_haplotypes,
        log_mismatch: log_probability(inputs.job.emission_mismatch / inputs.job.emission_match),
        segment_first: inputs.window.start_segment as usize,
        segment_last: inputs.window.stop_segment as usize,
        locus_first: inputs.window.start_locus as usize,
        locus_last: inputs.window.stop_locus as usize,
        ambiguous_first: inputs.window.start_ambiguous as usize,
        missing_first: inputs.window.start_missing as usize,
        transition_last: inputs.window.stop_transition as usize,
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
        prob_sum_t: LOG_ZERO,
        prob_sum_h: [LOG_ZERO; HAPLOTYPES],
        sum_h_probs: LOG_ZERO,
        sum_d_probs: LOG_ZERO,
        h_probs: [LOG_ZERO; HAPLOTYPES * HAPLOTYPES],
        d_probs: [LOG_ZERO; HAPLOTYPES * HAPLOTYPES * HAPLOTYPES * HAPLOTYPES],
    };
    Ok(engine.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_add_retains_extreme_relative_mass() {
        let total = log_add(0.0, -1000.0);
        assert_eq!(total, 0.0);
        assert_eq!(log_add(LOG_ZERO, -1000.0), -1000.0);
        assert!(log_add(f64::NAN, 0.0).is_nan());
    }
}
