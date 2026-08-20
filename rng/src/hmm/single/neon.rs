//! AArch64 NEON kernels for the normal single-precision common HMM.
//!
//! Kept separate from `single.rs` so the public ABI remains easy to audit and
//! the first-class x86 implementation stays undisturbed.
use core::arch::aarch64::{
    float32x4_t, int32x4_t, uint32x4_t, vaddq_f32, vandq_u32, vbslq_f32, vceqq_u32, vdupq_n_f32,
    vdupq_n_s32, vdupq_n_u32, veorq_u32, vfmaq_f32, vld1q_f32, vld1q_s32, vld1q_u32, vmulq_f32,
    vshlq_u32, vst1q_f32,
};

use super::HAPLOTYPES;

const MAX_COMPRESSED_QUADS: usize = 4;
const GRAPH_SHIFTS_LOW: [i32; 4] = [0, -1, -2, -3];
const GRAPH_SHIFTS_HIGH: [i32; 4] = [-4, -5, -6, -7];

#[inline(always)]
unsafe fn emission(
    packed: uint32x4_t,
    shifts: int32x4_t,
    graph: uint32x4_t,
    one_bits: uint32x4_t,
    ones: float32x4_t,
    mismatch: float32x4_t,
) -> float32x4_t {
    let allele = vandq_u32(vshlq_u32(packed, shifts), one_bits);
    let differs = veorq_u32(allele, graph);
    vbslq_f32(vceqq_u32(differs, one_bits), mismatch, ones)
}

#[inline(always)]
unsafe fn graph_mask<const AMBIGUOUS: bool>(
    ambiguous_code: u8,
    genotype_allele: bool,
    shifts: *const i32,
) -> uint32x4_t {
    if AMBIGUOUS {
        let packed = vdupq_n_u32(u32::from(ambiguous_code));
        let bits = vandq_u32(vshlq_u32(packed, vld1q_s32(shifts)), vdupq_n_u32(1));
        vceqq_u32(bits, vdupq_n_u32(1))
    } else {
        vdupq_n_u32(if genotype_allele { u32::MAX } else { 0 })
    }
}

/// Advance a one- or two-lane state matrix. The accumulator
/// layout deliberately matches the two 128-bit halves of the AVX2 kernel:
/// lane `k % 8` accumulates complete blocks and lane zero receives tails.
pub(super) unsafe fn run_compressed<const N: usize, const AMBIGUOUS: bool>(
    probability: *mut f32,
    conditioning_haplotypes: usize,
    prob_sum_h: *const f32,
    allele_bytes: *const u8,
    factor: f32,
    stay_factor: f32,
    mismatch_scalar: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    debug_assert!(matches!(N, 1 | 2));

    let zero = vdupq_n_f32(0.0);
    let mut sums = [zero; MAX_COMPRESSED_QUADS];
    let mut shifts = [vdupq_n_s32(0); MAX_COMPRESSED_QUADS];
    let mut graph = [vdupq_n_u32(0); MAX_COMPRESSED_QUADS];
    let mut transferred = [zero; MAX_COMPRESSED_QUADS];
    let quad_count = 2 * N;

    // Each complete block contains eight conditioning haplotypes and N
    // graph lanes per conditioning haplotype, stored conditioning-major.
    for quad in 0..quad_count {
        let mut shift_lanes = [0i32; 4];
        let mut graph_lanes = [0u32; 4];
        let mut transferred_lanes = [0.0f32; 4];
        for lane in 0..4 {
            let state = quad * 4 + lane;
            let k = state / N;
            let h = state % N;
            shift_lanes[lane] = k as i32 - 7;
            graph_lanes[lane] = u32::from(if AMBIGUOUS {
                ((ambiguous_code >> h) & 1) != 0
            } else {
                genotype_allele
            });
            transferred_lanes[lane] = *prob_sum_h.add(h) * factor;
        }
        shifts[quad] = vld1q_s32(shift_lanes.as_ptr());
        graph[quad] = vld1q_u32(graph_lanes.as_ptr());
        transferred[quad] = vld1q_f32(transferred_lanes.as_ptr());
    }

    let stay = vdupq_n_f32(stay_factor);
    let ones = vdupq_n_f32(1.0);
    let mismatch = vdupq_n_f32(mismatch_scalar);
    let one_bits = vdupq_n_u32(1);
    let block_count = conditioning_haplotypes / HAPLOTYPES;
    for block in 0..block_count {
        let packed = vdupq_n_u32(u32::from(*allele_bytes.add(block)));
        let block_probability = probability.add(block * HAPLOTYPES * N);
        for quad in 0..quad_count {
            let value = vfmaq_f32(
                transferred[quad],
                vld1q_f32(block_probability.add(quad * 4)),
                stay,
            );
            let value = vmulq_f32(
                value,
                emission(packed, shifts[quad], graph[quad], one_bits, ones, mismatch),
            );
            sums[quad] = vaddq_f32(sums[quad], value);
            vst1q_f32(block_probability.add(quad * 4), value);
        }
    }

    let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
    for quad in 0..quad_count {
        vst1q_f32(sum_lanes.as_mut_ptr().add(quad * 4), sums[quad]);
    }

    // Match the AVX2 tail path: all remaining conditioning haplotypes are
    // accumulated into the first logical vector before the fixed tree.
    let mut k = block_count * HAPLOTYPES;
    let mut probability_index = k * N;
    while k < conditioning_haplotypes {
        let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
        for h in 0..N {
            let graph_haplotype = if AMBIGUOUS {
                ((ambiguous_code >> h) & 1) != 0
            } else {
                genotype_allele
            };
            let emission = if graph_haplotype != conditioning_allele {
                mismatch_scalar
            } else {
                1.0
            };
            let value = (*probability.add(probability_index + h))
                .mul_add(stay_factor, *prob_sum_h.add(h) * factor)
                * emission;
            *probability.add(probability_index + h) = value;
            sum_lanes[h] += value;
        }
        probability_index += N;
        k += 1;
    }

    let mut unique_sums = [0.0f32; HAPLOTYPES];
    for h in 0..N {
        let sum01 = sum_lanes[h] + sum_lanes[N + h];
        let sum23 = sum_lanes[2 * N + h] + sum_lanes[3 * N + h];
        let sum45 = sum_lanes[4 * N + h] + sum_lanes[5 * N + h];
        let sum67 = sum_lanes[6 * N + h] + sum_lanes[7 * N + h];
        unique_sums[h] = (sum01 + sum23) + (sum45 + sum67);
    }
    unique_sums
}

/// Four graph lanes fit in one NEON register, so each conditioning allele
/// selects one of two prebuilt emission vectors. This avoids the packed-bit
/// vector decode that is profitable only for N=1 and N=2.
pub(super) unsafe fn run_four<const AMBIGUOUS: bool>(
    probability: *mut f32,
    conditioning_haplotypes: usize,
    prob_sum_h: *const f32,
    allele_bytes: *const u8,
    factor: f32,
    stay_factor: f32,
    mismatch_scalar: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    let transferred = vmulq_f32(vld1q_f32(prob_sum_h), vdupq_n_f32(factor));
    let graph = graph_mask::<AMBIGUOUS>(ambiguous_code, genotype_allele, GRAPH_SHIFTS_LOW.as_ptr());
    let ones = vdupq_n_f32(1.0);
    let mismatch = vdupq_n_f32(mismatch_scalar);
    let emission_zero = vbslq_f32(graph, mismatch, ones);
    let emission_one = vbslq_f32(graph, ones, mismatch);
    let stay = vdupq_n_f32(stay_factor);
    let zero = vdupq_n_f32(0.0);
    let mut vector_sums = [zero; HAPLOTYPES];

    let block_count = conditioning_haplotypes / HAPLOTYPES;
    for block in 0..block_count {
        let packed = *allele_bytes.add(block);
        for lane in 0..HAPLOTYPES {
            let k = block * HAPLOTYPES + lane;
            let conditioning_allele = ((packed >> (7 - lane)) & 1) != 0;
            let state = probability.add(k * 4);
            let value = vfmaq_f32(transferred, vld1q_f32(state), stay);
            let value = vmulq_f32(
                value,
                if conditioning_allele {
                    emission_one
                } else {
                    emission_zero
                },
            );
            vector_sums[lane] = vaddq_f32(vector_sums[lane], value);
            vst1q_f32(state, value);
        }
    }

    let mut k = block_count * HAPLOTYPES;
    while k < conditioning_haplotypes {
        let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
        let state = probability.add(k * 4);
        let value = vfmaq_f32(transferred, vld1q_f32(state), stay);
        let value = vmulq_f32(
            value,
            if conditioning_allele {
                emission_one
            } else {
                emission_zero
            },
        );
        vector_sums[0] = vaddq_f32(vector_sums[0], value);
        vst1q_f32(state, value);
        k += 1;
    }

    let mut sums = [0.0f32; HAPLOTYPES];
    vst1q_f32(sums.as_mut_ptr(), reduce_eight(&vector_sums));
    sums
}

/// The full-width path uses paired NEON registers. Its sixteen running
/// sums remain independent, matching the two halves of each AVX2 lane and
/// retaining the same fixed eight-way reduction tree.
pub(super) unsafe fn run_full<const AMBIGUOUS: bool>(
    probability: *mut f32,
    conditioning_haplotypes: usize,
    prob_sum_h: *const f32,
    allele_bytes: *const u8,
    factor: f32,
    stay_factor: f32,
    mismatch_scalar: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    let factor4 = vdupq_n_f32(factor);
    let transferred_low = vmulq_f32(vld1q_f32(prob_sum_h), factor4);
    let transferred_high = vmulq_f32(vld1q_f32(prob_sum_h.add(4)), factor4);
    let graph_low =
        graph_mask::<AMBIGUOUS>(ambiguous_code, genotype_allele, GRAPH_SHIFTS_LOW.as_ptr());
    let graph_high =
        graph_mask::<AMBIGUOUS>(ambiguous_code, genotype_allele, GRAPH_SHIFTS_HIGH.as_ptr());
    let ones = vdupq_n_f32(1.0);
    let mismatch = vdupq_n_f32(mismatch_scalar);
    let emission_zero_low = vbslq_f32(graph_low, mismatch, ones);
    let emission_zero_high = vbslq_f32(graph_high, mismatch, ones);
    let emission_one_low = vbslq_f32(graph_low, ones, mismatch);
    let emission_one_high = vbslq_f32(graph_high, ones, mismatch);
    let stay = vdupq_n_f32(stay_factor);
    let zero = vdupq_n_f32(0.0);
    let mut sum_low = [zero; HAPLOTYPES];
    let mut sum_high = [zero; HAPLOTYPES];

    let block_count = conditioning_haplotypes / HAPLOTYPES;
    for block in 0..block_count {
        let packed = *allele_bytes.add(block);
        for lane in 0..HAPLOTYPES {
            let k = block * HAPLOTYPES + lane;
            let conditioning_allele = ((packed >> (7 - lane)) & 1) != 0;
            let state = probability.add(k * HAPLOTYPES);
            let low = vfmaq_f32(transferred_low, vld1q_f32(state), stay);
            let high = vfmaq_f32(transferred_high, vld1q_f32(state.add(4)), stay);
            let low = vmulq_f32(
                low,
                if conditioning_allele {
                    emission_one_low
                } else {
                    emission_zero_low
                },
            );
            let high = vmulq_f32(
                high,
                if conditioning_allele {
                    emission_one_high
                } else {
                    emission_zero_high
                },
            );
            sum_low[lane] = vaddq_f32(sum_low[lane], low);
            sum_high[lane] = vaddq_f32(sum_high[lane], high);
            vst1q_f32(state, low);
            vst1q_f32(state.add(4), high);
        }
    }

    let mut k = block_count * HAPLOTYPES;
    while k < conditioning_haplotypes {
        let conditioning_allele = ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0;
        let state = probability.add(k * HAPLOTYPES);
        let low = vfmaq_f32(transferred_low, vld1q_f32(state), stay);
        let high = vfmaq_f32(transferred_high, vld1q_f32(state.add(4)), stay);
        let low = vmulq_f32(
            low,
            if conditioning_allele {
                emission_one_low
            } else {
                emission_zero_low
            },
        );
        let high = vmulq_f32(
            high,
            if conditioning_allele {
                emission_one_high
            } else {
                emission_zero_high
            },
        );
        sum_low[0] = vaddq_f32(sum_low[0], low);
        sum_high[0] = vaddq_f32(sum_high[0], high);
        vst1q_f32(state, low);
        vst1q_f32(state.add(4), high);
        k += 1;
    }

    let mut sums = [0.0f32; HAPLOTYPES];
    vst1q_f32(sums.as_mut_ptr(), reduce_eight(&sum_low));
    vst1q_f32(sums.as_mut_ptr().add(4), reduce_eight(&sum_high));
    sums
}

#[inline(always)]
unsafe fn reduce_eight(values: &[float32x4_t; HAPLOTYPES]) -> float32x4_t {
    let sum01 = vaddq_f32(values[0], values[1]);
    let sum23 = vaddq_f32(values[2], values[3]);
    let sum45 = vaddq_f32(values[4], values[5]);
    let sum67 = vaddq_f32(values[6], values[7]);
    vaddq_f32(vaddq_f32(sum01, sum23), vaddq_f32(sum45, sum67))
}

/// Expand conditioning-haplotype sums back to eight graph lanes at a
/// segment boundary. KIND 0 is homozygous, 1 ambiguous, and 2 missing.
pub(super) unsafe fn collapse<const KIND: u8>(
    probability: *mut f32,
    prob_sum_k: *const f32,
    conditioning_haplotypes: usize,
    allele_bytes: *const u8,
    transferred_scalar: f32,
    stay_factor: f32,
    mismatch_scalar: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    debug_assert!(KIND <= 2);
    let zero = vdupq_n_f32(0.0);
    let mut sum_low = [zero; HAPLOTYPES];
    let mut sum_high = [zero; HAPLOTYPES];
    let stay = vdupq_n_f32(stay_factor);
    let transferred = vdupq_n_f32(transferred_scalar);
    let ones = vdupq_n_f32(1.0);
    let mismatch = vdupq_n_f32(mismatch_scalar);

    let graph_low = graph_mask::<true>(ambiguous_code, false, GRAPH_SHIFTS_LOW.as_ptr());
    let graph_high = graph_mask::<true>(ambiguous_code, false, GRAPH_SHIFTS_HIGH.as_ptr());
    let emission_zero_low = vbslq_f32(graph_low, mismatch, ones);
    let emission_zero_high = vbslq_f32(graph_high, mismatch, ones);
    let emission_one_low = vbslq_f32(graph_low, ones, mismatch);
    let emission_one_high = vbslq_f32(graph_high, ones, mismatch);

    let block_count = conditioning_haplotypes / HAPLOTYPES;
    for block in 0..block_count {
        // KIND 2 deliberately avoids touching allele_bytes, which may be
        // dangling for a missing-only caller.
        let packed = if KIND < 2 {
            *allele_bytes.add(block)
        } else {
            0
        };
        for lane in 0..HAPLOTYPES {
            let k = block * HAPLOTYPES + lane;
            let conditioning_allele = ((packed >> (7 - lane)) & 1) != 0;
            let base = vfmaq_f32(transferred, vdupq_n_f32(*prob_sum_k.add(k)), stay);
            let (mut low, mut high) = (base, base);
            if KIND == 0 {
                let emission = if genotype_allele != conditioning_allele {
                    mismatch
                } else {
                    ones
                };
                low = vmulq_f32(low, emission);
                high = vmulq_f32(high, emission);
            } else if KIND == 1 {
                low = vmulq_f32(
                    low,
                    if conditioning_allele {
                        emission_one_low
                    } else {
                        emission_zero_low
                    },
                );
                high = vmulq_f32(
                    high,
                    if conditioning_allele {
                        emission_one_high
                    } else {
                        emission_zero_high
                    },
                );
            }
            sum_low[lane] = vaddq_f32(sum_low[lane], low);
            sum_high[lane] = vaddq_f32(sum_high[lane], high);
            let output = probability.add(k * HAPLOTYPES);
            vst1q_f32(output, low);
            vst1q_f32(output.add(4), high);
        }
    }

    let mut k = block_count * HAPLOTYPES;
    while k < conditioning_haplotypes {
        let conditioning_allele = if KIND < 2 {
            ((*allele_bytes.add(k >> 3) >> (7 - (k & 7))) & 1) != 0
        } else {
            false
        };
        let base = vfmaq_f32(transferred, vdupq_n_f32(*prob_sum_k.add(k)), stay);
        let (mut low, mut high) = (base, base);
        if KIND == 0 {
            let emission = if genotype_allele != conditioning_allele {
                mismatch
            } else {
                ones
            };
            low = vmulq_f32(low, emission);
            high = vmulq_f32(high, emission);
        } else if KIND == 1 {
            low = vmulq_f32(
                low,
                if conditioning_allele {
                    emission_one_low
                } else {
                    emission_zero_low
                },
            );
            high = vmulq_f32(
                high,
                if conditioning_allele {
                    emission_one_high
                } else {
                    emission_zero_high
                },
            );
        }
        sum_low[0] = vaddq_f32(sum_low[0], low);
        sum_high[0] = vaddq_f32(sum_high[0], high);
        let output = probability.add(k * HAPLOTYPES);
        vst1q_f32(output, low);
        vst1q_f32(output.add(4), high);
        k += 1;
    }

    let mut sums = [0.0f32; HAPLOTYPES];
    vst1q_f32(sums.as_mut_ptr(), reduce_eight(&sum_low));
    vst1q_f32(sums.as_mut_ptr().add(4), reduce_eight(&sum_high));
    sums
}

/// Contract two full, conditioning-major state matrices into the 8x8
/// row-major haplotype transition matrix.
pub(super) unsafe fn transition_full(
    alpha: *const f32,
    beta: *const f32,
    alpha_sum: *const f32,
    conditioning_haplotypes: usize,
    alpha_total: f32,
    transition: f32,
    stay_factor: f32,
    output: *mut f32,
) -> f32 {
    let zero = vdupq_n_f32(0.0);
    let mut sum_low = [zero; HAPLOTYPES];
    let mut sum_high = [zero; HAPLOTYPES];
    let mut transferred = [0.0f32; HAPLOTYPES];
    for h1 in 0..HAPLOTYPES {
        transferred[h1] =
            (*alpha_sum.add(h1) / alpha_total) * transition / conditioning_haplotypes as f32;
    }

    for k in 0..conditioning_haplotypes {
        let beta_row = beta.add(k * HAPLOTYPES);
        let beta_low = vld1q_f32(beta_row);
        let beta_high = vld1q_f32(beta_row.add(4));
        for h1 in 0..HAPLOTYPES {
            let alpha_value =
                (*alpha.add(k * HAPLOTYPES + h1)).mul_add(stay_factor, transferred[h1]);
            let alpha4 = vdupq_n_f32(alpha_value);
            sum_low[h1] = vfmaq_f32(sum_low[h1], alpha4, beta_low);
            sum_high[h1] = vfmaq_f32(sum_high[h1], alpha4, beta_high);
        }
    }

    let mut total = 0.0f32;
    for h1 in 0..HAPLOTYPES {
        let row = output.add(h1 * HAPLOTYPES);
        vst1q_f32(row, sum_low[h1]);
        vst1q_f32(row.add(4), sum_high[h1]);
        for h2 in 0..HAPLOTYPES {
            total += *row.add(h2);
        }
    }
    total
}

#[cfg(test)]
mod tests;
