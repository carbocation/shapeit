//! Scalar-oracle differential tests for the AArch64 single-precision kernels.

use super::*;
use core::ptr;

const NEON_TEST_K: [usize; 10] = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17];

const NEON_TEST_ALLELES: [u8; 5] = [0x00, 0xff, 0x80, 0x01, 0x96];

fn assert_f32_bits_eq(left: &[f32], right: &[f32], context: &str) {
    assert_eq!(left.len(), right.len(), "{context}: length");
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{context}: float {index}: left={left:?}, right={right:?}"
        );
    }
}

fn reduced_scalar_oracle<const N: usize, const AMBIGUOUS: bool>(
    probability: &mut [f32],
    conditioning_haplotypes: usize,
    prob_sum_h: &[f32; HAPLOTYPES],
    allele_bytes: &[u8],
    factor: f32,
    stay_factor: f32,
    mismatch: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
    let complete = conditioning_haplotypes / HAPLOTYPES * HAPLOTYPES;
    for k in 0..conditioning_haplotypes {
        let conditioning_allele = ((allele_bytes[k >> 3] >> (7 - (k & 7))) & 1) != 0;
        for h in 0..N {
            let graph_haplotype = if AMBIGUOUS {
                ((ambiguous_code >> h) & 1) != 0
            } else {
                genotype_allele
            };
            let emission = if graph_haplotype != conditioning_allele {
                mismatch
            } else {
                1.0
            };
            let state = k * N + h;
            let value = probability[state].mul_add(stay_factor, prob_sum_h[h] * factor) * emission;
            probability[state] = value;
            let sum_index = if k < complete { (k & 7) * N + h } else { h };
            sum_lanes[sum_index] += value;
        }
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

fn assert_reduced_case<const N: usize, const AMBIGUOUS: bool>(
    conditioning_haplotypes: usize,
    packed_allele: u8,
    genotype_allele: bool,
    ambiguous_code: u8,
) {
    const SENTINEL: f32 = -12_345.25;
    let active = conditioning_haplotypes * N;
    let initial: Vec<f32> = (0..active)
        .map(|index| ((index * 37 + 11) % 251 + 1) as f32 / 997.0)
        .collect();
    let mut expected = initial.clone();
    expected.extend([SENTINEL; 7]);
    let mut actual = expected.clone();
    let allele_bytes = vec![packed_allele; conditioning_haplotypes.div_ceil(8)];
    let prob_sum_h = [0.113f32, 0.071, 0.193, 0.157, 0.089, 0.131, 0.173, 0.103];
    let factor = 0.003_718_75f32;
    let stay_factor = 0.917_187_5f32;
    let mismatch = 0.000_100_01f32;

    let expected_sums = reduced_scalar_oracle::<N, AMBIGUOUS>(
        &mut expected,
        conditioning_haplotypes,
        &prob_sum_h,
        &allele_bytes,
        factor,
        stay_factor,
        mismatch,
        genotype_allele,
        ambiguous_code,
    );
    let actual_sums = unsafe {
        if N == 1 {
            super::run_compressed::<1, AMBIGUOUS>(
                actual.as_mut_ptr(),
                conditioning_haplotypes,
                prob_sum_h.as_ptr(),
                allele_bytes.as_ptr(),
                factor,
                stay_factor,
                mismatch,
                genotype_allele,
                ambiguous_code,
            )
        } else if N == 2 {
            super::run_compressed::<2, AMBIGUOUS>(
                actual.as_mut_ptr(),
                conditioning_haplotypes,
                prob_sum_h.as_ptr(),
                allele_bytes.as_ptr(),
                factor,
                stay_factor,
                mismatch,
                genotype_allele,
                ambiguous_code,
            )
        } else if N == 4 {
            super::run_four::<AMBIGUOUS>(
                actual.as_mut_ptr(),
                conditioning_haplotypes,
                prob_sum_h.as_ptr(),
                allele_bytes.as_ptr(),
                factor,
                stay_factor,
                mismatch,
                genotype_allele,
                ambiguous_code,
            )
        } else {
            debug_assert_eq!(N, HAPLOTYPES);
            super::run_full::<AMBIGUOUS>(
                actual.as_mut_ptr(),
                conditioning_haplotypes,
                prob_sum_h.as_ptr(),
                allele_bytes.as_ptr(),
                factor,
                stay_factor,
                mismatch,
                genotype_allele,
                ambiguous_code,
            )
        }
    };
    let context = format!(
        "reduced N={N}, K={conditioning_haplotypes}, packed={packed_allele:#04x}, \
         genotype={genotype_allele}, code={ambiguous_code:#04x}"
    );
    assert_f32_bits_eq(&actual, &expected, &context);
    assert_f32_bits_eq(&actual_sums, &expected_sums, &context);

    // Exercise update_total's compressed-lane contract: unique sums are
    // duplicated modulo N into all eight logical graph lanes.
    let mut actual_full = [0.0f32; HAPLOTYPES];
    let mut expected_full = [0.0f32; HAPLOTYPES];
    for h in 0..HAPLOTYPES {
        actual_full[h] = actual_sums[h % N];
        expected_full[h] = expected_sums[h % N];
    }
    assert_f32_bits_eq(&actual_full, &expected_full, &context);
    let actual_total = actual_full[0]
        + actual_full[1]
        + actual_full[2]
        + actual_full[3]
        + actual_full[4]
        + actual_full[5]
        + actual_full[6]
        + actual_full[7];
    let expected_total = expected_full[0]
        + expected_full[1]
        + expected_full[2]
        + expected_full[3]
        + expected_full[4]
        + expected_full[5]
        + expected_full[6]
        + expected_full[7];
    assert_eq!(
        actual_total.to_bits(),
        expected_total.to_bits(),
        "{context}"
    );
}

fn assert_reduced_width<const N: usize>() {
    for conditioning_haplotypes in NEON_TEST_K {
        for packed_allele in NEON_TEST_ALLELES {
            for genotype_allele in [false, true] {
                assert_reduced_case::<N, false>(
                    conditioning_haplotypes,
                    packed_allele,
                    genotype_allele,
                    0,
                );
            }
            for ambiguous_code in u8::MIN..=u8::MAX {
                assert_reduced_case::<N, true>(
                    conditioning_haplotypes,
                    packed_allele,
                    false,
                    ambiguous_code,
                );
            }
        }
    }
}

fn collapse_scalar_oracle<const KIND: u8>(
    probability: &mut [f32],
    prob_sum_k: &[f32],
    conditioning_haplotypes: usize,
    allele_bytes: &[u8],
    transferred: f32,
    stay_factor: f32,
    mismatch: f32,
    genotype_allele: bool,
    ambiguous_code: u8,
) -> [f32; HAPLOTYPES] {
    let mut sum_lanes = [0.0f32; HAPLOTYPES * HAPLOTYPES];
    let complete = conditioning_haplotypes / HAPLOTYPES * HAPLOTYPES;
    for k in 0..conditioning_haplotypes {
        let conditioning_allele = if KIND < 2 {
            ((allele_bytes[k >> 3] >> (7 - (k & 7))) & 1) != 0
        } else {
            false
        };
        let base = prob_sum_k[k].mul_add(stay_factor, transferred);
        for h in 0..HAPLOTYPES {
            let graph_haplotype = if KIND == 0 {
                Some(genotype_allele)
            } else if KIND == 1 {
                Some(((ambiguous_code >> h) & 1) != 0)
            } else {
                None
            };
            let emission = if graph_haplotype.is_some_and(|graph| graph != conditioning_allele) {
                mismatch
            } else {
                1.0
            };
            let value = base * emission;
            probability[k * HAPLOTYPES + h] = value;
            let sum_index = if k < complete {
                (k & 7) * HAPLOTYPES + h
            } else {
                h
            };
            sum_lanes[sum_index] += value;
        }
    }
    let mut sums = [0.0f32; HAPLOTYPES];
    for h in 0..HAPLOTYPES {
        let sum01 = sum_lanes[h] + sum_lanes[HAPLOTYPES + h];
        let sum23 = sum_lanes[2 * HAPLOTYPES + h] + sum_lanes[3 * HAPLOTYPES + h];
        let sum45 = sum_lanes[4 * HAPLOTYPES + h] + sum_lanes[5 * HAPLOTYPES + h];
        let sum67 = sum_lanes[6 * HAPLOTYPES + h] + sum_lanes[7 * HAPLOTYPES + h];
        sums[h] = (sum01 + sum23) + (sum45 + sum67);
    }
    sums
}

fn assert_collapse_case<const KIND: u8>(
    conditioning_haplotypes: usize,
    packed_allele: u8,
    genotype_allele: bool,
    ambiguous_code: u8,
) {
    const SENTINEL: f32 = -98_765.5;
    let active = conditioning_haplotypes * HAPLOTYPES;
    let mut expected = vec![0.0f32; active];
    expected.extend([SENTINEL; 7]);
    let mut actual = expected.clone();
    let prob_sum_k: Vec<f32> = (0..conditioning_haplotypes)
        .map(|index| ((index * 29 + 7) % 113 + 1) as f32 / 509.0)
        .collect();
    let allele_bytes = vec![packed_allele; conditioning_haplotypes.div_ceil(8)];
    let transferred = 0.001_171_875f32;
    let stay_factor = 0.873_437_5f32;
    let mismatch = 0.000_100_01f32;
    let expected_sums = collapse_scalar_oracle::<KIND>(
        &mut expected,
        &prob_sum_k,
        conditioning_haplotypes,
        &allele_bytes,
        transferred,
        stay_factor,
        mismatch,
        genotype_allele,
        ambiguous_code,
    );
    let allele_pointer = if KIND < 2 {
        allele_bytes.as_ptr()
    } else {
        ptr::null()
    };
    let actual_sums = unsafe {
        super::collapse::<KIND>(
            actual.as_mut_ptr(),
            prob_sum_k.as_ptr(),
            conditioning_haplotypes,
            allele_pointer,
            transferred,
            stay_factor,
            mismatch,
            genotype_allele,
            ambiguous_code,
        )
    };
    let context = format!(
        "collapse kind={KIND}, K={conditioning_haplotypes}, \
         packed={packed_allele:#04x}, genotype={genotype_allele}, \
         code={ambiguous_code:#04x}"
    );
    assert_f32_bits_eq(&actual, &expected, &context);
    assert_f32_bits_eq(&actual_sums, &expected_sums, &context);
}

fn transition_full_scalar_oracle(
    alpha: &[f32],
    beta: &[f32],
    alpha_sum: &[f32; HAPLOTYPES],
    conditioning_haplotypes: usize,
    alpha_total: f32,
    transition: f32,
    stay_factor: f32,
    output: &mut [f32; HAPLOTYPES * HAPLOTYPES],
) -> f32 {
    let mut transferred = [0.0f32; HAPLOTYPES];
    for h1 in 0..HAPLOTYPES {
        transferred[h1] =
            (alpha_sum[h1] / alpha_total) * transition / conditioning_haplotypes as f32;
    }
    let mut total = 0.0f32;
    for h1 in 0..HAPLOTYPES {
        let mut sums = [0.0f32; HAPLOTYPES];
        for k in 0..conditioning_haplotypes {
            let alpha_value = alpha[k * HAPLOTYPES + h1].mul_add(stay_factor, transferred[h1]);
            for h2 in 0..HAPLOTYPES {
                sums[h2] = alpha_value.mul_add(beta[k * HAPLOTYPES + h2], sums[h2]);
            }
        }
        let row = h1 * HAPLOTYPES;
        output[row..row + HAPLOTYPES].copy_from_slice(&sums);
        for value in sums {
            total += value;
        }
    }
    total
}

fn invalid_single_total(total: f32) -> bool {
    total.is_nan() || total.is_infinite() || total < f32::MIN_POSITIVE
}

#[test]
fn neon_reduced_kernels_match_scalar_oracle_exhaustively() {
    assert_reduced_width::<1>();
    assert_reduced_width::<2>();
    assert_reduced_width::<4>();
    assert_reduced_width::<HAPLOTYPES>();
}

#[test]
fn neon_reduced_recurrence_preserves_fused_boundary() {
    let multiplicand = f32::from_bits(0x3f80_0001);
    let addend = -f32::from_bits(0x3f80_0002);
    let fused = multiplicand.mul_add(multiplicand, addend);
    assert_ne!(
        fused.to_bits(),
        (multiplicand * multiplicand + addend).to_bits()
    );

    let mut probability = vec![multiplicand; HAPLOTYPES];
    probability.push(-91.25);
    let mut prob_sum_h = [0.0f32; HAPLOTYPES];
    prob_sum_h[0] = addend;
    let alleles = [0u8];
    let sums = unsafe {
        super::run_compressed::<1, false>(
            probability.as_mut_ptr(),
            HAPLOTYPES,
            prob_sum_h.as_ptr(),
            alleles.as_ptr(),
            1.0,
            multiplicand,
            0.0001,
            false,
            0,
        )
    };
    for value in &probability[..HAPLOTYPES] {
        assert_eq!(value.to_bits(), fused.to_bits());
    }
    let pair = fused + fused;
    assert_eq!(sums[0].to_bits(), ((pair + pair) + (pair + pair)).to_bits());
    assert_eq!(probability[HAPLOTYPES].to_bits(), (-91.25f32).to_bits());
}

#[test]
fn neon_collapse_kernels_match_scalar_oracle() {
    for conditioning_haplotypes in NEON_TEST_K {
        for packed_allele in NEON_TEST_ALLELES {
            for genotype_allele in [false, true] {
                assert_collapse_case::<0>(
                    conditioning_haplotypes,
                    packed_allele,
                    genotype_allele,
                    0,
                );
            }
            for ambiguous_code in u8::MIN..=u8::MAX {
                assert_collapse_case::<1>(
                    conditioning_haplotypes,
                    packed_allele,
                    false,
                    ambiguous_code,
                );
            }
        }
        // A null allele pointer proves that missing collapse is genuinely
        // independent of the conditioning allele slab.
        assert_collapse_case::<2>(conditioning_haplotypes, 0, false, 0);
    }
}

#[test]
fn neon_full_transition_contraction_matches_scalar_oracle_and_guards() {
    for conditioning_haplotypes in NEON_TEST_K {
        let alpha: Vec<f32> = (0..conditioning_haplotypes * HAPLOTYPES)
            .map(|index| ((index * 31 + 5) % 233 + 1) as f32 / 887.0)
            .collect();
        let beta: Vec<f32> = (0..conditioning_haplotypes * HAPLOTYPES)
            .map(|index| ((index * 43 + 17) % 241 + 1) as f32 / 911.0)
            .collect();
        let alpha_sum = [0.109f32, 0.127, 0.083, 0.151, 0.097, 0.139, 0.173, 0.121];
        let alpha_total = 0.9375f32;
        let transition = 0.018_75f32;
        let stay_factor = 0.981_25f32 / alpha_total;
        let mut expected = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let expected_total = transition_full_scalar_oracle(
            &alpha,
            &beta,
            &alpha_sum,
            conditioning_haplotypes,
            alpha_total,
            transition,
            stay_factor,
            &mut expected,
        );
        let mut actual = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let actual_total = unsafe {
            super::transition_full(
                alpha.as_ptr(),
                beta.as_ptr(),
                alpha_sum.as_ptr(),
                conditioning_haplotypes,
                alpha_total,
                transition,
                stay_factor,
                actual.as_mut_ptr(),
            )
        };
        let context = format!("full transition K={conditioning_haplotypes}");
        assert_f32_bits_eq(&actual, &expected, &context);
        assert_eq!(
            actual_total.to_bits(),
            expected_total.to_bits(),
            "{context}"
        );
        assert!(!invalid_single_total(actual_total));
    }

    for exceptional in [0.0f32, f32::INFINITY, f32::NAN] {
        let alpha = [exceptional; HAPLOTYPES];
        let beta = [1.0f32; HAPLOTYPES];
        let alpha_sum = [0.0f32; HAPLOTYPES];
        let mut output = [0.0f32; HAPLOTYPES * HAPLOTYPES];
        let total = unsafe {
            super::transition_full(
                alpha.as_ptr(),
                beta.as_ptr(),
                alpha_sum.as_ptr(),
                1,
                1.0,
                0.0,
                1.0,
                output.as_mut_ptr(),
            )
        };
        assert!(invalid_single_total(total), "exceptional={exceptional:?}");
    }
    assert!(invalid_single_total(f32::from_bits(
        f32::MIN_POSITIVE.to_bits() - 1
    )));
    assert!(!invalid_single_total(f32::MIN_POSITIVE));
}
