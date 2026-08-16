mod bitmatrix;
mod conditioning;
mod genotype;
mod hmm;
mod pbwt;

const ABI_VERSION: u32 = 1;

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;
const PHILOX_ROUNDS: usize = 10;

#[inline]
fn multiply_high_low(lhs: u32, rhs: u32) -> (u32, u32) {
    let product = u64::from(lhs) * u64::from(rhs);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (high0, low0) = multiply_high_low(PHILOX_M0, counter[0]);
    let (high1, low1) = multiply_high_low(PHILOX_M1, counter[2]);
    [
        high1 ^ counter[1] ^ key[0],
        low1,
        high0 ^ counter[3] ^ key[1],
        low0,
    ]
}

#[inline]
fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..PHILOX_ROUNDS {
        counter = philox_round(counter, key);
        if round + 1 != PHILOX_ROUNDS {
            key[0] = key[0].wrapping_add(PHILOX_W0);
            key[1] = key[1].wrapping_add(PHILOX_W1);
        }
    }
    counter
}

// A bijection over u64. For a fixed master seed this maps each packed
// (domain, iteration) pair to a distinct Philox key.
#[inline]
fn permute_u64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
fn application_block(seed: u64, domain: u32, iteration: u32, item: u64, block: u64) -> [u32; 4] {
    let stream = (u64::from(domain) << 32) | u64::from(iteration);
    let key64 = seed ^ permute_u64(stream);
    let key = [key64 as u32, (key64 >> 32) as u32];
    let counter = [
        block as u32,
        (block >> 32) as u32,
        item as u32,
        (item >> 32) as u32,
    ];
    philox4x32_10(counter, key)
}

#[no_mangle]
pub extern "C" fn shapeit_rng_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Evaluate the raw Philox4x32-10 primitive.
///
/// # Safety
///
/// Unless null, `counter`, `key`, and `output` must point to readable arrays of
/// 4 words, readable arrays of 2 words, and writable arrays of 4 words,
/// respectively. Null pointers make the call a no-op.
pub unsafe extern "C" fn shapeit_rng_philox4x32_10_raw(
    counter: *const u32,
    key: *const u32,
    output: *mut u32,
) {
    if counter.is_null() || key.is_null() || output.is_null() {
        return;
    }
    let counter = [*counter, *counter.add(1), *counter.add(2), *counter.add(3)];
    let key = [*key, *key.add(1)];
    let result = philox4x32_10(counter, key);
    for (index, word) in result.into_iter().enumerate() {
        *output.add(index) = word;
    }
}

#[no_mangle]
/// Evaluate one block of SHAPEIT RNG ABI version 1.
///
/// # Safety
///
/// Unless null, `output` must point to a writable array of 4 words. A null
/// pointer makes the call a no-op.
pub unsafe extern "C" fn shapeit_rng_block_v1(
    seed: u64,
    domain: u32,
    iteration: u32,
    item: u64,
    block: u64,
    output: *mut u32,
) {
    if output.is_null() {
        return;
    }
    let result = application_block(seed, domain, iteration, item, block);
    for (index, word) in result.into_iter().enumerate() {
        *output.add(index) = word;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_random123_known_answer_vectors() {
        let vectors = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
            ),
            (
                [u32::MAX; 4],
                [u32::MAX; 2],
                [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
            ),
            (
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0],
                [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
            ),
        ];

        for (counter, key, expected) in vectors {
            assert_eq!(philox4x32_10(counter, key), expected);
        }
    }

    #[test]
    fn logical_coordinates_select_distinct_blocks() {
        let baseline = application_block(42, 2, 3, 5, 7);
        assert_eq!(
            baseline,
            [0x5cce_e54f, 0x552d_93d6, 0xc636_cb6d, 0x8201_2c17]
        );
        assert_ne!(baseline, application_block(43, 2, 3, 5, 7));
        assert_ne!(baseline, application_block(42, 3, 3, 5, 7));
        assert_ne!(baseline, application_block(42, 2, 4, 5, 7));
        assert_ne!(baseline, application_block(42, 2, 3, 6, 7));
        assert_ne!(baseline, application_block(42, 2, 3, 5, 8));
    }
}
