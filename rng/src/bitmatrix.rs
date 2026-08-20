use core::slice;

const ABI_VERSION: u32 = 1;
const STATUS_OK: u32 = 0;
const STATUS_NULL_POINTER: u32 = 1;
const STATUS_INVALID_DIMENSIONS: u32 = 2;
const STATUS_OUT_OF_BOUNDS: u32 = 3;
const STATUS_INTEGER_OVERFLOW: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TransposeLayout {
    source_stride: usize,
    row_count_padded: usize,
    source_byte_first: usize,
    source_byte_count: usize,
    target_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FullTransposeLayout {
    source_stride: usize,
    max_rows: usize,
    max_cols: usize,
    target_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HetOverlapLayout {
    offsets: [usize; 4],
    byte_count: usize,
}

#[inline]
fn padded_rows(row_count: usize) -> Option<usize> {
    row_count.checked_add(7).map(|count| count & !7)
}

fn validate_layout(
    source_length: usize,
    source_stride: usize,
    rows: &[u32],
    source_byte_first: usize,
    source_byte_count: usize,
    target_length: usize,
    target_stride: usize,
) -> Result<TransposeLayout, u32> {
    if source_byte_count == 0 || source_stride == 0 {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let row_count_padded = padded_rows(rows.len()).ok_or(STATUS_INTEGER_OVERFLOW)?;
    if target_stride != row_count_padded >> 3 {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let source_byte_end = source_byte_first
        .checked_add(source_byte_count)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if source_byte_end > source_stride {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let target_rows = source_byte_count
        .checked_mul(8)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    let required_target_length = target_rows
        .checked_mul(target_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_target_length > target_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    for &row in rows {
        let row_start = (row as usize)
            .checked_mul(source_stride)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        let row_end = row_start
            .checked_add(source_byte_end)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if row_end > source_length {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
    }
    Ok(TransposeLayout {
        source_stride,
        row_count_padded,
        source_byte_first,
        source_byte_count,
        target_stride,
    })
}

fn validate_full_layout(
    source_length: usize,
    source_rows: usize,
    source_stride: usize,
    max_rows: usize,
    max_cols: usize,
    target_length: usize,
    target_stride: usize,
) -> Result<FullTransposeLayout, u32> {
    if source_rows == 0
        || source_stride == 0
        || max_rows == 0
        || max_cols == 0
        || target_stride == 0
        || source_rows & 7 != 0
        || max_rows & 7 != 0
        || max_cols & 7 != 0
    {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    if max_rows > source_rows {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let source_cols = source_stride
        .checked_mul(8)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if max_cols > source_cols {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    if target_stride < max_rows >> 3 {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let required_source_length = source_rows
        .checked_mul(source_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_source_length > source_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let required_target_length = max_cols
        .checked_mul(target_stride)
        .ok_or(STATUS_INTEGER_OVERFLOW)?;
    if required_target_length > target_length {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    Ok(FullTransposeLayout {
        source_stride,
        max_rows,
        max_cols,
        target_stride,
    })
}

fn validate_het_overlap_layout(
    source_length: usize,
    source_stride: usize,
    individual0: usize,
    individual1: usize,
    start: usize,
    stop: usize,
) -> Result<HetOverlapLayout, u32> {
    if source_stride == 0 || start > stop {
        return Err(STATUS_INVALID_DIMENSIONS);
    }
    let byte_first = start >> 3;
    let byte_last = stop >> 3;
    if byte_last >= source_stride {
        return Err(STATUS_OUT_OF_BOUNDS);
    }
    let byte_count = byte_last - byte_first + 1;
    let row00 = individual0.checked_mul(2).ok_or(STATUS_INTEGER_OVERFLOW)?;
    let row01 = row00.checked_add(1).ok_or(STATUS_INTEGER_OVERFLOW)?;
    let row10 = individual1.checked_mul(2).ok_or(STATUS_INTEGER_OVERFLOW)?;
    let row11 = row10.checked_add(1).ok_or(STATUS_INTEGER_OVERFLOW)?;
    let mut offsets = [0usize; 4];
    for (offset, row) in offsets.iter_mut().zip([row00, row01, row10, row11]) {
        *offset = row
            .checked_mul(source_stride)
            .and_then(|value| value.checked_add(byte_first))
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        let end = offset
            .checked_add(byte_count)
            .ok_or(STATUS_INTEGER_OVERFLOW)?;
        if end > source_length {
            return Err(STATUS_OUT_OF_BOUNDS);
        }
    }
    Ok(HetOverlapLayout {
        offsets,
        byte_count,
    })
}

#[inline]
#[cfg(any(test, not(target_arch = "aarch64")))]
fn multiply_high(lhs: u32, rhs: u32) -> u32 {
    ((u64::from(lhs) * u64::from(rhs)) >> 32) as u32
}

#[inline]
#[cfg(any(test, not(target_arch = "aarch64")))]
fn transpose_word_portable(packed: u64) -> [u8; 8] {
    let low = packed as u32;
    let high = (packed >> 32) as u32;
    let mut transposed = [0u8; 8];
    for (bit, target) in transposed.iter_mut().enumerate().take(7) {
        let mask = 0x8080_8080 >> bit;
        let multiplier = 0x0204_0810 << bit;
        let upper = multiply_high(high & mask, multiplier) & 0x0f;
        let lower = multiply_high(low & mask, multiplier) & 0x0f;
        *target = ((upper << 4) | lower) as u8;
    }
    let upper = multiply_high((high << 7) & 0x8080_8080, 0x0204_0810) & 0x0f;
    let lower = multiply_high((low << 7) & 0x8080_8080, 0x0204_0810) & 0x0f;
    transposed[7] = ((upper << 4) | lower) as u8;
    transposed
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn transpose_portable(source: &[u8], rows: &[u32], target: &mut [u8], layout: TransposeLayout) {
    for byte in 0..layout.source_byte_count {
        for row in (0..layout.row_count_padded).step_by(8) {
            let mut packed = 0u64;
            for lane in 0..8 {
                let value = if row + lane < rows.len() {
                    let source_row = rows[row + lane] as usize;
                    source[source_row * layout.source_stride + layout.source_byte_first + byte]
                } else {
                    0
                };
                packed |= u64::from(value) << ((7 - lane) * 8);
            }
            for (bit, value) in transpose_word_portable(packed).into_iter().enumerate() {
                target[(byte * 8 + bit) * layout.target_stride + (row >> 3)] = value;
            }
        }
    }
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn full_transpose_portable(source: &[u8], target: &mut [u8], layout: FullTransposeLayout) {
    for byte in 0..layout.max_cols >> 3 {
        for row in (0..layout.max_rows).step_by(8) {
            let mut packed = 0u64;
            for lane in 0..8 {
                packed |= u64::from(source[(row + lane) * layout.source_stride + byte])
                    << ((7 - lane) * 8);
            }
            for (bit, value) in transpose_word_portable(packed).into_iter().enumerate() {
                target[(byte * 8 + bit) * layout.target_stride + (row >> 3)] = value;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn transpose_word_aarch64(mut packed: u64) -> u64 {
    // Transpose an 8-by-8 bit matrix in three SWAR butterfly stages. `packed`
    // contains the first source row in its most-significant byte, and the
    // result contains the first source bit in its most-significant byte. This
    // is exactly the byte and bit ordering used by the portable and BMI2
    // implementations.
    let mut swap = (packed ^ (packed >> 7)) & 0x00aa_00aa_00aa_00aa;
    packed ^= swap ^ (swap << 7);
    swap = (packed ^ (packed >> 14)) & 0x0000_cccc_0000_cccc;
    packed ^= swap ^ (swap << 14);
    swap = (packed ^ (packed >> 28)) & 0x0000_0000_f0f0_f0f0;
    packed ^ swap ^ (swap << 28)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_word_pair_aarch64(
    mut packed: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::{vandq_u64, vdupq_n_u64, veorq_u64, vshlq_n_u64, vshrq_n_u64};

    let mut swap = vandq_u64(
        veorq_u64(packed, vshrq_n_u64::<7>(packed)),
        vdupq_n_u64(0x00aa_00aa_00aa_00aa),
    );
    packed = veorq_u64(veorq_u64(packed, swap), vshlq_n_u64::<7>(swap));
    swap = vandq_u64(
        veorq_u64(packed, vshrq_n_u64::<14>(packed)),
        vdupq_n_u64(0x0000_cccc_0000_cccc),
    );
    packed = veorq_u64(veorq_u64(packed, swap), vshlq_n_u64::<14>(swap));
    swap = vandq_u64(
        veorq_u64(packed, vshrq_n_u64::<28>(packed)),
        vdupq_n_u64(0x0000_0000_f0f0_f0f0),
    );
    veorq_u64(veorq_u64(packed, swap), vshlq_n_u64::<28>(swap))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_byte_tile_aarch64(source: *const u8, offsets: [usize; 8]) -> [u64; 8] {
    use core::arch::aarch64::{
        uint8x8_t, vcombine_u64, vgetq_lane_u64, vld1_u8, vreinterpret_u16_u8,
        vreinterpret_u32_u16, vreinterpret_u64_u8, vreinterpret_u8_u32, vrev64_u8, vtrn1_u16,
        vtrn1_u32, vtrn1_u8, vtrn2_u16, vtrn2_u32, vtrn2_u8,
    };

    // First transpose the eight contiguous source-byte vectors. Each resulting
    // vector contains one byte position gathered from all eight rows, which is
    // then bit-transposed by the shared SWAR core below.
    let rows = [
        vld1_u8(source.add(offsets[0])),
        vld1_u8(source.add(offsets[1])),
        vld1_u8(source.add(offsets[2])),
        vld1_u8(source.add(offsets[3])),
        vld1_u8(source.add(offsets[4])),
        vld1_u8(source.add(offsets[5])),
        vld1_u8(source.add(offsets[6])),
        vld1_u8(source.add(offsets[7])),
    ];

    let bytes = [
        vtrn1_u8(rows[0], rows[1]),
        vtrn2_u8(rows[0], rows[1]),
        vtrn1_u8(rows[2], rows[3]),
        vtrn2_u8(rows[2], rows[3]),
        vtrn1_u8(rows[4], rows[5]),
        vtrn2_u8(rows[4], rows[5]),
        vtrn1_u8(rows[6], rows[7]),
        vtrn2_u8(rows[6], rows[7]),
    ];
    let halfwords = [
        vtrn1_u16(vreinterpret_u16_u8(bytes[0]), vreinterpret_u16_u8(bytes[2])),
        vtrn2_u16(vreinterpret_u16_u8(bytes[0]), vreinterpret_u16_u8(bytes[2])),
        vtrn1_u16(vreinterpret_u16_u8(bytes[1]), vreinterpret_u16_u8(bytes[3])),
        vtrn2_u16(vreinterpret_u16_u8(bytes[1]), vreinterpret_u16_u8(bytes[3])),
        vtrn1_u16(vreinterpret_u16_u8(bytes[4]), vreinterpret_u16_u8(bytes[6])),
        vtrn2_u16(vreinterpret_u16_u8(bytes[4]), vreinterpret_u16_u8(bytes[6])),
        vtrn1_u16(vreinterpret_u16_u8(bytes[5]), vreinterpret_u16_u8(bytes[7])),
        vtrn2_u16(vreinterpret_u16_u8(bytes[5]), vreinterpret_u16_u8(bytes[7])),
    ];
    let words = [
        vtrn1_u32(
            vreinterpret_u32_u16(halfwords[0]),
            vreinterpret_u32_u16(halfwords[4]),
        ),
        vtrn1_u32(
            vreinterpret_u32_u16(halfwords[2]),
            vreinterpret_u32_u16(halfwords[6]),
        ),
        vtrn1_u32(
            vreinterpret_u32_u16(halfwords[1]),
            vreinterpret_u32_u16(halfwords[5]),
        ),
        vtrn1_u32(
            vreinterpret_u32_u16(halfwords[3]),
            vreinterpret_u32_u16(halfwords[7]),
        ),
        vtrn2_u32(
            vreinterpret_u32_u16(halfwords[0]),
            vreinterpret_u32_u16(halfwords[4]),
        ),
        vtrn2_u32(
            vreinterpret_u32_u16(halfwords[2]),
            vreinterpret_u32_u16(halfwords[6]),
        ),
        vtrn2_u32(
            vreinterpret_u32_u16(halfwords[1]),
            vreinterpret_u32_u16(halfwords[5]),
        ),
        vtrn2_u32(
            vreinterpret_u32_u16(halfwords[3]),
            vreinterpret_u32_u16(halfwords[7]),
        ),
    ];

    let pack = |word| {
        let bytes: uint8x8_t = vreinterpret_u8_u32(word);
        vreinterpret_u64_u8(vrev64_u8(bytes))
    };
    let pairs = [
        transpose_word_pair_aarch64(vcombine_u64(pack(words[0]), pack(words[1]))),
        transpose_word_pair_aarch64(vcombine_u64(pack(words[2]), pack(words[3]))),
        transpose_word_pair_aarch64(vcombine_u64(pack(words[4]), pack(words[5]))),
        transpose_word_pair_aarch64(vcombine_u64(pack(words[6]), pack(words[7]))),
    ];
    [
        vgetq_lane_u64::<0>(pairs[0]),
        vgetq_lane_u64::<1>(pairs[0]),
        vgetq_lane_u64::<0>(pairs[1]),
        vgetq_lane_u64::<1>(pairs[1]),
        vgetq_lane_u64::<0>(pairs[2]),
        vgetq_lane_u64::<1>(pairs[2]),
        vgetq_lane_u64::<0>(pairs[3]),
        vgetq_lane_u64::<1>(pairs[3]),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_transposed_word_aarch64(
    target: *mut u8,
    packed: u64,
    source_byte: usize,
    target_byte: usize,
    target_stride: usize,
) {
    for bit in 0..8 {
        target
            .add((source_byte * 8 + bit) * target_stride + target_byte)
            .write((packed >> ((7 - bit) * 8)) as u8);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_transposed_word_pair_aarch64(
    target: *mut u8,
    packed0: u64,
    packed1: u64,
    source_byte: usize,
    target_byte: usize,
    target_stride: usize,
) {
    for bit in 0..8 {
        let values = [
            (packed0 >> ((7 - bit) * 8)) as u8,
            (packed1 >> ((7 - bit) * 8)) as u8,
        ];
        target
            .add((source_byte * 8 + bit) * target_stride + target_byte)
            .cast::<u16>()
            .write_unaligned(u16::from_ne_bytes(values));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_aarch64(
    source: &[u8],
    rows: &[u32],
    target: &mut [u8],
    layout: TransposeLayout,
) {
    let source = source.as_ptr();
    let target = target.as_mut_ptr();
    let full_rows = rows.len() & !7;
    let full_bytes = layout.source_byte_count & !7;

    // Match the x86 cache-blocking shape: for each 64 selected rows, visit a
    // compact eight-byte source span and its 64 target rows before advancing.
    // Adjacent eight-row groups are stored as native-endian u16 pairs, making
    // the target writes contiguous even though the selected source rows are a
    // gather.
    for row_block in (0..full_rows).step_by(64) {
        let row_stop = row_block + core::cmp::min(64, full_rows - row_block);
        for byte_block in (0..full_bytes).step_by(8) {
            for row in (row_block..row_stop).step_by(16) {
                let offsets0 = core::array::from_fn(|lane| {
                    *rows.get_unchecked(row + lane) as usize * layout.source_stride
                        + layout.source_byte_first
                        + byte_block
                });
                let packed0 = transpose_byte_tile_aarch64(source, offsets0);
                if row_stop - row >= 16 {
                    let offsets1 = core::array::from_fn(|lane| {
                        *rows.get_unchecked(row + 8 + lane) as usize * layout.source_stride
                            + layout.source_byte_first
                            + byte_block
                    });
                    let packed1 = transpose_byte_tile_aarch64(source, offsets1);
                    for byte in 0..8 {
                        store_transposed_word_pair_aarch64(
                            target,
                            packed0[byte],
                            packed1[byte],
                            byte_block + byte,
                            row >> 3,
                            layout.target_stride,
                        );
                    }
                } else {
                    for (byte, packed) in packed0.into_iter().enumerate() {
                        store_transposed_word_aarch64(
                            target,
                            packed,
                            byte_block + byte,
                            row >> 3,
                            layout.target_stride,
                        );
                    }
                }
            }
        }
        for byte in full_bytes..layout.source_byte_count {
            for row in (row_block..row_stop).step_by(16) {
                let mut packed0 = 0u64;
                for lane in 0..8 {
                    let source_row = *rows.get_unchecked(row + lane) as usize;
                    let value = *source
                        .add(source_row * layout.source_stride + layout.source_byte_first + byte);
                    packed0 |= u64::from(value) << ((7 - lane) * 8);
                }
                let packed0 = transpose_word_aarch64(packed0);
                if row_stop - row >= 16 {
                    let mut packed1 = 0u64;
                    for lane in 0..8 {
                        let source_row = *rows.get_unchecked(row + 8 + lane) as usize;
                        let value = *source.add(
                            source_row * layout.source_stride + layout.source_byte_first + byte,
                        );
                        packed1 |= u64::from(value) << ((7 - lane) * 8);
                    }
                    store_transposed_word_pair_aarch64(
                        target,
                        packed0,
                        transpose_word_aarch64(packed1),
                        byte,
                        row >> 3,
                        layout.target_stride,
                    );
                } else {
                    store_transposed_word_aarch64(
                        target,
                        packed0,
                        byte,
                        row >> 3,
                        layout.target_stride,
                    );
                }
            }
        }
    }

    if full_rows < rows.len() {
        let row = full_rows;
        for byte in 0..layout.source_byte_count {
            let mut packed = 0u64;
            for lane in 0..rows.len() - full_rows {
                let source_row = *rows.get_unchecked(row + lane) as usize;
                let value = *source
                    .add(source_row * layout.source_stride + layout.source_byte_first + byte);
                packed |= u64::from(value) << ((7 - lane) * 8);
            }
            store_transposed_word_aarch64(
                target,
                transpose_word_aarch64(packed),
                byte,
                row >> 3,
                layout.target_stride,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn full_transpose_aarch64(source: &[u8], target: &mut [u8], layout: FullTransposeLayout) {
    let source = source.as_ptr();
    let target = target.as_mut_ptr();
    let source_byte_count = layout.max_cols >> 3;
    let full_bytes = source_byte_count & !7;

    for row_block in (0..layout.max_rows).step_by(64) {
        let row_stop = row_block + core::cmp::min(64, layout.max_rows - row_block);
        for byte_block in (0..full_bytes).step_by(8) {
            for row in (row_block..row_stop).step_by(16) {
                let offsets0 =
                    core::array::from_fn(|lane| (row + lane) * layout.source_stride + byte_block);
                let packed0 = transpose_byte_tile_aarch64(source, offsets0);
                if row_stop - row >= 16 {
                    let offsets1 = core::array::from_fn(|lane| {
                        (row + 8 + lane) * layout.source_stride + byte_block
                    });
                    let packed1 = transpose_byte_tile_aarch64(source, offsets1);
                    for byte in 0..8 {
                        store_transposed_word_pair_aarch64(
                            target,
                            packed0[byte],
                            packed1[byte],
                            byte_block + byte,
                            row >> 3,
                            layout.target_stride,
                        );
                    }
                } else {
                    for (byte, packed) in packed0.into_iter().enumerate() {
                        store_transposed_word_aarch64(
                            target,
                            packed,
                            byte_block + byte,
                            row >> 3,
                            layout.target_stride,
                        );
                    }
                }
            }
        }
        for byte in full_bytes..source_byte_count {
            for row in (row_block..row_stop).step_by(16) {
                let mut packed0 = 0u64;
                for lane in 0..8 {
                    let value = *source.add((row + lane) * layout.source_stride + byte);
                    packed0 |= u64::from(value) << ((7 - lane) * 8);
                }
                let packed0 = transpose_word_aarch64(packed0);
                if row_stop - row >= 16 {
                    let mut packed1 = 0u64;
                    for lane in 0..8 {
                        let value = *source.add((row + 8 + lane) * layout.source_stride + byte);
                        packed1 |= u64::from(value) << ((7 - lane) * 8);
                    }
                    store_transposed_word_pair_aarch64(
                        target,
                        packed0,
                        transpose_word_aarch64(packed1),
                        byte,
                        row >> 3,
                        layout.target_stride,
                    );
                } else {
                    store_transposed_word_aarch64(
                        target,
                        packed0,
                        byte,
                        row >> 3,
                        layout.target_stride,
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn bmi2_available() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    // CPUID is available on every x86_64 processor. Leaf 7 is queried only
    // when the maximum supported basic leaf includes it.
    // SAFETY: CPUID is available on x86-64, and leaf 7 is queried only after
    // the maximum supported basic leaf has been checked.
    unsafe { __cpuid(0).eax >= 7 && (__cpuid_count(7, 0).ebx & (1 << 8)) != 0 }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn transpose_bmi2(source: &[u8], rows: &[u32], target: &mut [u8], layout: TransposeLayout) {
    use core::arch::x86_64::_pext_u64;

    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
    for row_block in (0..layout.row_count_padded).step_by(64) {
        let row_stop = core::cmp::min(row_block + 64, layout.row_count_padded);
        for byte_block in (0..layout.source_byte_count).step_by(8) {
            let byte_stop = core::cmp::min(byte_block + 8, layout.source_byte_count);
            for row in (row_block..row_stop).step_by(8) {
                for byte in byte_block..byte_stop {
                    let mut packed = 0u64;
                    for lane in 0..8 {
                        let value = if row + lane < rows.len() {
                            let source_row = *rows.get_unchecked(row + lane) as usize;
                            *source.get_unchecked(
                                source_row * layout.source_stride + layout.source_byte_first + byte,
                            )
                        } else {
                            0
                        };
                        packed |= u64::from(value) << ((7 - lane) * 8);
                    }
                    for bit in 0..8 {
                        let target_index = (byte * 8 + bit) * layout.target_stride + (row >> 3);
                        *target.get_unchecked_mut(target_index) =
                            _pext_u64(packed, HIGH_BITS >> bit) as u8;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn full_transpose_bmi2(source: &[u8], target: &mut [u8], layout: FullTransposeLayout) {
    use core::arch::x86_64::_pext_u64;

    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
    let source_byte_count = layout.max_cols >> 3;
    for row_block in (0..layout.max_rows).step_by(64) {
        let row_stop = core::cmp::min(row_block + 64, layout.max_rows);
        for byte_block in (0..source_byte_count).step_by(8) {
            let byte_stop = core::cmp::min(byte_block + 8, source_byte_count);
            for row in (row_block..row_stop).step_by(8) {
                for byte in byte_block..byte_stop {
                    let mut packed = 0u64;
                    for lane in 0..8 {
                        packed |= u64::from(
                            *source.get_unchecked((row + lane) * layout.source_stride + byte),
                        ) << ((7 - lane) * 8);
                    }
                    for bit in 0..8 {
                        let target_index = (byte * 8 + bit) * layout.target_stride + (row >> 3);
                        *target.get_unchecked_mut(target_index) =
                            _pext_u64(packed, HIGH_BITS >> bit) as u8;
                    }
                }
            }
        }
    }
}

fn subset_transpose(source: &[u8], rows: &[u32], target: &mut [u8], layout: TransposeLayout) {
    #[cfg(target_arch = "x86_64")]
    if bmi2_available() {
        // SAFETY: the feature check establishes BMI2 support, and the caller
        // has validated every source and target bound.
        unsafe {
            transpose_bmi2(source, rows, target, layout);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Advanced SIMD is part of the AArch64 execution environment,
        // and the caller has validated every source and target bound.
        unsafe {
            transpose_aarch64(source, rows, target, layout);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    transpose_portable(source, rows, target, layout);
}

fn full_transpose(source: &[u8], target: &mut [u8], layout: FullTransposeLayout) {
    #[cfg(target_arch = "x86_64")]
    if bmi2_available() {
        // SAFETY: the feature check establishes BMI2 support, and the caller
        // has validated every source and target bound.
        unsafe {
            full_transpose_bmi2(source, target, layout);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Advanced SIMD is part of the AArch64 execution environment,
        // and the caller has validated every source and target bound.
        unsafe {
            full_transpose_aarch64(source, target, layout);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    full_transpose_portable(source, target, layout);
}

fn refresh_sample_haplotypes(
    variants: &[u8],
    variant_count: usize,
    first_time: bool,
    haplotype0: &mut [u8],
    haplotype1: &mut [u8],
) {
    for byte in 0..variant_count.div_ceil(8) {
        let mut update_mask = 0u8;
        let mut allele0 = 0u8;
        let mut allele1 = 0u8;
        let locus_start = byte * 8;
        let locus_stop = core::cmp::min(locus_start + 8, variant_count);
        for locus in locus_start..locus_stop {
            let packed = variants[locus >> 1];
            let code = (packed >> ((locus & 1) * 4)) & 15;
            let genotype = code & 3;
            let bit = 0x80 >> (locus - locus_start);
            if first_time || genotype == 1 || genotype == 2 {
                update_mask |= bit;
            }
            if code & 4 != 0 {
                allele0 |= bit;
            }
            if code & 8 != 0 {
                allele1 |= bit;
            }
        }
        haplotype0[byte] = (haplotype0[byte] & !update_mask) | (allele0 & update_mask);
        haplotype1[byte] = (haplotype1[byte] & !update_mask) | (allele1 & update_mask);
    }
}

#[inline]
fn overlap_from_counts(intersection: u32, union: u32) -> f32 {
    if union == 0 {
        0.0
    } else {
        union.wrapping_sub(intersection) as f32 / union as f32
    }
}

fn het_overlap_portable(source: &[u8], layout: HetOverlapLayout) -> f32 {
    let [offset00, offset01, offset10, offset11] = layout.offsets;
    let mut intersection = 0u32;
    let mut union = 0u32;
    for byte in 0..layout.byte_count {
        let het0 = source[offset00 + byte] ^ source[offset01 + byte];
        let het1 = source[offset10 + byte] ^ source[offset11 + byte];
        intersection = intersection.wrapping_add((het0 ^ het1).count_ones());
        union = union.wrapping_add((het0 | het1).count_ones());
    }
    overlap_from_counts(intersection, union)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn popcnt_available() -> bool {
    use core::arch::x86_64::__cpuid;

    // SAFETY: Basic CPUID leaf 1 is available on every x86-64 processor.
    unsafe { (__cpuid(1).ecx & (1 << 23)) != 0 }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn het_overlap_popcnt(source: &[u8], layout: HetOverlapLayout) -> f32 {
    let [offset00, offset01, offset10, offset11] = layout.offsets;
    let mut intersection = 0u32;
    let mut union = 0u32;
    let mut byte = 0usize;
    while byte + 8 <= layout.byte_count {
        let word00 = core::ptr::read_unaligned(source.as_ptr().add(offset00 + byte).cast::<u64>());
        let word01 = core::ptr::read_unaligned(source.as_ptr().add(offset01 + byte).cast::<u64>());
        let word10 = core::ptr::read_unaligned(source.as_ptr().add(offset10 + byte).cast::<u64>());
        let word11 = core::ptr::read_unaligned(source.as_ptr().add(offset11 + byte).cast::<u64>());
        let het0 = word00 ^ word01;
        let het1 = word10 ^ word11;
        intersection = intersection.wrapping_add((het0 ^ het1).count_ones());
        union = union.wrapping_add((het0 | het1).count_ones());
        byte += 8;
    }
    while byte < layout.byte_count {
        let het0 = *source.get_unchecked(offset00 + byte) ^ *source.get_unchecked(offset01 + byte);
        let het1 = *source.get_unchecked(offset10 + byte) ^ *source.get_unchecked(offset11 + byte);
        intersection = intersection.wrapping_add((het0 ^ het1).count_ones());
        union = union.wrapping_add((het0 | het1).count_ones());
        byte += 1;
    }
    overlap_from_counts(intersection, union)
}

fn het_overlap(source: &[u8], layout: HetOverlapLayout) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if popcnt_available() {
        // SAFETY: CPUID establishes POPCNT support and layout validation covers
        // every unaligned word and byte read.
        return unsafe { het_overlap_popcnt(source, layout) };
    }
    het_overlap_portable(source, layout)
}

pub(crate) fn heterozygote_overlap(
    source: &[u8],
    source_stride: usize,
    individual0: usize,
    individual1: usize,
    start: usize,
    stop: usize,
) -> Result<f32, u32> {
    let layout = validate_het_overlap_layout(
        source.len(),
        source_stride,
        individual0,
        individual1,
        start,
        stop,
    )?;
    Ok(het_overlap(source, layout))
}

#[no_mangle]
pub extern "C" fn shapeit_bitmatrix_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
/// Select rows from a row-major bitmatrix and transpose the selected byte span.
///
/// # Safety
///
/// Non-empty buffers must be valid for their stated lengths. `rows` must be
/// readable for `row_count` elements. Source and target buffers must not
/// overlap. Invalid dimensions and indexes are reported without writing the
/// target.
pub unsafe extern "C" fn shapeit_bitmatrix_subset_transpose_v1(
    source: *const u8,
    source_length: usize,
    source_stride: usize,
    rows: *const u32,
    row_count: usize,
    source_byte_first: usize,
    source_byte_count: usize,
    target: *mut u8,
    target_length: usize,
    target_stride: usize,
) -> u32 {
    if row_count == 0 {
        return if target_stride == 0 && target_length == 0 && source_byte_count != 0 {
            STATUS_OK
        } else {
            STATUS_INVALID_DIMENSIONS
        };
    }
    if source.is_null() || rows.is_null() || target.is_null() {
        return STATUS_NULL_POINTER;
    }

    let rows = slice::from_raw_parts(rows, row_count);
    let layout = match validate_layout(
        source_length,
        source_stride,
        rows,
        source_byte_first,
        source_byte_count,
        target_length,
        target_stride,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source = slice::from_raw_parts(source, source_length);
    let target = slice::from_raw_parts_mut(target, target_length);
    subset_transpose(source, rows, target, layout);
    STATUS_OK
}

#[no_mangle]
/// Transpose an eight-aligned rectangle from one row-major bitmatrix into another.
///
/// Bytes outside the requested target rectangle remain unchanged.
///
/// # Safety
///
/// Non-empty buffers must be valid for their stated lengths. Source and target
/// buffers must not overlap. Invalid dimensions are reported without writing
/// the target.
pub unsafe extern "C" fn shapeit_bitmatrix_transpose_v1(
    source: *const u8,
    source_length: usize,
    source_rows: usize,
    source_stride: usize,
    max_rows: usize,
    max_cols: usize,
    target: *mut u8,
    target_length: usize,
    target_stride: usize,
) -> u32 {
    if source.is_null() || target.is_null() {
        return STATUS_NULL_POINTER;
    }
    let layout = match validate_full_layout(
        source_length,
        source_rows,
        source_stride,
        max_rows,
        max_cols,
        target_length,
        target_stride,
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source = slice::from_raw_parts(source, source_length);
    let target = slice::from_raw_parts_mut(target, target_length);
    full_transpose(source, target, layout);
    STATUS_OK
}

#[no_mangle]
/// Refresh every target haplotype row from packed common-phasing genotypes.
///
/// On the first refresh every real locus is written. Later refreshes update
/// only heterozygous and missing loci, preserving fixed homozygous and
/// scaffolded alleles exactly as the established common phaser does.
///
/// # Safety
///
/// `variants` must point to `individual_count` readable pointers, each valid
/// for `variants_length` bytes. `haplotypes` must be writable for
/// `haplotypes_length` bytes, and the input buffers must not overlap it.
/// Invalid dimensions are reported before writing.
pub unsafe extern "C" fn shapeit_bitmatrix_refresh_haplotypes_v1(
    variants: *const *const u8,
    individual_count: usize,
    variants_length: usize,
    variant_count: usize,
    first_time: u8,
    haplotypes: *mut u8,
    haplotypes_length: usize,
    haplotype_rows: usize,
    haplotype_stride: usize,
) -> u32 {
    if variants.is_null() || haplotypes.is_null() {
        return STATUS_NULL_POINTER;
    }
    if individual_count == 0
        || variant_count == 0
        || first_time > 1
        || haplotype_rows == 0
        || haplotype_rows & 7 != 0
        || haplotype_stride == 0
    {
        return STATUS_INVALID_DIMENSIONS;
    }
    let required_variants = match variant_count.checked_add(1) {
        Some(value) => value >> 1,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let required_haplotype_bytes = match variant_count.checked_add(7) {
        Some(value) => value >> 3,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let target_rows = match individual_count.checked_mul(2) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    let required_length = match haplotype_rows.checked_mul(haplotype_stride) {
        Some(value) => value,
        None => return STATUS_INTEGER_OVERFLOW,
    };
    if required_variants > variants_length
        || required_haplotype_bytes > haplotype_stride
        || target_rows > haplotype_rows
        || required_length > haplotypes_length
    {
        return STATUS_OUT_OF_BOUNDS;
    }
    let variant_views = slice::from_raw_parts(variants, individual_count);
    if variant_views.iter().any(|pointer| pointer.is_null()) {
        return STATUS_NULL_POINTER;
    }

    let haplotypes = slice::from_raw_parts_mut(haplotypes, haplotypes_length);
    for (individual, &variant_view) in variant_views.iter().enumerate() {
        let packed_variants = slice::from_raw_parts(variant_view, variants_length);
        let row0_start = individual * 2 * haplotype_stride;
        let (_, rows) = haplotypes.split_at_mut(row0_start);
        let (haplotype0, rows) = rows.split_at_mut(haplotype_stride);
        let (haplotype1, _) = rows.split_at_mut(haplotype_stride);
        refresh_sample_haplotypes(
            packed_variants,
            variant_count,
            first_time != 0,
            haplotype0,
            haplotype1,
        );
    }
    STATUS_OK
}

#[no_mangle]
/// Calculate the matching-heterozygote proportion for two diploid individuals.
///
/// The inclusive locus interval is deliberately rounded out to complete bytes,
/// matching the established common-phasing calculation.
///
/// # Safety
///
/// `source` must be readable for `source_length` bytes and `overlap` must be
/// writable. Invalid dimensions and indexes are reported without writing the
/// output.
pub unsafe extern "C" fn shapeit_bitmatrix_het_overlap_v1(
    source: *const u8,
    source_length: usize,
    source_stride: usize,
    individual0: usize,
    individual1: usize,
    start: usize,
    stop: usize,
    overlap: *mut f32,
) -> u32 {
    if source.is_null() || overlap.is_null() {
        return STATUS_NULL_POINTER;
    }
    let source = slice::from_raw_parts(source, source_length);
    *overlap =
        match heterozygote_overlap(source, source_stride, individual0, individual1, start, stop) {
            Ok(value) => value,
            Err(status) => return status,
        };
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        source: &[u8],
        source_stride: usize,
        rows: &[u32],
        source_byte_first: usize,
        source_byte_count: usize,
    ) -> Vec<u8> {
        let row_count_padded = padded_rows(rows.len()).unwrap();
        let target_stride = row_count_padded >> 3;
        let mut target = vec![0u8; source_byte_count * 8 * target_stride];
        for byte in 0..source_byte_count {
            for bit in 0..8 {
                for (target_row, &source_row) in rows.iter().enumerate() {
                    let source_value =
                        source[source_row as usize * source_stride + source_byte_first + byte];
                    target[(byte * 8 + bit) * target_stride + (target_row >> 3)] |=
                        ((source_value >> (7 - bit)) & 1) << (7 - (target_row & 7));
                }
            }
        }
        target
    }

    fn reference_full(
        source: &[u8],
        source_stride: usize,
        max_rows: usize,
        max_cols: usize,
        target_stride: usize,
        target_length: usize,
    ) -> Vec<u8> {
        let mut target = vec![0xa5; target_length];
        for col in 0..max_cols {
            for row in (0..max_rows).step_by(8) {
                let mut value = 0u8;
                for lane in 0..8 {
                    let source_byte = source[(row + lane) * source_stride + (col >> 3)];
                    value |= ((source_byte >> (7 - (col & 7))) & 1) << (7 - lane);
                }
                target[col * target_stride + (row >> 3)] = value;
            }
        }
        target
    }

    fn reference_het_overlap(
        source: &[u8],
        source_stride: usize,
        individual0: usize,
        individual1: usize,
        start: usize,
        stop: usize,
    ) -> f32 {
        let first = start >> 3;
        let count = (stop >> 3) - first + 1;
        let offsets = [
            2 * individual0 * source_stride + first,
            (2 * individual0 + 1) * source_stride + first,
            2 * individual1 * source_stride + first,
            (2 * individual1 + 1) * source_stride + first,
        ];
        let mut intersection = 0u32;
        let mut union = 0u32;
        for byte in 0..count {
            let het0 = source[offsets[0] + byte] ^ source[offsets[1] + byte];
            let het1 = source[offsets[2] + byte] ^ source[offsets[3] + byte];
            intersection += (het0 ^ het1).count_ones();
            union += (het0 | het1).count_ones();
        }
        overlap_from_counts(intersection, union)
    }

    fn pack_variant_codes(codes: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; codes.len().div_ceil(2)];
        for (locus, &code) in codes.iter().enumerate() {
            packed[locus >> 1] |= code << ((locus & 1) * 4);
        }
        packed
    }

    fn haplotype_bit(haplotypes: &[u8], stride: usize, row: usize, locus: usize) -> u8 {
        (haplotypes[row * stride + (locus >> 3)] >> (7 - (locus & 7))) & 1
    }

    #[test]
    fn whole_haplotype_refresh_preserves_fixed_loci_after_initialization() {
        const VARIANT_COUNT: usize = 9;
        const HAPLOTYPE_ROWS: usize = 8;
        const HAPLOTYPE_STRIDE: usize = 2;
        let initial_codes = [
            [0u8, 12, 6, 10, 1, 5, 9, 13, 3],
            [12u8, 0, 10, 6, 13, 9, 5, 1, 15],
        ];
        let initial: Vec<Vec<u8>> = initial_codes
            .iter()
            .map(|codes| pack_variant_codes(codes))
            .collect();
        let initial_views: Vec<*const u8> = initial.iter().map(|packed| packed.as_ptr()).collect();
        let mut haplotypes = vec![0xa5u8; HAPLOTYPE_ROWS * HAPLOTYPE_STRIDE];
        let status = unsafe {
            shapeit_bitmatrix_refresh_haplotypes_v1(
                initial_views.as_ptr(),
                initial_views.len(),
                initial[0].len(),
                VARIANT_COUNT,
                1,
                haplotypes.as_mut_ptr(),
                haplotypes.len(),
                HAPLOTYPE_ROWS,
                HAPLOTYPE_STRIDE,
            )
        };
        assert_eq!(status, STATUS_OK);
        for (individual, codes) in initial_codes.iter().enumerate() {
            for (locus, &code) in codes.iter().enumerate() {
                assert_eq!(
                    haplotype_bit(&haplotypes, HAPLOTYPE_STRIDE, individual * 2, locus),
                    (code >> 2) & 1
                );
                assert_eq!(
                    haplotype_bit(&haplotypes, HAPLOTYPE_STRIDE, individual * 2 + 1, locus),
                    (code >> 3) & 1
                );
            }
        }

        let changed_codes = [
            [12u8, 0, 10, 6, 13, 1, 5, 9, 15],
            [0u8, 12, 6, 10, 1, 5, 9, 13, 3],
        ];
        let changed: Vec<Vec<u8>> = changed_codes
            .iter()
            .map(|codes| pack_variant_codes(codes))
            .collect();
        let changed_views: Vec<*const u8> = changed.iter().map(|packed| packed.as_ptr()).collect();
        let status = unsafe {
            shapeit_bitmatrix_refresh_haplotypes_v1(
                changed_views.as_ptr(),
                changed_views.len(),
                changed[0].len(),
                VARIANT_COUNT,
                0,
                haplotypes.as_mut_ptr(),
                haplotypes.len(),
                HAPLOTYPE_ROWS,
                HAPLOTYPE_STRIDE,
            )
        };
        assert_eq!(status, STATUS_OK);
        for individual in 0..initial_codes.len() {
            for locus in 0..VARIANT_COUNT {
                let changed = changed_codes[individual][locus];
                let code = if matches!(changed & 3, 1 | 2) {
                    changed
                } else {
                    initial_codes[individual][locus]
                };
                assert_eq!(
                    haplotype_bit(&haplotypes, HAPLOTYPE_STRIDE, individual * 2, locus),
                    (code >> 2) & 1
                );
                assert_eq!(
                    haplotype_bit(&haplotypes, HAPLOTYPE_STRIDE, individual * 2 + 1, locus),
                    (code >> 3) & 1
                );
            }
        }
    }

    #[test]
    fn subset_transpose_matches_reference_across_boundaries() {
        const SOURCE_ROWS: usize = 173;
        const SOURCE_STRIDE: usize = 37;
        let mut source_storage = vec![0u8; SOURCE_ROWS * SOURCE_STRIDE + 2];
        for (index, value) in source_storage[1..=SOURCE_ROWS * SOURCE_STRIDE]
            .iter_mut()
            .enumerate()
        {
            *value = (index as u8).wrapping_mul(73).wrapping_add(41);
        }
        let source = &source_storage[1..=SOURCE_ROWS * SOURCE_STRIDE];

        for row_count in [
            1usize, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257,
        ] {
            let mut rows: Vec<u32> = (0..row_count)
                .map(|index| {
                    if index % 11 == 0 {
                        7
                    } else {
                        ((index * 37 + 11) % SOURCE_ROWS) as u32
                    }
                })
                .collect();
            if row_count & 1 == 0 {
                rows.reverse();
            }
            for (source_byte_first, source_byte_count) in [
                (0, 1),
                (1, 2),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 15),
                (6, 16),
                (7, 17),
                (6, 31),
                (SOURCE_STRIDE - 1, 1),
            ] {
                let expected = reference(
                    source,
                    SOURCE_STRIDE,
                    &rows,
                    source_byte_first,
                    source_byte_count,
                );
                let target_stride = padded_rows(rows.len()).unwrap() >> 3;
                let layout = TransposeLayout {
                    source_stride: SOURCE_STRIDE,
                    row_count_padded: padded_rows(rows.len()).unwrap(),
                    source_byte_first,
                    source_byte_count,
                    target_stride,
                };
                let mut portable = vec![0xa5; expected.len()];
                transpose_portable(source, &rows, &mut portable, layout);
                assert_eq!(portable, expected);

                let mut actual_storage = vec![0xa5; expected.len() + 2];
                let actual = &mut actual_storage[1..=expected.len()];
                subset_transpose(source, &rows, actual, layout);
                assert_eq!(actual, expected);

                let used_lanes = row_count & 7;
                if used_lanes != 0 {
                    let padding_mask = (1u8 << (8 - used_lanes)) - 1;
                    for target_row in 0..source_byte_count * 8 {
                        assert_eq!(
                            actual[target_row * target_stride + target_stride - 1] & padding_mask,
                            0
                        );
                    }
                }
                assert_eq!(actual_storage[0], 0xa5);
                assert_eq!(actual_storage[expected.len() + 1], 0xa5);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_swar_core_matches_portable_for_every_lane_byte() {
        // A bit transpose is linear. Exhausting all 256 values independently
        // in every input lane therefore covers every input bit and all
        // within-lane combinations without enumerating 2^64 words.
        for lane in 0..8 {
            for value in 0u64..=u8::MAX.into() {
                let packed = value << ((7 - lane) * 8);
                assert_eq!(
                    transpose_word_aarch64(packed).to_be_bytes(),
                    transpose_word_portable(packed)
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_neon_byte_tile_matches_portable_columns() {
        const STRIDE: usize = 13;
        let mut source = vec![0u8; 1 + 8 * STRIDE];
        for row in 0..8 {
            for byte in 0..8 {
                source[1 + row * STRIDE + byte] = (row as u8)
                    .wrapping_mul(53)
                    .wrapping_add((byte as u8).wrapping_mul(97))
                    .wrapping_add(19);
            }
        }
        let offsets = core::array::from_fn(|row| 1 + row * STRIDE);
        let actual = unsafe { transpose_byte_tile_aarch64(source.as_ptr(), offsets) };
        for byte in 0..8 {
            let mut packed = 0u64;
            for row in 0..8 {
                packed |= u64::from(source[1 + row * STRIDE + byte]) << ((7 - row) * 8);
            }
            assert_eq!(
                actual[byte],
                u64::from_be_bytes(transpose_word_portable(packed))
            );
        }
    }

    #[test]
    fn full_transpose_matches_independent_reference_and_preserves_remainder() {
        for (source_rows, source_stride, max_rows, max_cols, target_padding) in [
            (8, 1, 8, 8, 0),
            (24, 3, 16, 16, 2),
            (72, 9, 64, 56, 1),
            (72, 9, 72, 72, 3),
            (136, 17, 128, 120, 1),
            (136, 17, 136, 128, 2),
        ] {
            let mut source_storage = vec![0u8; source_rows * source_stride + 2];
            for (index, value) in source_storage[1..=source_rows * source_stride]
                .iter_mut()
                .enumerate()
            {
                *value = (index as u8).wrapping_mul(73).wrapping_add(41);
            }
            let source = &source_storage[1..=source_rows * source_stride];
            let target_stride = (max_rows >> 3) + target_padding;
            let target_length = max_cols * target_stride + 5;
            let expected = reference_full(
                source,
                source_stride,
                max_rows,
                max_cols,
                target_stride,
                target_length,
            );
            let layout = FullTransposeLayout {
                source_stride,
                max_rows,
                max_cols,
                target_stride,
            };
            let mut portable = vec![0xa5; target_length];
            full_transpose_portable(source, &mut portable, layout);
            assert_eq!(portable, expected);

            let mut actual_storage = vec![0xa5; target_length + 2];
            let actual = &mut actual_storage[1..=target_length];
            let status = unsafe {
                shapeit_bitmatrix_transpose_v1(
                    source.as_ptr(),
                    source.len(),
                    source_rows,
                    source_stride,
                    max_rows,
                    max_cols,
                    actual.as_mut_ptr(),
                    actual.len(),
                    target_stride,
                )
            };
            assert_eq!(status, STATUS_OK);
            assert_eq!(actual, expected);
            assert_eq!(actual_storage[0], 0xa5);
            assert_eq!(actual_storage[target_length + 1], 0xa5);
        }
    }

    #[test]
    fn het_overlap_matches_byte_reference_across_word_boundaries() {
        const INDIVIDUALS: usize = 6;
        for source_stride in [1usize, 7, 8, 9, 33] {
            let source: Vec<u8> = (0..2 * INDIVIDUALS * source_stride)
                .map(|index| (index as u8).wrapping_mul(73).wrapping_add(41))
                .collect();
            let final_bit = source_stride * 8 - 1;
            for (individual0, individual1) in [(0, 1), (2, 5), (4, 4)] {
                for (start, stop) in [
                    (0, 0),
                    (1, core::cmp::min(7, final_bit)),
                    (3, core::cmp::min(8, final_bit)),
                    (0, final_bit),
                ] {
                    let expected = reference_het_overlap(
                        &source,
                        source_stride,
                        individual0,
                        individual1,
                        start,
                        stop,
                    );
                    let mut actual = -1.0f32;
                    let status = unsafe {
                        shapeit_bitmatrix_het_overlap_v1(
                            source.as_ptr(),
                            source.len(),
                            source_stride,
                            individual0,
                            individual1,
                            start,
                            stop,
                            &mut actual,
                        )
                    };
                    assert_eq!(status, STATUS_OK);
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn het_overlap_rejects_bad_layout_without_writing() {
        let source = [0u8; 16];
        let mut overlap = 3.25f32;
        let status = unsafe {
            shapeit_bitmatrix_het_overlap_v1(
                source.as_ptr(),
                source.len(),
                2,
                0,
                4,
                0,
                15,
                &mut overlap,
            )
        };
        assert_eq!(status, STATUS_OUT_OF_BOUNDS);
        assert_eq!(overlap, 3.25);

        let status = unsafe {
            shapeit_bitmatrix_het_overlap_v1(
                source.as_ptr(),
                source.len(),
                2,
                0,
                1,
                9,
                8,
                &mut overlap,
            )
        };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
        assert_eq!(overlap, 3.25);
    }

    #[test]
    fn layout_validation_rejects_bad_indexes_and_sizes() {
        let rows = [0u32, 3];
        assert_eq!(
            validate_layout(24, 8, &rows, 0, 2, 8, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_layout(32, 8, &rows, 7, 2, 16, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_layout(32, 8, &rows, 0, 2, 15, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_layout(32, 8, &rows, 0, 2, 16, 2),
            Err(STATUS_INVALID_DIMENSIONS)
        );
    }

    #[test]
    fn empty_subset_preserves_public_zero_row_contract() {
        let status = unsafe {
            shapeit_bitmatrix_subset_transpose_v1(
                core::ptr::null(),
                0,
                0,
                core::ptr::null(),
                0,
                0,
                1,
                core::ptr::null_mut(),
                0,
                0,
            )
        };
        assert_eq!(status, STATUS_OK);

        let status = unsafe {
            shapeit_bitmatrix_subset_transpose_v1(
                core::ptr::null(),
                0,
                0,
                core::ptr::null(),
                0,
                0,
                0,
                core::ptr::null_mut(),
                0,
                0,
            )
        };
        assert_eq!(status, STATUS_INVALID_DIMENSIONS);
    }

    #[test]
    fn full_layout_validation_rejects_bad_dimensions_without_writing() {
        assert_eq!(
            validate_full_layout(71, 8, 9, 8, 72, 72, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_full_layout(72, 9, 8, 8, 72, 72, 1),
            Err(STATUS_INVALID_DIMENSIONS)
        );
        assert_eq!(
            validate_full_layout(72, 8, 9, 16, 72, 72, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_full_layout(72, 8, 9, 8, 80, 80, 1),
            Err(STATUS_OUT_OF_BOUNDS)
        );
        assert_eq!(
            validate_full_layout(usize::MAX, usize::MAX & !7, 9, 8, 8, 8, 1),
            Err(STATUS_INTEGER_OVERFLOW)
        );

        let source = [0u8; 8];
        let mut target = [0xa5u8; 8];
        let status = unsafe {
            shapeit_bitmatrix_transpose_v1(
                source.as_ptr(),
                source.len(),
                8,
                1,
                8,
                8,
                target.as_mut_ptr(),
                target.len() - 1,
                1,
            )
        };
        assert_eq!(status, STATUS_OUT_OF_BOUNDS);
        assert_eq!(target, [0xa5; 8]);
    }
}
