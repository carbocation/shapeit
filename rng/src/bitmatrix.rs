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

#[inline]
fn multiply_high(lhs: u32, rhs: u32) -> u32 {
    ((u64::from(lhs) * u64::from(rhs)) >> 32) as u32
}

#[inline]
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

#[cfg(target_arch = "x86_64")]
#[inline]
fn bmi2_available() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    // CPUID is available on every x86_64 processor. Leaf 7 is queried only
    // when the maximum supported basic leaf includes it.
    __cpuid(0).eax >= 7 && (__cpuid_count(7, 0).ebx & (1 << 8)) != 0
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
    full_transpose_portable(source, target, layout);
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
        transpose_portable(
            source,
            rows,
            &mut target,
            TransposeLayout {
                source_stride,
                row_count_padded,
                source_byte_first,
                source_byte_count,
                target_stride,
            },
        );
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

    #[test]
    fn subset_transpose_matches_reference_across_boundaries() {
        const SOURCE_ROWS: usize = 137;
        const SOURCE_STRIDE: usize = 19;
        let source: Vec<u8> = (0..SOURCE_ROWS * SOURCE_STRIDE)
            .map(|index| (index as u8).wrapping_mul(73).wrapping_add(41))
            .collect();

        for row_count in [1usize, 7, 8, 9, 63, 64, 65, 129] {
            let rows: Vec<u32> = (0..row_count)
                .map(|index| ((index * 37 + 11) % SOURCE_ROWS) as u32)
                .collect();
            for (source_byte_first, source_byte_count) in [(0, 1), (1, 8), (7, 12)] {
                let expected = reference(
                    &source,
                    SOURCE_STRIDE,
                    &rows,
                    source_byte_first,
                    source_byte_count,
                );
                let target_stride = padded_rows(rows.len()).unwrap() >> 3;
                let mut actual = vec![0xa5; expected.len()];
                subset_transpose(
                    &source,
                    &rows,
                    &mut actual,
                    TransposeLayout {
                        source_stride: SOURCE_STRIDE,
                        row_count_padded: padded_rows(rows.len()).unwrap(),
                        source_byte_first,
                        source_byte_count,
                        target_stride,
                    },
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn full_transpose_matches_independent_reference_and_preserves_remainder() {
        for (source_rows, source_stride, max_rows, max_cols) in [
            (8, 1, 8, 8),
            (16, 3, 8, 16),
            (72, 9, 64, 56),
            (72, 9, 72, 72),
        ] {
            let source: Vec<u8> = (0..source_rows * source_stride)
                .map(|index| (index as u8).wrapping_mul(73).wrapping_add(41))
                .collect();
            let target_stride = source_rows >> 3;
            let target_length = source_stride * 8 * target_stride;
            let expected = reference_full(
                &source,
                source_stride,
                max_rows,
                max_cols,
                target_stride,
                target_length,
            );
            let mut actual = vec![0xa5; target_length];
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
        }
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
