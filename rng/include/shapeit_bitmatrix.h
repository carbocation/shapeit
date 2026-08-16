#ifndef SHAPEIT_BITMATRIX_H
#define SHAPEIT_BITMATRIX_H

#include <stddef.h>
#include <stdint.h>

#define SHAPEIT_BITMATRIX_ABI_VERSION 1U

#define SHAPEIT_BITMATRIX_STATUS_OK 0U
#define SHAPEIT_BITMATRIX_STATUS_NULL_POINTER 1U
#define SHAPEIT_BITMATRIX_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_BITMATRIX_STATUS_INTEGER_OVERFLOW 4U

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_bitmatrix_abi_version(void);

uint32_t shapeit_bitmatrix_subset_transpose_v1(
    const uint8_t * source,
    size_t source_length,
    size_t source_stride,
    const uint32_t * rows,
    size_t row_count,
    size_t source_byte_first,
    size_t source_byte_count,
    uint8_t * target,
    size_t target_length,
    size_t target_stride);

uint32_t shapeit_bitmatrix_transpose_v1(
    const uint8_t * source,
    size_t source_length,
    size_t source_rows,
    size_t source_stride,
    size_t max_rows,
    size_t max_cols,
    uint8_t * target,
    size_t target_length,
    size_t target_stride);

uint32_t shapeit_bitmatrix_het_overlap_v1(
    const uint8_t * source,
    size_t source_length,
    size_t source_stride,
    size_t individual0,
    size_t individual1,
    size_t start,
    size_t stop,
    float * overlap);

#ifdef __cplusplus
}
#endif

#endif
