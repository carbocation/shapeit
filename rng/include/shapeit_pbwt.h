#ifndef SHAPEIT_PBWT_H
#define SHAPEIT_PBWT_H

#include <stddef.h>
#include <stdint.h>

#include <shapeit_ibd2.h>

#define SHAPEIT_PBWT_ABI_VERSION 1U

#define SHAPEIT_PBWT_STATUS_OK 0U
#define SHAPEIT_PBWT_STATUS_NULL_POINTER 1U
#define SHAPEIT_PBWT_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_PBWT_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_PBWT_STATUS_INTEGER_OVERFLOW 4U
#define SHAPEIT_PBWT_STATUS_INSUFFICIENT_STATES 5U

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_pbwt_abi_version(void);

uint32_t shapeit_pbwt_select_sites_v1(
    const uint8_t * evaluated_sites,
    size_t evaluated_sites_length,
    const int32_t * site_groups,
    size_t site_groups_length,
    size_t group_count,
    uint64_t seed,
    uint32_t domain,
    uint32_t iteration,
    uint8_t * selected_sites,
    size_t selected_sites_length);

uint32_t shapeit_pbwt_solve_chunk_v1(
    uint8_t * haplotypes,
    size_t haplotypes_length,
    size_t haplotype_stride,
    size_t site_count,
    size_t haplotype_count,
    const uint8_t * const * genotype_variants,
    size_t individual_count,
    size_t genotype_variants_length,
    const int32_t * site_chunks,
    size_t site_chunks_length,
    size_t chunk,
    size_t buffer_start,
    const uint8_t * buffer,
    size_t buffer_length,
    const float * scores,
    size_t scores_length);

uint32_t shapeit_pbwt_select_chunk_v1(
    const uint8_t * haplotypes,
    size_t haplotypes_length,
    size_t haplotype_stride,
    size_t site_count,
    size_t haplotype_count,
    size_t target_individual_count,
    const uint8_t * evaluated_sites,
    size_t evaluated_sites_length,
    const uint8_t * selected_sites,
    size_t selected_sites_length,
    const int32_t * site_groups,
    size_t site_groups_length,
    size_t group_count,
    const int32_t * site_chunks,
    size_t site_chunks_length,
    size_t chunk,
    size_t buffer_start,
    size_t depth,
    const shapeit_ibd2_tracks_v1 * ibd2,
    int32_t * neighbors,
    size_t neighbors_length);

uint32_t shapeit_pbwt_transpose_neighbors_v1(
    int32_t * neighbors,
    size_t neighbors_length,
    size_t target_haplotype_count,
    size_t group_count,
    size_t depth);

#ifdef __cplusplus
}
#endif

#endif
