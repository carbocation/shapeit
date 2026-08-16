#ifndef SHAPEIT_CONDITIONING_H
#define SHAPEIT_CONDITIONING_H

#include <stddef.h>
#include <stdint.h>

#include <shapeit_genotype.h>

#define SHAPEIT_CONDITIONING_ABI_VERSION 1U

#define SHAPEIT_CONDITIONING_STATUS_OK 0U
#define SHAPEIT_CONDITIONING_STATUS_NULL_POINTER 1U
#define SHAPEIT_CONDITIONING_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_CONDITIONING_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_CONDITIONING_STATUS_INTEGER_OVERFLOW 4U
#define SHAPEIT_CONDITIONING_STATUS_INSUFFICIENT_STATES 5U

typedef struct shapeit_conditioning_build_v1 {
    uint32_t abi_version;
    size_t struct_size;

    const uint8_t * variants;
    size_t variants_length;
    size_t variant_count;
    const uint64_t * diplotypes;
    size_t diplotypes_length;
    const uint16_t * segment_lengths;
    size_t segment_lengths_length;
    const double * segment_start_centimorgans;
    size_t segment_start_centimorgans_length;
    const double * segment_stop_centimorgans;
    size_t segment_stop_centimorgans_length;
    float minimum_window_centimorgans;

    const uint8_t * selected_sites;
    size_t selected_sites_length;
    const int32_t * site_grouping;
    size_t site_grouping_length;
    const int32_t * pbwt_neighbors;
    size_t pbwt_neighbors_length;
    size_t pbwt_depth;
    size_t pbwt_group_count;

    size_t target_individual;
    size_t target_individual_count;
    size_t haplotype_count;
    const uint8_t * haploid_individuals;
    size_t haploid_individuals_length;

    const uint8_t * haplotypes;
    size_t haplotypes_length;
    size_t haplotype_stride;
    float maximum_heterozygote_mismatch;

    uint64_t window_seed;
    uint32_t window_domain;
    uint32_t window_iteration;
    uint64_t window_item;
    uint64_t fallback_seed;
    uint32_t fallback_domain;
    uint32_t fallback_iteration;
    uint64_t fallback_item;
} shapeit_conditioning_build_v1;

typedef struct shapeit_conditioning_track_v1 {
    int32_t individual;
    int32_t from;
    int32_t to;
} shapeit_conditioning_track_v1;

typedef struct shapeit_conditioning_job_v1 shapeit_conditioning_job_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_conditioning_abi_version(void);

uint32_t shapeit_conditioning_job_build_v1(
    const shapeit_conditioning_build_v1 * parameters,
    /* Null allocates a job; a live job is rebuilt while retaining workspace. */
    shapeit_conditioning_job_v1 ** job);

void shapeit_conditioning_job_free_v1(shapeit_conditioning_job_v1 * job);

size_t shapeit_conditioning_job_window_count_v1(
    const shapeit_conditioning_job_v1 * job);

uint32_t shapeit_conditioning_job_window_v1(
    const shapeit_conditioning_job_v1 * job,
    size_t index,
    shapeit_genotype_window_v1 * window,
    const uint32_t ** states,
    size_t * states_length,
    uint8_t * used_fallback);

uint32_t shapeit_conditioning_job_tracks_v1(
    const shapeit_conditioning_job_v1 * job,
    const shapeit_conditioning_track_v1 ** tracks,
    size_t * tracks_length);

#ifdef __cplusplus
}
#endif

#endif
