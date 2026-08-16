#ifndef SHAPEIT_IBD2_H
#define SHAPEIT_IBD2_H

#include <stddef.h>
#include <stdint.h>

#define SHAPEIT_IBD2_ABI_VERSION 1U

#define SHAPEIT_IBD2_STATUS_OK 0U
#define SHAPEIT_IBD2_STATUS_NULL_POINTER 1U
#define SHAPEIT_IBD2_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_IBD2_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_IBD2_STATUS_INTEGER_OVERFLOW 4U

typedef struct shapeit_ibd2_track_v1 {
    int32_t individual;
    int32_t from;
    int32_t to;
} shapeit_ibd2_track_v1;

typedef struct shapeit_ibd2_stats_v1 {
    size_t individuals;
    size_t tracks;
    size_t merged;
} shapeit_ibd2_stats_v1;

typedef struct shapeit_ibd2_tracks_v1 shapeit_ibd2_tracks_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_ibd2_abi_version(void);

uint32_t shapeit_ibd2_tracks_create_v1(
    size_t individual_count,
    const float * centimorgans,
    size_t centimorgans_length,
    shapeit_ibd2_tracks_v1 ** registry);

void shapeit_ibd2_tracks_free_v1(shapeit_ibd2_tracks_v1 * registry);

uint32_t shapeit_ibd2_tracks_push_v1(
    shapeit_ibd2_tracks_v1 * registry,
    size_t source_individual,
    const shapeit_ibd2_track_v1 * tracks,
    size_t tracks_length);

uint32_t shapeit_ibd2_tracks_collapse_v1(
    shapeit_ibd2_tracks_v1 * registry,
    shapeit_ibd2_stats_v1 * stats);

#ifdef __cplusplus
}
#endif

#endif
