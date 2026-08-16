#ifndef SHAPEIT_GENOTYPE_H
#define SHAPEIT_GENOTYPE_H

#include <stddef.h>
#include <stdint.h>

#define SHAPEIT_GENOTYPE_ABI_VERSION 1U

#define SHAPEIT_GENOTYPE_STATUS_OK 0U
#define SHAPEIT_GENOTYPE_STATUS_NULL_POINTER 1U
#define SHAPEIT_GENOTYPE_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_GENOTYPE_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_GENOTYPE_STATUS_INTEGER_OVERFLOW 4U

typedef struct shapeit_genotype_window_v1 {
    int32_t start_locus;
    int32_t start_segment;
    int32_t start_ambiguous;
    int32_t start_missing;
    int32_t start_transition;
    int32_t stop_locus;
    int32_t stop_segment;
    int32_t stop_ambiguous;
    int32_t stop_missing;
    int32_t stop_transition;
} shapeit_genotype_window_v1;

typedef struct shapeit_genotype_storage_view_v1 {
    size_t transition_count;
    const uint8_t * transition_mask;
    size_t transition_mask_length;
    const float * transition_probabilities;
    size_t transition_probabilities_length;
    const float * missing_probabilities;
    size_t missing_probabilities_length;
    uint32_t storage_events;
} shapeit_genotype_storage_view_v1;

typedef struct shapeit_genotype_storage_v1 shapeit_genotype_storage_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_genotype_abi_version(void);

uint32_t shapeit_genotype_graph_sizes_v1(
    const uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    size_t * segment_count,
    size_t * ambiguous_count,
    size_t * missing_count);

uint32_t shapeit_genotype_graph_build_v1(
    const uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    uint16_t * segment_lengths,
    size_t segment_lengths_length,
    uint8_t * ambiguous,
    size_t ambiguous_length,
    uint64_t * diplotypes,
    size_t diplotypes_length,
    uint32_t * transition_count);

uint32_t shapeit_genotype_sample_v1(
    uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    const uint8_t * ambiguous,
    size_t ambiguous_length,
    const uint64_t * diplotypes,
    size_t diplotypes_length,
    const uint16_t * segment_lengths,
    size_t segment_lengths_length,
    const double * transition_probabilities,
    size_t transition_probabilities_length,
    const float * missing_probabilities,
    size_t missing_probabilities_length,
    uint8_t haploid,
    uint64_t seed,
    uint32_t domain,
    uint32_t iteration,
    uint64_t item);

uint32_t shapeit_genotype_solve_v1(
    uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    const uint8_t * ambiguous,
    size_t ambiguous_length,
    const uint64_t * diplotypes,
    size_t diplotypes_length,
    const uint16_t * segment_lengths,
    size_t segment_lengths_length,
    const uint32_t * stored_transition_indexes,
    size_t stored_transition_indexes_length,
    const float * stored_transition_probabilities,
    size_t stored_transition_probabilities_length,
    const float * missing_probabilities,
    size_t missing_probabilities_length,
    uint8_t haploid,
    uint32_t storage_events);

uint32_t shapeit_genotype_storage_update_v1(
    shapeit_genotype_storage_v1 ** storage,
    const double * transition_probabilities,
    size_t transition_count,
    const float * missing_probabilities,
    size_t missing_probabilities_length);

void shapeit_genotype_storage_free_v1(shapeit_genotype_storage_v1 * storage);

uint32_t shapeit_genotype_storage_borrow_v1(
    const shapeit_genotype_storage_v1 * storage,
    shapeit_genotype_storage_view_v1 * view);

uint32_t shapeit_genotype_solve_storage_v1(
    uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    const uint8_t * ambiguous,
    size_t ambiguous_length,
    const uint64_t * diplotypes,
    size_t diplotypes_length,
    const uint16_t * segment_lengths,
    size_t segment_lengths_length,
    const shapeit_genotype_storage_v1 * storage,
    uint8_t haploid);

uint32_t shapeit_genotype_windows_v1(
    const uint8_t * variants,
    size_t variants_length,
    size_t variant_count,
    const uint64_t * diplotypes,
    size_t diplotypes_length,
    const uint16_t * segment_lengths,
    size_t segment_lengths_length,
    const double * segment_start_centimorgans,
    size_t segment_start_centimorgans_length,
    const double * segment_stop_centimorgans,
    size_t segment_stop_centimorgans_length,
    float minimum_window_centimorgans,
    uint64_t seed,
    uint32_t domain,
    uint32_t iteration,
    uint64_t item,
    shapeit_genotype_window_v1 * windows,
    size_t windows_capacity,
    size_t * windows_length);

#ifdef __cplusplus
}
#endif

#endif
