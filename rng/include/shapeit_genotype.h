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

#ifdef __cplusplus
}
#endif

#endif
