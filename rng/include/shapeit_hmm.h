#ifndef SHAPEIT_HMM_H
#define SHAPEIT_HMM_H

#include <stddef.h>
#include <stdint.h>

#define SHAPEIT_HMM_ABI_VERSION 1U

#define SHAPEIT_HMM_STATUS_OK 0U
#define SHAPEIT_HMM_STATUS_NULL_POINTER 1U
#define SHAPEIT_HMM_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_HMM_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_HMM_STATUS_INTEGER_OVERFLOW 4U

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;

    const uint8_t * variants;
    size_t variants_length;
    const uint8_t * ambiguous;
    size_t ambiguous_length;
    const uint16_t * segment_lengths;
    size_t segment_lengths_length;
    const uint64_t * diplotypes;
    size_t diplotypes_length;

    const uint8_t * haplotypes;
    size_t haplotypes_length;
    size_t haplotype_stride;
    size_t conditioning_haplotypes;
    uint32_t locus_offset;

    const float * centimorgans;
    size_t centimorgans_length;
    const float * recombination;
    size_t recombination_length;
    const int8_t * rare_alleles;
    size_t rare_alleles_length;
    int32_t effective_population_size;
    int32_t total_haplotypes;
    double emission_match;
    double emission_mismatch;

    int32_t segment_first;
    int32_t segment_last;
    int32_t locus_first;
    int32_t locus_last;
    int32_t ambiguous_first;
    int32_t ambiguous_last;
    int32_t missing_first;
    int32_t missing_last;
    int32_t transition_first;
    int32_t transition_last;

    double * transition_probabilities;
    size_t transition_probabilities_length;
    float * missing_probabilities;
    size_t missing_probabilities_length;

    double * scratch;
    size_t scratch_length;
    int32_t * alpha_locus_scratch;
    size_t alpha_locus_scratch_length;
} shapeit_hmm_segment_double_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_hmm_abi_version(void);

uint32_t shapeit_hmm_double_scratch_len_v1(
    size_t conditioning_haplotypes,
    size_t segment_count,
    size_t missing_count,
    size_t * scratch_length);

uint32_t shapeit_hmm_run_segment_double_v1(
    const shapeit_hmm_segment_double_v1 * parameters,
    int32_t * outcome);

#ifdef __cplusplus
}
#endif

#endif
