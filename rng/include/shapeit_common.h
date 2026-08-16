#ifndef SHAPEIT_COMMON_H
#define SHAPEIT_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include <shapeit_conditioning.h>
#include <shapeit_hmm.h>

#define SHAPEIT_COMMON_ABI_VERSION 1U

#define SHAPEIT_COMMON_STATUS_OK 0U
#define SHAPEIT_COMMON_STATUS_NULL_POINTER 1U
#define SHAPEIT_COMMON_STATUS_INVALID_DIMENSIONS 2U
#define SHAPEIT_COMMON_STATUS_OUT_OF_BOUNDS 3U
#define SHAPEIT_COMMON_STATUS_INTEGER_OVERFLOW 4U
#define SHAPEIT_COMMON_STATUS_INSUFFICIENT_STATES 5U
#define SHAPEIT_COMMON_STATUS_THREAD_FAILURE 6U

typedef struct shapeit_common_phase_job_v1 {
    uint32_t abi_version;
    size_t struct_size;
    shapeit_conditioning_graph_build_v1 conditioning;
    shapeit_hmm_phase_job_v1 phase;
} shapeit_common_phase_job_v1;

typedef struct shapeit_common_workers_v1 shapeit_common_workers_v1;

typedef void (*shapeit_common_progress_v1)(
    size_t completed,
    size_t total,
    void * context);

typedef struct shapeit_common_iteration_v1 {
    uint32_t abi_version;
    size_t struct_size;
    shapeit_common_phase_job_v1 sample_template;
    const int32_t * base_pair_positions;
    size_t base_pair_positions_length;
    shapeit_ibd2_tracks_v1 * ibd2_registry;
    shapeit_common_progress_v1 progress;
    void * progress_context;
} shapeit_common_iteration_v1;

typedef struct shapeit_common_iteration_result_v1 {
    uint64_t underflow_recovered_summing;
    uint64_t underflow_recovered_precision;
    int32_t fatal_outcome;
    size_t failed_sample;
    size_t windows;
    double conditioning_states_mean;
    double conditioning_states_sd;
    double window_megabases_mean;
    double window_megabases_sd;
} shapeit_common_iteration_result_v1;

typedef struct shapeit_common_fallback_v1 {
    size_t sample;
    size_t window;
    size_t states;
} shapeit_common_fallback_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_common_phase_job_run_v1(
    const shapeit_common_phase_job_v1 * parameters,
    shapeit_conditioning_job_v1 ** job,
    shapeit_hmm_job_result_v1 * result);

uint32_t shapeit_common_workers_create_v1(
    size_t worker_count,
    shapeit_genotype_graph_v1 * const * graphs,
    size_t graph_count,
    const uint8_t * haploid_individuals,
    size_t haploid_individuals_length,
    shapeit_common_workers_v1 ** workers);

void shapeit_common_workers_free_v1(shapeit_common_workers_v1 * workers);

uint32_t shapeit_common_workers_run_iteration_v1(
    shapeit_common_workers_v1 * workers,
    const shapeit_common_iteration_v1 * parameters,
    shapeit_common_iteration_result_v1 * result);

size_t shapeit_common_workers_fallback_count_v1(
    const shapeit_common_workers_v1 * workers);

uint32_t shapeit_common_workers_fallback_v1(
    const shapeit_common_workers_v1 * workers,
    size_t index,
    shapeit_common_fallback_v1 * fallback);

#ifdef __cplusplus
}
#endif

#endif
