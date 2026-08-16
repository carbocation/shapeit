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

typedef struct shapeit_common_phase_job_v1 {
    uint32_t abi_version;
    size_t struct_size;
    shapeit_conditioning_graph_build_v1 conditioning;
    shapeit_hmm_phase_job_v1 phase;
} shapeit_common_phase_job_v1;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_common_phase_job_run_v1(
    const shapeit_common_phase_job_v1 * parameters,
    shapeit_conditioning_job_v1 ** job,
    shapeit_hmm_job_result_v1 * result);

#ifdef __cplusplus
}
#endif

#endif
