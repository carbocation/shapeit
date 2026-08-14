#ifndef SHAPEIT_RNG_H
#define SHAPEIT_RNG_H

#include <stdint.h>

#define SHAPEIT_RNG_ABI_VERSION 1U

#define SHAPEIT_RNG_DOMAIN_SERIAL 0U
#define SHAPEIT_RNG_DOMAIN_PHASE_COMMON_PBWT_SITE 1U
#define SHAPEIT_RNG_DOMAIN_PHASE_COMMON_WINDOW 2U
#define SHAPEIT_RNG_DOMAIN_PHASE_COMMON_MCMC 3U
#define SHAPEIT_RNG_DOMAIN_PHASE_RARE_SELECTION_ORDER 4U
#define SHAPEIT_RNG_DOMAIN_PHASE_RARE_PBWT_SITE 5U
#define SHAPEIT_RNG_DOMAIN_PHASE_RARE_FALLBACK 6U
#define SHAPEIT_RNG_DOMAIN_PHASE_RARE_SOLVE_ORDER 7U

#ifdef __cplusplus
extern "C" {
#endif

uint32_t shapeit_rng_abi_version(void);

void shapeit_rng_philox4x32_10_raw(
    const uint32_t * counter,
    const uint32_t * key,
    uint32_t * output);

void shapeit_rng_block_v1(
    uint64_t seed,
    uint32_t domain,
    uint32_t iteration,
    uint64_t item,
    uint64_t block,
    uint32_t * output);

#ifdef __cplusplus
}
#endif

#endif
