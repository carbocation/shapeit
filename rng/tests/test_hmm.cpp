#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <shapeit_hmm.h>

static void test_single_locus_segment() {
	const uint8_t variants[] = {0};
	const uint16_t lengths[] = {1};
	const uint64_t diplotypes[] = {(uint64_t(1) << 0) | (uint64_t(1) << 9)};
	const uint8_t haplotypes[] = {
		0b01010101, 0b01010101, 0b01010101, 0b01010101,
		0b01010101, 0b01010101, 0b01010101, 0b01010101};
	const float centimorgans[] = {0.0f};
	const int8_t rare_alleles[] = {-1};
	double transitions[] = {0.0, 0.0};

	size_t scratch_length = 0;
	assert(shapeit_hmm_double_scratch_len_v1(8, 1, 0, &scratch_length) ==
		SHAPEIT_HMM_STATUS_OK);
	std::vector < double > scratch(scratch_length, 0.0);
	int32_t alpha_locus[] = {0};

	shapeit_hmm_segment_double_v1 parameters = {};
	parameters.abi_version = SHAPEIT_HMM_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);
	parameters.variants = variants;
	parameters.variants_length = 1;
	parameters.segment_lengths = lengths;
	parameters.segment_lengths_length = 1;
	parameters.diplotypes = diplotypes;
	parameters.diplotypes_length = 1;
	parameters.haplotypes = haplotypes;
	parameters.haplotypes_length = 8;
	parameters.haplotype_stride = 1;
	parameters.conditioning_haplotypes = 8;
	parameters.centimorgans = centimorgans;
	parameters.centimorgans_length = 1;
	parameters.rare_alleles = rare_alleles;
	parameters.rare_alleles_length = 1;
	parameters.effective_population_size = 15000;
	parameters.total_haplotypes = 16;
	parameters.emission_match = static_cast<double>(0.9999f);
	parameters.emission_mismatch = static_cast<double>(0.0001f);
	parameters.segment_first = 0;
	parameters.segment_last = 0;
	parameters.locus_first = 0;
	parameters.locus_last = 0;
	parameters.ambiguous_first = 0;
	parameters.ambiguous_last = -1;
	parameters.missing_first = 0;
	parameters.missing_last = -1;
	parameters.transition_first = 2;
	parameters.transition_last = 1;
	parameters.transition_probabilities = transitions;
	parameters.transition_probabilities_length = 2;
	parameters.scratch = scratch.data();
	parameters.scratch_length = scratch.size();
	parameters.alpha_locus_scratch = alpha_locus;
	parameters.alpha_locus_scratch_length = 1;

	int32_t outcome = -99;
	assert(shapeit_hmm_run_segment_double_v1(&parameters, &outcome) ==
		SHAPEIT_HMM_STATUS_OK);
	assert(outcome == 0);
	assert(transitions[0] == 0.5);
	assert(transitions[1] == 0.5);
}

int main() {
	assert(shapeit_hmm_abi_version() == SHAPEIT_HMM_ABI_VERSION);
	test_single_locus_segment();
	return 0;
}
