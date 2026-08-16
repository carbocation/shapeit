/*******************************************************************************
 * Copyright (C) 2022-2023 Olivier Delaneau
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include <models/haplotype_segment_double_rust.h>

#include <shapeit_hmm.h>
#include <algorithm>
#include <boost/align/aligned_allocator.hpp>
#include <cstdint>
#include <stdexcept>

using namespace std;

int run_haplotype_segment_double_rust(
	genotype * G, bitmatrix & H, vector < unsigned int > & conditioning_haplotypes,
	window & W, hmm_parameters & M, vector < double > & transition_probabilities,
	vector < float > & missing_probabilities) {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));

	bitmatrix Hvar;
	const int locus_offset = Hvar.subsetTranspose(
		H, conditioning_haplotypes, W.start_locus, W.stop_locus);
	const size_t segment_count = W.stop_segment - W.start_segment + 1;
	const size_t missing_count = std::max(0, W.stop_missing - W.start_missing + 1);
	size_t scratch_length = 0;
	uint32_t status = shapeit_hmm_double_scratch_len_v1(
		conditioning_haplotypes.size(), segment_count, missing_count, &scratch_length);
	if (status != SHAPEIT_HMM_STATUS_OK) {
		throw runtime_error("Rust HMM rejected scratch dimensions (status " +
			to_string(status) + ")");
	}
	vector < double > scratch(scratch_length, 0.0);
	vector < int32_t > alpha_locus(segment_count, 0);

	shapeit_hmm_segment_double_v1 parameters = {};
	parameters.abi_version = SHAPEIT_HMM_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);
	parameters.variants = G->Variants.data();
	parameters.variants_length = G->Variants.size();
	parameters.ambiguous = G->Ambiguous.data();
	parameters.ambiguous_length = G->Ambiguous.size();
	parameters.segment_lengths = G->Lengths.data();
	parameters.segment_lengths_length = G->Lengths.size();
	parameters.diplotypes = reinterpret_cast<const uint64_t *>(G->Diplotypes.data());
	parameters.diplotypes_length = G->Diplotypes.size();
	parameters.haplotypes = Hvar.bytes;
	parameters.haplotypes_length = Hvar.n_bytes;
	parameters.haplotype_stride = Hvar.n_cols >> 3;
	parameters.conditioning_haplotypes = conditioning_haplotypes.size();
	parameters.locus_offset = locus_offset;
	parameters.centimorgans = M.cm.data();
	parameters.centimorgans_length = M.cm.size();
	parameters.recombination = M.t.data();
	parameters.recombination_length = M.t.size();
	parameters.rare_alleles = reinterpret_cast<const int8_t *>(M.rare_allele.data());
	parameters.rare_alleles_length = M.rare_allele.size();
	parameters.effective_population_size = M.Neff;
	parameters.total_haplotypes = M.Nhap;
	parameters.emission_match = M.ee;
	parameters.emission_mismatch = M.ed;
	parameters.segment_first = W.start_segment;
	parameters.segment_last = W.stop_segment;
	parameters.locus_first = W.start_locus;
	parameters.locus_last = W.stop_locus;
	parameters.ambiguous_first = W.start_ambiguous;
	parameters.ambiguous_last = W.stop_ambiguous;
	parameters.missing_first = W.start_missing;
	parameters.missing_last = W.stop_missing;
	parameters.transition_first = W.start_transition;
	parameters.transition_last = W.stop_transition;
	parameters.transition_probabilities = transition_probabilities.data();
	parameters.transition_probabilities_length = transition_probabilities.size();
	parameters.missing_probabilities = missing_probabilities.data();
	parameters.missing_probabilities_length = missing_probabilities.size();
	parameters.scratch = scratch.data();
	parameters.scratch_length = scratch.size();
	parameters.alpha_locus_scratch = alpha_locus.data();
	parameters.alpha_locus_scratch_length = alpha_locus.size();

	int32_t outcome = 0;
	status = shapeit_hmm_run_segment_double_v1(&parameters, &outcome);
	if (status != SHAPEIT_HMM_STATUS_OK) {
		throw runtime_error("Rust HMM rejected the segment layout (status " +
			to_string(status) + ")");
	}
	return outcome;
}

int run_haplotype_segment_single_rust(
	genotype * G, bitmatrix & H, vector < unsigned int > & conditioning_haplotypes,
	window & W, hmm_parameters & M, vector < double > & transition_probabilities,
	vector < float > & missing_probabilities) {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));

	bitmatrix Hvar;
	const int locus_offset = Hvar.subsetTranspose(
		H, conditioning_haplotypes, W.start_locus, W.stop_locus);
	shapeit_hmm_segment_single_v1 parameters = {};
	parameters.abi_version = SHAPEIT_HMM_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);
	parameters.variants = G->Variants.data();
	parameters.variants_length = G->Variants.size();
	parameters.ambiguous = G->Ambiguous.data();
	parameters.ambiguous_length = G->Ambiguous.size();
	parameters.segment_lengths = G->Lengths.data();
	parameters.segment_lengths_length = G->Lengths.size();
	parameters.diplotypes = reinterpret_cast<const uint64_t *>(G->Diplotypes.data());
	parameters.diplotypes_length = G->Diplotypes.size();
	parameters.haplotypes = Hvar.bytes;
	parameters.haplotypes_length = Hvar.n_bytes;
	parameters.haplotype_stride = Hvar.n_cols >> 3;
	parameters.conditioning_haplotypes = conditioning_haplotypes.size();
	parameters.locus_offset = locus_offset;
	parameters.centimorgans = M.cm.data();
	parameters.centimorgans_length = M.cm.size();
	parameters.recombination = M.t.data();
	parameters.recombination_length = M.t.size();
	parameters.rare_alleles = reinterpret_cast<const int8_t *>(M.rare_allele.data());
	parameters.rare_alleles_length = M.rare_allele.size();
	parameters.effective_population_size = M.Neff;
	parameters.total_haplotypes = M.Nhap;
	parameters.emission_match = M.ee;
	parameters.emission_mismatch = M.ed;
	parameters.segment_first = W.start_segment;
	parameters.segment_last = W.stop_segment;
	parameters.locus_first = W.start_locus;
	parameters.locus_last = W.stop_locus;
	parameters.ambiguous_first = W.start_ambiguous;
	parameters.ambiguous_last = W.stop_ambiguous;
	parameters.missing_first = W.start_missing;
	parameters.missing_last = W.stop_missing;
	parameters.transition_first = W.start_transition;
	parameters.transition_last = W.stop_transition;
	parameters.transition_probabilities = transition_probabilities.data();
	parameters.transition_probabilities_length = transition_probabilities.size();
	parameters.missing_probabilities = missing_probabilities.data();
	parameters.missing_probabilities_length = missing_probabilities.size();

	const size_t segment_count = W.stop_segment - W.start_segment + 1;
	const size_t missing_count = std::max(0, W.stop_missing - W.start_missing + 1);
	size_t float_scratch_length = 0;
	uint32_t status = shapeit_hmm_double_scratch_len_v1(
		conditioning_haplotypes.size(), segment_count, missing_count,
		&float_scratch_length);
	if (status != SHAPEIT_HMM_STATUS_OK) {
		throw runtime_error("Rust single HMM rejected scratch dimensions (status " +
			to_string(status) + ")");
	}
	if (segment_count > (SIZE_MAX - 1) / 4) {
		throw runtime_error("Rust single HMM index workspace size overflow");
	}
	const size_t index_scratch_length = 4 * segment_count + 1;
	static thread_local vector < float,
		boost::alignment::aligned_allocator < float, 32 > > scratch;
	static thread_local vector < int32_t > alpha_locus;
	static thread_local vector < size_t > index_scratch;
	scratch.resize(float_scratch_length);
	alpha_locus.resize(segment_count);
	index_scratch.resize(index_scratch_length);
	parameters.scratch = scratch.data();
	parameters.scratch_length = scratch.size();
	parameters.alpha_locus_scratch = alpha_locus.data();
	parameters.alpha_locus_scratch_length = alpha_locus.size();
	parameters.index_scratch = index_scratch.data();
	parameters.index_scratch_length = index_scratch.size();

	int32_t outcome = 0;
	status = shapeit_hmm_run_segment_single_prevalidated_v1(&parameters, &outcome);
	if (status != SHAPEIT_HMM_STATUS_OK) {
		throw runtime_error("Rust single HMM rejected the segment layout (status " +
			to_string(status) + ")");
	}
	return outcome;
}
