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

#include <objects/compute_job.h>
#include <shapeit_common.h>
#include <shapeit_conditioning.h>

#include <objects/hmm_parameters.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

using namespace std;

compute_job::compute_job(genotype_set & G, conditioning_set & _H) : H(_H) {
	Conditioning = nullptr;
	Haploid = vector < uint8_t > (G.n_ind, 0);
	for (int ind = 0 ; ind < G.n_ind ; ind ++) Haploid[ind] = G.vecG[ind]->isHaploid();
}

compute_job::compute_job(const compute_job & other) : H(other.H) {
	Conditioning = nullptr;
	Haploid = other.Haploid;
}

compute_job::~compute_job() {
	free();
}

void compute_job::free () {
	if (Conditioning != nullptr) {
		shapeit_conditioning_job_free_v1(Conditioning);
		Conditioning = nullptr;
	}
	vector < uint8_t > ().swap(Haploid);
}

int compute_job::run(unsigned int ind, genotype * genotype_graph,
	hmm_parameters & model, double min_window_size, unsigned int stage,
	double prune_threshold, random_number_generator & window_rng,
	random_number_generator & fallback_rng, random_number_generator & sample_rng,
	int & underflow_recovered_summing, int & underflow_recovered_precision) {
	static_assert(sizeof(unsigned int) == sizeof(uint32_t));
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	static_assert(sizeof(int) == sizeof(int32_t));
	assert(window_rng.isFresh());
	assert(fallback_rng.isFresh());
	assert(sample_rng.isFresh());

	shapeit_common_phase_job_v1 parameters = {};
	parameters.abi_version = SHAPEIT_COMMON_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);

	shapeit_conditioning_graph_build_v1 & conditioning = parameters.conditioning;
	conditioning.abi_version = SHAPEIT_CONDITIONING_ABI_VERSION;
	conditioning.struct_size = sizeof(conditioning);
	conditioning.graph = genotype_graph->Graph;
	conditioning.centimorgans = model.cm_double.data();
	conditioning.centimorgans_length = model.cm_double.size();
	conditioning.minimum_window_centimorgans = min_window_size;
	conditioning.selected_sites = H.sites_pbwt_selection.data();
	conditioning.selected_sites_length = H.sites_pbwt_selection.size();
	conditioning.site_grouping = reinterpret_cast<const int32_t *>(H.sites_pbwt_grouping.data());
	conditioning.site_grouping_length = H.sites_pbwt_grouping.size();
	conditioning.pbwt_neighbors = reinterpret_cast<const int32_t *>(H.indexes_pbwt_neighbour.data());
	conditioning.pbwt_neighbors_length = H.indexes_pbwt_neighbour.size();
	conditioning.pbwt_depth = H.depth;
	conditioning.pbwt_group_count = H.sites_pbwt_ngroups;
	conditioning.target_individual = ind;
	conditioning.target_individual_count = H.n_ind;
	conditioning.haplotype_count = H.n_hap;
	conditioning.haploid_individuals = Haploid.data();
	conditioning.haploid_individuals_length = Haploid.size();
	conditioning.haplotypes = H.H_opt_hap.bytes;
	conditioning.haplotypes_length = H.H_opt_hap.n_bytes;
	conditioning.haplotype_stride = H.H_opt_hap.n_cols >> 3;
	conditioning.maximum_heterozygote_mismatch = 0.75f;
	conditioning.window_seed = window_rng.getSeed();
	conditioning.window_domain = window_rng.getDomain();
	conditioning.window_iteration = window_rng.getIteration();
	conditioning.window_item = window_rng.getItem();
	conditioning.fallback_seed = fallback_rng.getSeed();
	conditioning.fallback_domain = fallback_rng.getDomain();
	conditioning.fallback_iteration = fallback_rng.getIteration();
	conditioning.fallback_item = fallback_rng.getItem();

	shapeit_hmm_phase_job_v1 & phase = parameters.phase;
	phase.abi_version = SHAPEIT_HMM_ABI_VERSION;
	phase.struct_size = sizeof(phase);
	phase.graph = genotype_graph->Graph;
	phase.haplotypes = H.H_opt_hap.bytes;
	phase.haplotypes_length = H.H_opt_hap.n_bytes;
	phase.haplotype_stride = H.H_opt_hap.n_cols >> 3;
	phase.centimorgans = model.cm.data();
	phase.centimorgans_length = model.cm.size();
	phase.recombination = model.t.data();
	phase.recombination_length = model.t.size();
	phase.rare_alleles = reinterpret_cast<const int8_t *>(model.rare_allele.data());
	phase.rare_alleles_length = model.rare_allele.size();
	phase.effective_population_size = model.Neff;
	phase.total_haplotypes = model.Nhap;
	phase.emission_match = model.ee;
	phase.emission_mismatch = model.ed;
	phase.stage = stage;
	phase.prune_threshold = prune_threshold;
	phase.sample_seed = sample_rng.getSeed();
	phase.sample_domain = sample_rng.getDomain();
	phase.sample_iteration = sample_rng.getIteration();
	phase.sample_item = sample_rng.getItem();

	shapeit_hmm_job_result_v1 result = {};
	const uint32_t status = shapeit_common_phase_job_run_v1(
		&parameters, &Conditioning, &result);
	if (status == SHAPEIT_COMMON_STATUS_INSUFFICIENT_STATES) {
		vrb.error("Fewer than two conditioning haplotypes are available for [" +
			genotype_graph->name + "]");
	}
	if (status != SHAPEIT_COMMON_STATUS_OK) {
		throw runtime_error("Rust common phase job rejected its input layout (status " +
			to_string(status) + ")");
	}
	underflow_recovered_summing = result.underflow_recovered_summing;
	underflow_recovered_precision = result.underflow_recovered_precision;
	return result.fatal_outcome;
}

size_t compute_job::size() {
	return shapeit_conditioning_job_window_count_v1(Conditioning);
}

void compute_job::windowStats(size_t index, int & start_locus, int & stop_locus,
	size_t & states_length, bool & used_fallback) {
	shapeit_genotype_window_v1 window = {};
	const uint32_t * states = nullptr;
	uint8_t fallback = 0;
	const uint32_t status = shapeit_conditioning_job_window_v1(
		Conditioning, index, &window, &states, &states_length, &fallback);
	if (status != SHAPEIT_CONDITIONING_STATUS_OK) {
		throw runtime_error("Rust conditioning window accessor failed (status " +
			to_string(status) + ")");
	}
	start_locus = window.start_locus;
	stop_locus = window.stop_locus;
	used_fallback = fallback != 0;
}
