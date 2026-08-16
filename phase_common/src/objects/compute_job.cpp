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
#include <shapeit_conditioning.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

using namespace std;

compute_job::compute_job(variant_map & _V, genotype_set & _G, conditioning_set & _H, unsigned int n_max_transitions, unsigned int n_max_missing) : V(_V), G(_G), H(_H) {
	T = vector < double > (n_max_transitions, 0.0);
	M = vector < float > (n_max_missing , 0.0);
	Conditioning = nullptr;
	Haploid = vector < uint8_t > (G.n_ind, 0);
	for (int ind = 0 ; ind < G.n_ind ; ind ++) Haploid[ind] = G.vecG[ind]->isHaploid();
}

compute_job::compute_job(const compute_job & other) : V(other.V), G(other.G), H(other.H) {
	T = other.T;
	M = other.M;
	Conditioning = nullptr;
	Haploid = other.Haploid;
}

compute_job::~compute_job() {
	free();
}

void compute_job::free () {
	Kstates.clear();
	if (Conditioning != nullptr) {
		shapeit_conditioning_job_free_v1(Conditioning);
		Conditioning = nullptr;
	}
	vector < double > ().swap(T);
	vector < float > ().swap(M);
	vector < uint8_t > ().swap(Haploid);
	Kbanned.clear();
	Windows.clear();
}

void compute_job::make(unsigned int ind, double min_window_size, random_number_generator & job_rng, random_number_generator & fallback_rng) {
	static_assert(sizeof(unsigned int) == sizeof(uint32_t));
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	static_assert(sizeof(int) == sizeof(int32_t));
	assert(job_rng.isFresh());
	assert(fallback_rng.isFresh());
	Kstates.clear();
	Kbanned.clear();
	Windows.clear();

	genotype * genotype_graph = G.vecG[ind];
	const shapeit_genotype_graph_view_v1 graph = genotype_graph->graphView();
	vector < double > start_centimorgans(graph.segment_lengths_length);
	vector < double > stop_centimorgans(graph.segment_lengths_length);
	for (unsigned int segment = 0, locus = 0 ; segment < graph.segment_lengths_length ; segment ++) {
		start_centimorgans[segment] = V.vec_pos[locus]->cm;
		locus += graph.segment_lengths[segment];
		stop_centimorgans[segment] = V.vec_pos[locus - 1]->cm;
	}

	shapeit_conditioning_build_v1 parameters = {};
	parameters.abi_version = SHAPEIT_CONDITIONING_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);
	parameters.variants = graph.variants;
	parameters.variants_length = graph.variants_length;
	parameters.variant_count = graph.variant_count;
	parameters.diplotypes = graph.diplotypes;
	parameters.diplotypes_length = graph.diplotypes_length;
	parameters.segment_lengths = graph.segment_lengths;
	parameters.segment_lengths_length = graph.segment_lengths_length;
	parameters.segment_start_centimorgans = start_centimorgans.data();
	parameters.segment_start_centimorgans_length = start_centimorgans.size();
	parameters.segment_stop_centimorgans = stop_centimorgans.data();
	parameters.segment_stop_centimorgans_length = stop_centimorgans.size();
	parameters.minimum_window_centimorgans = min_window_size;
	parameters.selected_sites = H.sites_pbwt_selection.data();
	parameters.selected_sites_length = H.sites_pbwt_selection.size();
	parameters.site_grouping = reinterpret_cast<const int32_t *>(H.sites_pbwt_grouping.data());
	parameters.site_grouping_length = H.sites_pbwt_grouping.size();
	parameters.pbwt_neighbors = reinterpret_cast<const int32_t *>(H.indexes_pbwt_neighbour.data());
	parameters.pbwt_neighbors_length = H.indexes_pbwt_neighbour.size();
	parameters.pbwt_depth = H.depth;
	parameters.pbwt_group_count = H.sites_pbwt_ngroups;
	parameters.target_individual = ind;
	parameters.target_individual_count = H.n_ind;
	parameters.haplotype_count = H.n_hap;
	parameters.haploid_individuals = Haploid.data();
	parameters.haploid_individuals_length = Haploid.size();
	parameters.haplotypes = H.H_opt_hap.bytes;
	parameters.haplotypes_length = H.H_opt_hap.n_bytes;
	parameters.haplotype_stride = H.H_opt_hap.n_cols >> 3;
	parameters.maximum_heterozygote_mismatch = 0.75f;
	parameters.window_seed = job_rng.getSeed();
	parameters.window_domain = job_rng.getDomain();
	parameters.window_iteration = job_rng.getIteration();
	parameters.window_item = job_rng.getItem();
	parameters.fallback_seed = fallback_rng.getSeed();
	parameters.fallback_domain = fallback_rng.getDomain();
	parameters.fallback_iteration = fallback_rng.getIteration();
	parameters.fallback_item = fallback_rng.getItem();

	uint32_t status = shapeit_conditioning_job_build_v1(&parameters, &Conditioning);
	if (status == SHAPEIT_CONDITIONING_STATUS_INSUFFICIENT_STATES) {
		vrb.error("Fewer than two conditioning haplotypes are available for [" +
			genotype_graph->name + "]");
	}
	if (status != SHAPEIT_CONDITIONING_STATUS_OK) {
		throw runtime_error("Rust conditioning job rejected its input layout (status " +
			to_string(status) + ")");
	}

	const size_t n_windows = shapeit_conditioning_job_window_count_v1(Conditioning);
	Windows.W = vector < window > (n_windows);
	Kstates.resize(n_windows);
	for (size_t w = 0 ; w < n_windows ; w ++) {
		shapeit_genotype_window_v1 source_window = {};
		const uint32_t * states = nullptr;
		size_t states_length = 0;
		uint8_t used_fallback = 0;
		status = shapeit_conditioning_job_window_v1(
			Conditioning, w, &source_window, &states, &states_length, &used_fallback);
		if (status != SHAPEIT_CONDITIONING_STATUS_OK) {
			throw runtime_error("Rust conditioning window accessor failed (status " +
				to_string(status) + ")");
		}
		window & target = Windows.W[w];
		target.start_locus = source_window.start_locus;
		target.start_segment = source_window.start_segment;
		target.start_ambiguous = source_window.start_ambiguous;
		target.start_missing = source_window.start_missing;
		target.start_transition = source_window.start_transition;
		target.stop_locus = source_window.stop_locus;
		target.stop_segment = source_window.stop_segment;
		target.stop_ambiguous = source_window.stop_ambiguous;
		target.stop_missing = source_window.stop_missing;
		target.stop_transition = source_window.stop_transition;
		Kstates[w] = span < const uint32_t > (states, states_length);
		if (used_fallback) {
			vrb.warning("No PBWT states found [" + genotype_graph->name + " / w=" +
				stb.str(w) + "] / Using " + stb.str(states_length) + " random states");
		}
	}

	const shapeit_conditioning_track_v1 * tracks = nullptr;
	size_t tracks_length = 0;
	status = shapeit_conditioning_job_tracks_v1(Conditioning, &tracks, &tracks_length);
	if (status != SHAPEIT_CONDITIONING_STATUS_OK) {
		throw runtime_error("Rust conditioning track accessor failed (status " +
			to_string(status) + ")");
	}
	Kbanned.reserve(tracks_length);
	for (size_t t = 0 ; t < tracks_length ; t ++) {
		Kbanned.push_back({tracks[t].individual, tracks[t].from, tracks[t].to});
	}
}
