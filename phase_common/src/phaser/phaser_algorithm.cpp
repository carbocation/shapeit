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

#include <phaser/phaser_header.h>
#include <shapeit_common.h>

using namespace std;

extern "C" void shapeit_common_hmm_progress_callback(
	size_t completed, size_t total, void *) {
	if (total > 0) vrb.progress("  * HMM computations", completed * 1.0 / total);
}

extern "C" void shapeit_common_pbwt_progress_callback(
	size_t completed, size_t total, void *) {
	if (total > 0) vrb.progress("  * PBWT selection", completed * 1.0 / total);
}

void phaser::phaseWindow() {
	vrb.progress("  * PBWT selection", 0.0);

	shapeit_common_full_iteration_v1 full = {};
	full.abi_version = SHAPEIT_COMMON_ABI_VERSION;
	full.struct_size = sizeof(full);
	full.haplotype_major = H.H_opt_hap.bytes;
	full.haplotype_major_length = H.H_opt_hap.n_bytes;
	full.haplotype_major_rows = H.H_opt_hap.n_rows;
	full.haplotype_major_stride = H.H_opt_hap.n_cols >> 3;
	full.variant_major = H.H_opt_var.bytes;
	full.variant_major_length = H.H_opt_var.n_bytes;
	full.variant_major_rows = H.H_opt_var.n_rows;
	full.variant_major_stride = H.H_opt_var.n_cols >> 3;

	shapeit_common_pbwt_selection_v1 & pbwt = full.pbwt;
	pbwt.abi_version = SHAPEIT_COMMON_ABI_VERSION;
	pbwt.struct_size = sizeof(pbwt);
	pbwt.haplotypes = H.H_opt_var.bytes;
	pbwt.haplotypes_length = H.H_opt_var.n_bytes;
	pbwt.haplotype_stride = H.H_opt_var.n_cols >> 3;
	pbwt.site_count = H.n_site;
	pbwt.haplotype_count = H.n_hap;
	pbwt.target_individual_count = H.n_ind;
	pbwt.evaluated_sites = H.sites_pbwt_evaluation.data();
	pbwt.evaluated_sites_length = H.sites_pbwt_evaluation.size();
	pbwt.selected_sites = H.sites_pbwt_selection.data();
	pbwt.selected_sites_length = H.sites_pbwt_selection.size();
	pbwt.site_groups = reinterpret_cast<const int32_t *>(H.sites_pbwt_grouping.data());
	pbwt.site_groups_length = H.sites_pbwt_grouping.size();
	pbwt.group_count = H.sites_pbwt_ngroups;
	pbwt.site_chunks = reinterpret_cast<const int32_t *>(H.sites_pbwt_mthreading.data());
	pbwt.site_chunks_length = H.sites_pbwt_mthreading.size();
	pbwt.chunk_starts = reinterpret_cast<const int32_t *>(H.starts_pbwt_mthreading.data());
	pbwt.chunk_count = H.starts_pbwt_mthreading.size();
	pbwt.depth = H.depth;
	pbwt.ibd2_registry = H.Kbanned.Handle;
	pbwt.neighbors = reinterpret_cast<int32_t *>(H.indexes_pbwt_neighbour.data());
	pbwt.neighbors_length = H.indexes_pbwt_neighbour.size();
	pbwt.seed = rng.getSeed();
	pbwt.domain = RNG_DOMAIN_PHASE_COMMON_PBWT_SITE;
	pbwt.iteration = iteration_index;
	pbwt.progress = shapeit_common_pbwt_progress_callback;

	shapeit_common_iteration_v1 & parameters = full.phase;
	parameters.abi_version = SHAPEIT_COMMON_ABI_VERSION;
	parameters.struct_size = sizeof(parameters);
	parameters.base_pair_positions = M.bp.data();
	parameters.base_pair_positions_length = M.bp.size();
	parameters.ibd2_registry = H.Kbanned.Handle;
	parameters.progress = shapeit_common_hmm_progress_callback;

	shapeit_common_phase_job_v1 & sample = parameters.sample_template;
	sample.abi_version = SHAPEIT_COMMON_ABI_VERSION;
	sample.struct_size = sizeof(sample);

	shapeit_conditioning_graph_build_v1 & conditioning = sample.conditioning;
	conditioning.abi_version = SHAPEIT_CONDITIONING_ABI_VERSION;
	conditioning.struct_size = sizeof(conditioning);
	conditioning.centimorgans = M.cm_double.data();
	conditioning.centimorgans_length = M.cm_double.size();
	conditioning.minimum_window_centimorgans = options["hmm-window"].as < double > ();
	conditioning.selected_sites = H.sites_pbwt_selection.data();
	conditioning.selected_sites_length = H.sites_pbwt_selection.size();
	conditioning.site_grouping = reinterpret_cast<const int32_t *>(H.sites_pbwt_grouping.data());
	conditioning.site_grouping_length = H.sites_pbwt_grouping.size();
	conditioning.pbwt_neighbors = reinterpret_cast<const int32_t *>(H.indexes_pbwt_neighbour.data());
	conditioning.pbwt_neighbors_length = H.indexes_pbwt_neighbour.size();
	conditioning.pbwt_depth = H.depth;
	conditioning.pbwt_group_count = H.sites_pbwt_ngroups;
	conditioning.haplotype_count = H.n_hap;
	conditioning.haplotypes = H.H_opt_hap.bytes;
	conditioning.haplotypes_length = H.H_opt_hap.n_bytes;
	conditioning.haplotype_stride = H.H_opt_hap.n_cols >> 3;
	conditioning.maximum_heterozygote_mismatch = 0.75f;
	conditioning.window_seed = rng.getSeed();
	conditioning.window_domain = RNG_DOMAIN_PHASE_COMMON_WINDOW;
	conditioning.window_iteration = iteration_index;
	conditioning.fallback_seed = rng.getSeed();
	conditioning.fallback_domain = RNG_DOMAIN_PHASE_COMMON_FALLBACK;
	conditioning.fallback_iteration = iteration_index;

	shapeit_hmm_phase_job_v1 & phase = sample.phase;
	phase.abi_version = SHAPEIT_HMM_ABI_VERSION;
	phase.struct_size = sizeof(phase);
	phase.haplotypes = H.H_opt_hap.bytes;
	phase.haplotypes_length = H.H_opt_hap.n_bytes;
	phase.haplotype_stride = H.H_opt_hap.n_cols >> 3;
	phase.centimorgans = M.cm.data();
	phase.centimorgans_length = M.cm.size();
	phase.recombination = M.t.data();
	phase.recombination_length = M.t.size();
	phase.rare_alleles = reinterpret_cast<const int8_t *>(M.rare_allele.data());
	phase.rare_alleles_length = M.rare_allele.size();
	phase.effective_population_size = M.Neff;
	phase.total_haplotypes = M.Nhap;
	phase.emission_match = M.ee;
	phase.emission_mismatch = M.ed;
	phase.stage = iteration_types[iteration_stage];
	phase.prune_threshold = options["mcmc-prune"].as < double > ();
	phase.sample_seed = rng.getSeed();
	phase.sample_domain = RNG_DOMAIN_PHASE_COMMON_MCMC;
	phase.sample_iteration = iteration_index;

	shapeit_common_full_iteration_result_v1 result = {};
	const uint32_t status = shapeit_common_workers_run_full_iteration_v1(
		phase_workers, &full, &result);
	const shapeit_common_iteration_result_v1 & phase_result = result.phase;
	const string failed_sample = phase_result.failed_sample < static_cast<size_t>(G.n_ind)
		? G.vecG[phase_result.failed_sample]->name
		: "unknown sample";
	if (status == SHAPEIT_COMMON_STATUS_INSUFFICIENT_STATES) {
		if (phase_result.failed_sample < static_cast<size_t>(G.n_ind)) {
			vrb.error("Fewer than two conditioning haplotypes are available for [" +
				failed_sample + "]");
		} else {
			vrb.error("Insufficient non-IBD2 PBWT neighbours for the requested depth");
		}
	}
	if (status != SHAPEIT_COMMON_STATUS_OK) {
		throw runtime_error("Rust full common iteration failed (status " +
			to_string(status) + ")");
	}
	if (phase_result.fatal_outcome == -2) {
		vrb.error("Diploid underflow impossible to recover for [" + failed_sample + "]");
	}
	if (phase_result.fatal_outcome == -1) {
		vrb.error("Haploid underflow impossible to recover for [" + failed_sample + "]");
	}
	vrb.bullet("PBWT selection (" + stb.str(result.pbwt_seconds, 2) + "s)");

	const size_t fallback_count = shapeit_common_workers_fallback_count_v1(phase_workers);
	for (size_t index = 0 ; index < fallback_count ; index ++) {
		shapeit_common_fallback_v1 fallback = {};
		const uint32_t fallback_status = shapeit_common_workers_fallback_v1(
			phase_workers, index, &fallback);
		if (fallback_status != SHAPEIT_COMMON_STATUS_OK ||
			fallback.sample >= static_cast<size_t>(G.n_ind)) {
			throw runtime_error("Rust common fallback reporting failed (status " +
				to_string(fallback_status) + ")");
		}
		vrb.warning("No PBWT states found [" + G.vecG[fallback.sample]->name + " / w=" +
			stb.str(fallback.window) + "] / Using " + stb.str(fallback.states) +
			" random states");
	}

	vrb.bullet("HMM computations [K=" + stb.str(phase_result.conditioning_states_mean, 1) +
		"+/-" + stb.str(phase_result.conditioning_states_sd, 1) + " / W=" +
		stb.str(phase_result.window_megabases_mean, 2) + "Mb / US=" +
		stb.str(phase_result.underflow_recovered_summing) + " / UP=" +
		stb.str(phase_result.underflow_recovered_precision) + "] (" +
		stb.str(result.hmm_seconds, 2) + "s)");
	vrb.bullet("IBD2 tracks [#inds=" + stb.str(result.ibd2.individuals) +
		" / #tracks=" + stb.str(result.ibd2.tracks) + " / #merged = " +
		stb.str(result.ibd2.merged) + "]");
	vrb.bullet("HAP update (" + stb.str(result.haplotype_refresh_seconds, 2) + "s)");
	vrb.bullet("H2V transpose (" + stb.str(result.transpose_seconds, 2) + "s)");
}

void phaser::phase() {
	unsigned long n_old_segments = G.numberOfSegments(), n_new_segments = 0;
	iteration_index = 0;
	for (iteration_stage = 0 ; iteration_stage < iteration_counts.size() ; iteration_stage ++) {
		for (int iter = 0 ; iter < iteration_counts[iteration_stage] ; iter ++) {
			//VERBOSE
			switch (iteration_types[iteration_stage]) {
			case STAGE_BURN:	vrb.title("Burn-in iteration [" + stb.str(iter+1) + "/" + stb.str(iteration_counts[iteration_stage]) + "]"); break;
			case STAGE_PRUN:	vrb.title("Pruning iteration [" + stb.str(iter+1) + "/" + stb.str(iteration_counts[iteration_stage]) + "]"); break;
			case STAGE_MAIN:	vrb.title("Main iteration [" + stb.str(iter+1) + "/" + stb.str(iteration_counts[iteration_stage]) + "]"); break;
			}
			//SELECT STATES, PHASE, COLLAPSE IBD2, AND REFRESH HAPLOTYPES IN RUST
			phaseWindow();
			//UPDATE PS after prunning
			if (iteration_types[iteration_stage] == STAGE_PRUN) {
				n_new_segments = G.numberOfSegments();
				vrb.bullet("Trimming [pc=" + stb.str((1-n_new_segments*1.0/n_old_segments)*100, 2) + "%]");
			}
			iteration_index++;
		}
	}
}
