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

#include <containers/conditioning_set/conditioning_set_header.h>
#include <shapeit_pbwt.h>

#include <cstdint>
#include <stdexcept>

using namespace std;

void * selecter_callback(void * ptr) {
	conditioning_set * S = static_cast< conditioning_set * >( ptr );

	int id_worker, id_job;
	pthread_mutex_lock(&S->mutex_workers);
	id_worker = S->i_worker ++;
	pthread_mutex_unlock(&S->mutex_workers);

	for(;;) {
		pthread_mutex_lock(&S->mutex_workers);
		id_job = S->i_job ++;
		pthread_mutex_unlock(&S->mutex_workers);

		if (id_job <= S->sites_pbwt_mthreading.back()) {
			S->select(id_job);
			pthread_mutex_lock(&S->mutex_workers);
			vrb.progress("  * PBWT selection", (++S->d_job)*1.0/(S->sites_pbwt_mthreading.back()+1));
			pthread_mutex_unlock(&S->mutex_workers);
		}
		else pthread_exit(NULL);
	}
}

void conditioning_set::transposePBWTneighbours() {
	static_assert(sizeof(int) == sizeof(int32_t));
	const uint32_t status = shapeit_pbwt_transpose_neighbors_v1(
		reinterpret_cast<int32_t *>(indexes_pbwt_neighbour.data()),
		indexes_pbwt_neighbour.size(), 2UL * n_ind, sites_pbwt_ngroups, depth);
	if (status != SHAPEIT_PBWT_STATUS_OK) {
		throw runtime_error("Rust PBWT neighbour transpose rejected its layout (status " +
			to_string(status) + ")");
	}
}


void conditioning_set::select(int chunk) {
	static_assert(sizeof(int) == sizeof(int32_t));
	const uint32_t status = shapeit_pbwt_select_chunk_v1(
		H_opt_var.bytes, H_opt_var.n_bytes, H_opt_var.n_cols >> 3,
		n_site, n_hap, n_ind, sites_pbwt_evaluation.data(),
		sites_pbwt_evaluation.size(), sites_pbwt_selection.data(),
		sites_pbwt_selection.size(),
		reinterpret_cast<const int32_t *>(sites_pbwt_grouping.data()),
		sites_pbwt_grouping.size(), sites_pbwt_ngroups,
		reinterpret_cast<const int32_t *>(sites_pbwt_mthreading.data()),
		sites_pbwt_mthreading.size(), chunk, starts_pbwt_mthreading[chunk], depth,
		ibd_offsets.data(), ibd_offsets.size(), ibd_individuals.data(),
		ibd_from.data(), ibd_to.data(), ibd_individuals.size(),
		reinterpret_cast<int32_t *>(indexes_pbwt_neighbour.data()),
		indexes_pbwt_neighbour.size());
	if (status == SHAPEIT_PBWT_STATUS_INSUFFICIENT_STATES) {
		vrb.error("Insufficient non-IBD2 PBWT neighbours for the requested depth");
	}
	if (status != SHAPEIT_PBWT_STATUS_OK) {
		throw runtime_error("Rust PBWT selection rejected its layout (status " +
			to_string(status) + ")");
	}
}

void conditioning_set::select(uint32_t iteration) {
	tac.clock();
	i_worker = 0; i_job = 0, d_job = 0;

	//Select new sites at which to trigger storage
	sites_pbwt_selection = vector < uint8_t > (n_site , 0);
	uint32_t status = shapeit_pbwt_select_sites_v1(
		sites_pbwt_evaluation.data(), sites_pbwt_evaluation.size(),
		reinterpret_cast<const int32_t *>(sites_pbwt_grouping.data()),
		sites_pbwt_grouping.size(), sites_pbwt_ngroups, rng.getSeed(),
		RNG_DOMAIN_PHASE_COMMON_PBWT_SITE, iteration,
		sites_pbwt_selection.data(), sites_pbwt_selection.size());
	if (status != SHAPEIT_PBWT_STATUS_OK) {
		throw runtime_error("Rust PBWT site selection rejected its layout (status " +
			to_string(status) + ")");
	}

	//Clean up previous selected states
	fill(indexes_pbwt_neighbour.begin(), indexes_pbwt_neighbour.end() , -1);
	ibd_offsets.assign(n_ind + 1, 0);
	ibd_individuals.clear();
	ibd_from.clear();
	ibd_to.clear();
	for (int source = 0 ; source < n_ind ; source ++) {
		for (const track & value : Kbanned.IBD2[source]) {
			ibd_individuals.push_back(value.ind);
			ibd_from.push_back(value.from);
			ibd_to.push_back(value.to);
		}
		ibd_offsets[source + 1] = ibd_individuals.size();
	}

	//Perform multi-threaded selection
	vrb.progress("  * PBWT selection", 0.0f);
	if (nthread > 1) {
		for (int t = 0 ; t < nthread ; t++) pthread_create( &id_workers[t] , NULL, selecter_callback, static_cast<void *>(this));
		for (int t = 0 ; t < nthread ; t++) pthread_join( id_workers[t] , NULL);
	} else for (int c = 0 ; c  <= sites_pbwt_mthreading.back() ; c ++) {
		select(c);
		vrb.progress("  * PBWT selection", c*1.0/(sites_pbwt_mthreading.back()+1));
	}

	//Transpose matrix with selected states
	transposePBWTneighbours();
	ibd_offsets.clear();
	ibd_individuals.clear();
	ibd_from.clear();
	ibd_to.clear();

	vrb.bullet("PBWT selection (" + stb.str(tac.rel_time()*1.0/1000, 2) + "s)");
}
