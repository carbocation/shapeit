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

struct solver_callback_params {
	genotype_set * GS;
	conditioning_set * CS;
};

void * solver_callback(void * ptr) {
	solver_callback_params * P  = static_cast < solver_callback_params * >( ptr );

	int id_worker, id_job;
	pthread_mutex_lock(&P->CS->mutex_workers);
	id_worker = P->CS->i_worker ++;
	pthread_mutex_unlock(&P->CS->mutex_workers);

	for(;;) {
		pthread_mutex_lock(&P->CS->mutex_workers);
		id_job = P->CS->i_job ++;
		pthread_mutex_unlock(&P->CS->mutex_workers);
		if (id_job <= P->CS->sites_pbwt_mthreading.back()) {
			P->CS->solve(id_job, P->GS);
			pthread_mutex_lock(&P->CS->mutex_workers);
			vrb.progress("  * PBWT phasing sweep", (++P->CS->d_job)*1.0/(P->CS->sites_pbwt_mthreading.back()+1));
			pthread_mutex_unlock(&P->CS->mutex_workers);
		} else pthread_exit(NULL);
	}
}

void conditioning_set::solve(int chunk, genotype_set * GS) {
	static_assert(sizeof(int) == sizeof(int32_t));

	vector < const uint8_t * > genotype_variants(GS->n_ind);
	for (int individual = 0 ; individual < GS->n_ind ; individual++)
		genotype_variants[individual] = GS->vecG[individual]->Variants.data();
	const size_t genotype_variants_length = GS->vecG.empty() ?
		0 : GS->vecG.front()->Variants.size();
	const vector < unsigned char > & buffer = solve_buffers[chunk];
	const uint32_t status = shapeit_pbwt_solve_chunk_v1(
		H_opt_var.bytes, H_opt_var.n_bytes, H_opt_var.n_cols >> 3,
		n_site, n_hap, genotype_variants.data(), genotype_variants.size(),
		genotype_variants_length,
		reinterpret_cast<const int32_t *>(sites_pbwt_mthreading.data()),
		sites_pbwt_mthreading.size(), chunk, starts_pbwt_mthreading[chunk],
		buffer.data(), buffer.size(), scoreBit.data(), scoreBit.size());
	if (status != SHAPEIT_PBWT_STATUS_OK) {
		throw runtime_error("Rust PBWT solver rejected chunk layout (status " +
			to_string(status) + ")");
	}
}

void conditioning_set::solve(genotype_set * GS) {
	tac.clock();
	i_worker = 0; i_job = 0, d_job = 0;

	scoreBit = vector < float > (n_site + 1, 0.0);
	for (int l = 0 ; l < n_site+1 ; ++l) scoreBit[l] = log (l + 1.0);

	// Each PBWT chunk replays a short prefix before its writable interval. Keep
	// those prefixes immutable so a chunk never observes another chunk's
	// concurrently phased output.
	unsigned long bytes_per_locus = H_opt_var.n_cols / 8;
	int n_chunks = sites_pbwt_mthreading.back() + 1;
	solve_buffers = vector < vector < unsigned char > > (n_chunks);
	for (int chunk = 0, first_locus = 0 ; chunk < n_chunks ; ++chunk) {
		while (first_locus < n_site && sites_pbwt_mthreading[first_locus] < chunk) first_locus++;
		int buffer_start = starts_pbwt_mthreading[chunk];
		unsigned long buffer_bytes = (first_locus - buffer_start) * bytes_per_locus;
		solve_buffers[chunk].resize(buffer_bytes);
		if (buffer_bytes) memcpy(solve_buffers[chunk].data(), H_opt_var.bytes + buffer_start * bytes_per_locus, buffer_bytes);
	}

	solver_callback_params tp;
	tp.GS = GS;
	tp.CS = this;

	vrb.progress("  * PBWT phasing sweep", 0.0f);
	if (nthread > 1) {
		for (int t = 0 ; t < nthread ; t++) pthread_create( &id_workers[t] , NULL, solver_callback, static_cast < void * > (&tp));
		for (int t = 0 ; t < nthread ; t++) pthread_join( id_workers[t] , NULL);
	} else for (int c = 0 ; c  <= sites_pbwt_mthreading.back() ; c ++) {
		solve(c, GS);
		vrb.progress("  * PBWT phasing sweep", c*1.0/(sites_pbwt_mthreading.back()+1));
	}
	solve_buffers.clear();

	//Transpose to push new haps into H hap first
	transposeHaplotypes_V2H(false, false);

	vrb.bullet("PBWT phasing sweep (" + stb.str(tac.rel_time()*1.0/1000, 2) + "s)");
}
