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

extern "C" void shapeit_pbwt_solve_progress_callback(
	size_t completed, size_t total, void *) {
	if (total > 0) vrb.progress("  * PBWT phasing sweep", completed * 1.0 / total);
}

void conditioning_set::solve(genotype_set * GS) {
	tac.clock();
	static_assert(sizeof(int) == sizeof(int32_t));

	scoreBit = vector < float > (n_site + 1, 0.0);
	for (int l = 0 ; l < n_site+1 ; ++l) scoreBit[l] = log (l + 1.0);
	vector < shapeit_genotype_graph_v1 * > graphs(GS->n_ind, nullptr);
	for (int individual = 0 ; individual < GS->n_ind ; individual ++) {
		graphs[individual] = GS->vecG[individual]->Graph;
	}

	vrb.progress("  * PBWT phasing sweep", 0.0f);
	shapeit_pbwt_batch_result_v1 result = {};
	const uint32_t status = shapeit_pbwt_solve_all_v1(
		nthread, H_opt_var.bytes, H_opt_var.n_bytes, H_opt_var.n_rows,
		H_opt_var.n_cols >> 3, n_site, n_hap, graphs.data(), graphs.size(),
		reinterpret_cast<const int32_t *>(sites_pbwt_mthreading.data()),
		sites_pbwt_mthreading.size(),
		reinterpret_cast<const int32_t *>(starts_pbwt_mthreading.data()),
		starts_pbwt_mthreading.size(), scoreBit.data(), scoreBit.size(),
		H_opt_hap.bytes, H_opt_hap.n_bytes, H_opt_hap.n_rows,
		H_opt_hap.n_cols >> 3, shapeit_pbwt_solve_progress_callback, nullptr, &result);
	if (status != SHAPEIT_PBWT_STATUS_OK) {
		throw runtime_error("Rust PBWT full solver failed (status " +
			to_string(status) + ")");
	}
	vrb.bullet("PBWT phasing sweep (" + stb.str(tac.rel_time()*1.0/1000, 2) + "s)");
}
