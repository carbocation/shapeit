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

#include <containers/window_set.h>
#include <shapeit_genotype.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

using namespace std;

window_set::window_set() {
	W.clear();
}

window_set::~window_set() {
	W.clear();
}

void window_set::clear() {
	W.clear();
}

int window_set::size() {
	return W.size();
}

int window_set::build (variant_map & V, genotype * g, float min_window_size, random_number_generator & job_rng) {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	assert(job_rng.isFresh());
	vector < double > start_centimorgans(g->n_segments);
	vector < double > stop_centimorgans(g->n_segments);
	for (unsigned int segment = 0, locus = 0 ; segment < g->n_segments ; segment ++) {
		start_centimorgans[segment] = V.vec_pos[locus]->cm;
		locus += g->Lengths[segment];
		stop_centimorgans[segment] = V.vec_pos[locus - 1]->cm;
	}

	vector < shapeit_genotype_window_v1 > output(g->n_segments);
	size_t n_windows = 0;
	const uint32_t status = shapeit_genotype_windows_v1(
		g->Variants.data(), g->Variants.size(), g->n_variants,
		reinterpret_cast<const uint64_t *>(g->Diplotypes.data()), g->Diplotypes.size(),
		g->Lengths.data(), g->Lengths.size(), start_centimorgans.data(),
		start_centimorgans.size(), stop_centimorgans.data(), stop_centimorgans.size(),
		min_window_size, job_rng.getSeed(), job_rng.getDomain(), job_rng.getIteration(),
		job_rng.getItem(), output.data(), output.size(), &n_windows);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype window builder rejected graph layout (status " +
			to_string(status) + ")");
	}

	W = vector < window > (n_windows);
	for (size_t w = 0 ; w < n_windows ; w ++) {
		W[w].start_locus = output[w].start_locus;
		W[w].start_segment = output[w].start_segment;
		W[w].start_ambiguous = output[w].start_ambiguous;
		W[w].start_missing = output[w].start_missing;
		W[w].start_transition = output[w].start_transition;
		W[w].stop_locus = output[w].stop_locus;
		W[w].stop_segment = output[w].stop_segment;
		W[w].stop_ambiguous = output[w].stop_ambiguous;
		W[w].stop_missing = output[w].stop_missing;
		W[w].stop_transition = output[w].stop_transition;
	}
	return static_cast<int>(n_windows);
}
