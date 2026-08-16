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

#include <objects/genotype/genotype_header.h>
#include <shapeit_genotype.h>

#include <cstdint>
#include <stdexcept>

using namespace std;

void genotype::prune(vector < double > & probabilities, double threshold_probability_mass) {
	uint32_t status = shapeit_genotype_graph_prune_v1(
		Graph, probabilities.data(), n_transitions, threshold_probability_mass);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype pruning rejected its graph (status " +
			to_string(status) + ")");
	}
	const shapeit_genotype_graph_view_v1 view = graphView();
	n_segments = view.segment_lengths_length;
	n_transitions = view.transition_count;
}
