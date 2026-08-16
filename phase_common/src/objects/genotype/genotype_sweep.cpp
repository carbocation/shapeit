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

#include <cassert>
#include <cstdint>
#include <stdexcept>

using namespace std;

void genotype::sample(vector < double > & CurrentTransProbabilities, vector < float > & CurrentMissingProbabilities, random_number_generator & job_rng) {
	assert(job_rng.isFresh());
	const uint32_t status = shapeit_genotype_graph_sample_current_v1(
		Graph, CurrentTransProbabilities.data(),
		CurrentTransProbabilities.size(), CurrentMissingProbabilities.data(),
		CurrentMissingProbabilities.size(), job_rng.getSeed(),
		job_rng.getDomain(), job_rng.getIteration(), job_rng.getItem());
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype sampler rejected graph layout (status " +
			to_string(status) + ")");
	}
}

void genotype::solve() {
	const uint32_t status = shapeit_genotype_graph_solve_current_v1(Graph);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype solver rejected stored graph state (status " +
			to_string(status) + ")");
	}
}

void genotype::store(vector < double > & CurrentTransProbabilities, vector < float > & CurrentMissingProbabilities) {
	const shapeit_genotype_graph_view_v1 graph = graphView();
	const size_t missing_probabilities = graph.missing_count * HAP_NUMBER;
	if (CurrentTransProbabilities.size() < graph.transition_count ||
		CurrentMissingProbabilities.size() < missing_probabilities) {
		throw runtime_error("Current genotype probabilities are shorter than the graph layout");
	}
	uint32_t status = shapeit_genotype_graph_store_v1(
		Graph, CurrentTransProbabilities.data(), graph.transition_count,
		CurrentMissingProbabilities.data(), missing_probabilities);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype storage rejected current probabilities (status " +
			to_string(status) + ")");
	}
	shapeit_genotype_storage_view_v1 view = {};
	status = shapeit_genotype_graph_storage_borrow_v1(Graph, &view);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK || view.transition_count != graph.transition_count) {
		throw runtime_error("Rust genotype storage returned an invalid view");
	}
}
