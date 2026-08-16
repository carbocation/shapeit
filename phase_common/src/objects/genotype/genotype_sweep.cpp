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
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	assert(job_rng.isFresh());
	const uint32_t status = shapeit_genotype_sample_v1(
		Variants.data(), Variants.size(), n_variants,
		Ambiguous.data(), Ambiguous.size(),
		reinterpret_cast<const uint64_t *>(Diplotypes.data()), Diplotypes.size(),
		Lengths.data(), Lengths.size(), CurrentTransProbabilities.data(),
		CurrentTransProbabilities.size(), CurrentMissingProbabilities.data(),
		CurrentMissingProbabilities.size(), haploid, job_rng.getSeed(),
		job_rng.getDomain(), job_rng.getIteration(), job_rng.getItem());
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype sampler rejected graph layout (status " +
			to_string(status) + ")");
	}
}

void genotype::solve() {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	if (ProbMask.size() != n_transitions) {
		throw runtime_error("Stored genotype transition mask has an invalid length");
	}
	vector < uint32_t > stored_indexes;
	stored_indexes.reserve(n_stored_transitionProbs);
	for (uint32_t transition = 0 ; transition < ProbMask.size() ; transition ++) {
		if (ProbMask[transition]) stored_indexes.push_back(transition);
	}
	const uint32_t status = shapeit_genotype_solve_v1(
		Variants.data(), Variants.size(), n_variants,
		Ambiguous.data(), Ambiguous.size(),
		reinterpret_cast<const uint64_t *>(Diplotypes.data()), Diplotypes.size(),
		Lengths.data(), Lengths.size(), stored_indexes.data(), stored_indexes.size(),
		ProbStored.data(), ProbStored.size(), ProbMissing.data(), ProbMissing.size(),
		haploid, n_storage_events);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype solver rejected stored graph state (status " +
			to_string(status) + ")");
	}
}

void genotype::store(vector < double > & CurrentTransProbabilities, vector < float > & CurrentMissingProbabilities) {
	if (ProbMask.size() == 0) {
		n_stored_transitionProbs = 0;
		ProbMask = vector < bool > (n_transitions, false);
		for (unsigned int t = 0 ; t < n_transitions ; t ++) if (CurrentTransProbabilities[t] >= 1e-6) {
			n_stored_transitionProbs ++;
			ProbMask[t] = true;
		}
		ProbStored = vector  < float > (n_stored_transitionProbs, 0.0);
		ProbMissing = vector < float > (n_missing * HAP_NUMBER, 0.0);
	}
	for (unsigned int t = 0, trel = 0 ; t < n_transitions ; t ++) {
		if (ProbMask[t]) ProbStored[trel++] += CurrentTransProbabilities[t];
	}
	for (unsigned int m = 0 ; m < (n_missing * HAP_NUMBER) ; m ++) ProbMissing[m] += CurrentMissingProbabilities[m];
	n_storage_events ++;
}
