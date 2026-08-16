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
	unsigned int curr_dipcount = 0, prev_dipcount = 1;
	vector < vector < double > > maxProbs = vector < vector < double > > (n_segments, vector < double > ());
	vector < vector < int > > maxIndexes = vector < vector < int > > (n_segments, vector < int > ());

	for (int s = 0, toffset = 0, trel = 0 ; s < n_segments ; s ++) {
		curr_dipcount = countDiplotypes(Diplotypes[s]);
		maxProbs[s] = vector < double > (curr_dipcount, 0.0);
		maxIndexes[s] = vector < int > (curr_dipcount, 0);
		for (int t = 0 ; t < prev_dipcount * curr_dipcount ; t++) {
			int prev_dip = t/curr_dipcount;
			int next_dip = t%curr_dipcount;
			//double currProb = (s?maxProbs[s-1][prev_dip]:1.0) * StoredProbs[t+toffset];
			double currProb = (s?maxProbs[s-1][prev_dip]:1.0) * (ProbMask[t+toffset]?ProbStored[trel++]:1e-6);
			if (currProb > maxProbs[s][next_dip]) {
				maxProbs[s][next_dip] = currProb;
				maxIndexes[s][next_dip] = prev_dip;
			}
		}
		double sumProb = 0.0;
		for (int d = 0 ; d < curr_dipcount ; d ++) sumProb += maxProbs[s][d];
		for (int d = 0 ; d < curr_dipcount ; d ++) maxProbs[s][d] /= sumProb;
		toffset += prev_dipcount * curr_dipcount;
		prev_dipcount = curr_dipcount;
	}

	vector < unsigned char > DipSampled = vector < unsigned char >(n_segments, 0);
	unsigned int bestDip = alg.imax(maxProbs.back());
	makeDiplotypes(Diplotypes.back());
	DipSampled.back() = curr_dipcodes[bestDip];
	for (int s = DipSampled.size() - 2 ; s >= 0 ; s --) {
		bestDip = maxIndexes[s+1][bestDip];
		makeDiplotypes(Diplotypes[s]);
		DipSampled[s] = curr_dipcodes[bestDip];
	}
	make(DipSampled);
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
