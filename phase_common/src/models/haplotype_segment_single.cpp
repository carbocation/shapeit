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

#include <models/haplotype_segment_single.h>

using namespace std;

haplotype_segment_single::haplotype_segment_single(genotype * _G, bitmatrix & H, vector < unsigned int > & idxH, window & W, hmm_parameters & _M) : G(_G), M(_M){
	segment_first = W.start_segment;
	segment_last = W.stop_segment;
	locus_first = W.start_locus;
	locus_last = W.stop_locus;
	ambiguous_first = W.start_ambiguous;
	ambiguous_last = W.stop_ambiguous;
	missing_first = W.start_missing;
	missing_last = W.stop_missing;
	transition_first = W.start_transition;
	transition_last = W.stop_transition;
	n_cond_haps = idxH.size();
	n_missing = missing_last - missing_first + 1;
	prob_haps = HAP_NUMBER;
	use_avx512 = __builtin_cpu_supports("avx512f");

	probSumT = 0.0f;
	prob = aligned_vector32 < float > (HAP_NUMBER * n_cond_haps, 0.0f);
	probSumH = aligned_vector32 < float > (HAP_NUMBER, 0.0f);
	probSumK = aligned_vector32 < float > (n_cond_haps, 0.0f);
	AlphaLocus = vector < int > (segment_last - segment_first + 1, 0);
	AlphaSum = vector < aligned_vector32 < float > > (segment_last - segment_first + 1, aligned_vector32 < float > (HAP_NUMBER, 0.0f));
	AlphaSumSum = aligned_vector32 < float > (segment_last - segment_first + 1, 0.0);
	SegmentHapCount = vector < unsigned char > (segment_last - segment_first + 1, 1U);
	SegmentFirstAmbiguous = vector < unsigned short > (segment_last - segment_first + 1, 0U);
	SegmentLastAmbiguous = vector < unsigned short > (segment_last - segment_first + 1, 0U);
	for (int s = segment_first, v = locus_first, a = ambiguous_first ; s <= segment_last ; ++s) {
		unsigned char n_haps = 1;
		unsigned short first_ambiguous = G->Lengths[s];
		unsigned short last_ambiguous = 0;
		for (unsigned int vrel = 0 ; vrel < G->Lengths[s] ; ++vrel, ++v) {
			if (!VAR_GET_AMB(MOD2(v), G->Variants[DIV2(v)])) continue;
			first_ambiguous = std::min(first_ambiguous, static_cast<unsigned short>(vrel));
			last_ambiguous = vrel;
			const unsigned char code = G->Ambiguous[a++];
			while (n_haps < HAP_NUMBER) {
				bool periodic = true;
				for (unsigned int h = n_haps ; h < HAP_NUMBER ; ++h)
					periodic &= HAP_GET(code, h) == HAP_GET(code, h % n_haps);
				if (periodic) break;
				n_haps <<= 1;
			}
		}
		const unsigned int srel = s - segment_first;
		SegmentHapCount[srel] = n_haps;
		SegmentFirstAmbiguous[srel] = first_ambiguous;
		SegmentLastAmbiguous[srel] = last_ambiguous;
	}
	Alpha.reserve(SegmentHapCount.size());
	for (const unsigned int n_haps : SegmentHapCount)
		Alpha.emplace_back(n_haps * n_cond_haps, 0.0f);
	if (n_missing > 0) {
		AlphaMissing = vector < aligned_vector32 < float > > (n_missing, aligned_vector32 < float > (HAP_NUMBER * n_cond_haps, 0.0f));
		AlphaSumMissing = vector < aligned_vector32 < float > > (n_missing, aligned_vector32 < float > (HAP_NUMBER, 0.0f));
	}
	//Cache efficient data transfer for conditioning haplotypes
	curr_rel_locus_offset = Hvar.subsetTranspose(H, idxH, locus_first, locus_last);
}

haplotype_segment_single::~haplotype_segment_single() {
	G = NULL;
	segment_first = 0;
	segment_last = 0;
	locus_first = 0;
	locus_last = 0;
	ambiguous_first = 0;
	ambiguous_last = 0;
	transition_first = 0;
	n_cond_haps = 0;
	n_missing = 0;
	curr_segment_index = 0;
	curr_segment_locus = 0;
	curr_abs_locus = 0;
	curr_rel_locus = 0;
	curr_abs_ambiguous = 0;
	curr_abs_transition = 0;
	probSumT = 0.0;
	prob_haps = HAP_NUMBER;
	use_avx512 = false;
	prob.clear();
	probSumK.clear();
	probSumH.clear();
	Alpha.clear();
	AlphaSum.clear();
	AlphaSumSum.clear();
	AlphaMissing.clear();
	AlphaSumMissing.clear();
	SegmentHapCount.clear();
	SegmentFirstAmbiguous.clear();
	SegmentLastAmbiguous.clear();
}

void haplotype_segment_single::forward() {
	curr_segment_index = segment_first;
	curr_segment_locus = 0;
	curr_abs_ambiguous = ambiguous_first;
	curr_abs_missing = missing_first;
	prev_abs_locus = locus_first;
	prob_haps = HAP_NUMBER;

	for (curr_abs_locus = locus_first ; curr_abs_locus <= locus_last ; curr_abs_locus++) {
		curr_rel_locus = curr_abs_locus - locus_first;
		const unsigned int curr_rel_segment = curr_segment_index - segment_first;
		const unsigned int segment_haps = SegmentHapCount[curr_rel_segment];
		const unsigned int desired_haps = curr_segment_locus < SegmentFirstAmbiguous[curr_rel_segment]
			? 1U : segment_haps;
		curr_rel_missing = curr_abs_missing - missing_first;
		bool update_prev_locus = true;
		char rare_allele = M.rare_allele[curr_abs_locus];
		bool amb = VAR_GET_AMB(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
		bool mis = VAR_GET_MIS(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
		bool hom = !(amb || mis);
		yt = (curr_abs_locus == locus_first)?0.0:M.getForwardTransProb(prev_abs_locus, curr_abs_locus);
		nt = 1.0f - yt;
		if (curr_rel_locus != 0 && curr_segment_locus != 0) RESHAPE_HAPS(desired_haps);

		if (curr_rel_locus == 0) {
			prob_haps = HAP_NUMBER;
			if (hom) INIT_HOM();
			else if (amb) INIT_AMB();
			else INIT_MIS();
		} else if (curr_segment_locus != 0) {
			if (hom) update_prev_locus = RUN_HOM(rare_allele);
			else if (amb) RUN_AMB();
			else RUN_MIS();
		} else {
			prob_haps = HAP_NUMBER;
			if (hom) COLLAPSE_HOM();
			else if (amb) COLLAPSE_AMB();
			else  COLLAPSE_MIS();
		}
		prev_abs_locus=update_prev_locus?curr_abs_locus:prev_abs_locus;

		if (mis) {
			AlphaMissing[curr_rel_missing] = prob;
			AlphaSumMissing[curr_rel_missing] = probSumH;
			curr_abs_missing ++;
		}
		if (curr_segment_locus == G->Lengths[curr_segment_index] - 1) {
			RESHAPE_HAPS(segment_haps);
			SUMK();
			std::copy_n(prob.begin(), SegmentHapCount[curr_rel_segment] * n_cond_haps,
				Alpha[curr_rel_segment].begin());
			AlphaSum[curr_rel_segment] = probSumH;
			AlphaSumSum[curr_rel_segment] = probSumT;
			AlphaLocus[curr_rel_segment] = prev_abs_locus;
		} else RESHAPE_HAPS(desired_haps);

		curr_segment_locus ++;
		curr_abs_ambiguous += amb;
		if (curr_segment_locus >= G->Lengths[curr_segment_index]) {
			curr_segment_index++;
			curr_segment_locus = 0;
		}
	}
}

int haplotype_segment_single::backward(vector < double > & transition_probabilities, vector < float > & missing_probabilities) {
	int n_underflow_recovered = 0;
	curr_segment_index = segment_last;
	curr_segment_locus = G->Lengths[segment_last] - 1;
	curr_abs_ambiguous = ambiguous_last;
	curr_abs_missing = missing_last;
	curr_abs_transition = transition_last;
	prev_abs_locus = locus_last;
	prob_haps = HAP_NUMBER;

	for (curr_abs_locus = locus_last ; curr_abs_locus >= locus_first ; curr_abs_locus--) {
		curr_rel_locus = curr_abs_locus - locus_first;
		const unsigned int curr_rel_segment = curr_segment_index - segment_first;
		const unsigned int segment_haps = SegmentHapCount[curr_rel_segment];
		const unsigned int desired_haps = curr_segment_locus > SegmentLastAmbiguous[curr_rel_segment]
			? 1U : segment_haps;
		curr_rel_missing = curr_abs_missing - missing_first;
		char rare_allele = M.rare_allele[curr_abs_locus];
		bool update_prev_locus = true;
		bool amb = VAR_GET_AMB(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
		bool mis = VAR_GET_MIS(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
		bool hom = !(amb || mis);
		yt = (curr_abs_locus == locus_last)?0.0:M.getBackwardTransProb(prev_abs_locus, curr_abs_locus);
		nt = 1.0f - yt;
		if (curr_abs_locus != locus_last && curr_segment_locus != G->Lengths[curr_segment_index] - 1)
			RESHAPE_HAPS(desired_haps);

		if (curr_abs_locus == locus_last) {
			prob_haps = HAP_NUMBER;
			if (hom) INIT_HOM();
			else if (amb) INIT_AMB();
			else INIT_MIS();
		} else if (curr_segment_locus != G->Lengths[curr_segment_index] - 1) {
			if (hom) update_prev_locus = RUN_HOM(rare_allele);
			else if (amb) RUN_AMB();
			else RUN_MIS();
		} else {
			prob_haps = HAP_NUMBER;
			if (hom) COLLAPSE_HOM();
			else if (amb) COLLAPSE_AMB();
			else COLLAPSE_MIS();
		}
		prev_abs_locus=update_prev_locus?curr_abs_locus:prev_abs_locus;
		if (mis) {
			IMPUTE(missing_probabilities);
			curr_abs_missing--;
		}
		if (curr_segment_locus == 0) {
			RESHAPE_HAPS(segment_haps);
			SUMK();
		}

		if (curr_abs_locus == 0) SET_FIRST_TRANS(transition_probabilities);
		if (curr_segment_locus == 0 && curr_abs_locus != locus_first) {
			int ret = SET_OTHER_TRANS(transition_probabilities);
			if (ret < 0) return ret;
			else n_underflow_recovered += ret;
		}

		if (curr_segment_locus != 0) RESHAPE_HAPS(desired_haps);


		curr_segment_locus--;
		curr_abs_ambiguous -= amb;
		if (curr_segment_locus < 0 && curr_segment_index > 0) {
			curr_segment_index--;
			curr_segment_locus = G->Lengths[curr_segment_index] - 1;
		}
	}
	return n_underflow_recovered;
}

void haplotype_segment_single::SET_FIRST_TRANS(vector < double > & transition_probabilities) {
	double scale = 1.0f / probSumT, scaleDip = 0.0f;
	unsigned int n_transitions = G->countDiplotypes(G->Diplotypes[0]);
	vector < double > cprobs = vector < double > (n_transitions, 0.0);
	unsigned int t = 0;
	for (unsigned long active = G->Diplotypes[0]; active; active &= active - 1) {
		const unsigned int d = std::countr_zero(active);
		cprobs[t] = (double)(probSumH[DIP_HAP0(d)]*scale) * (double)(probSumH[DIP_HAP1(d)]*scale);
		scaleDip += cprobs[t++];
	}
	scaleDip = 1.0f / scaleDip;
	for (unsigned int t = 0 ; t < n_transitions ; t ++) transition_probabilities[t] = cprobs[t] * scaleDip;
}

int haplotype_segment_single::SET_OTHER_TRANS(vector < double > & transition_probabilities) {
	int underflow_recovered = 0;
	if (TRANS_HAP()) {
		if (TRANS_HAP_DOUBLE() || TRANS_DIP_MULT_DOUBLE()) return -1;
	} else if (TRANS_DIP_MULT()) {
		//The dense HMM state is still available in float. Recompute only this
		//segment-boundary contraction in double before falling back to a full
		//double-precision forward/backward pass.
		if (TRANS_HAP_DOUBLE() || TRANS_DIP_MULT_DOUBLE()) {
			if (TRANS_DIP_ADD()) return -2;
			underflow_recovered = 1;
		}
	}
	unsigned int curr_dipcount = G->countDiplotypes(G->Diplotypes[curr_segment_index]);
	unsigned int prev_dipcount = G->countDiplotypes(G->Diplotypes[curr_segment_index-1]);
	unsigned int n_transitions = curr_dipcount * prev_dipcount;
	double scaleDip = 1.0 / sumDProbs;
	curr_abs_transition -= (n_transitions - 1);
	for (int t = 0 ; t < n_transitions ; t ++) transition_probabilities[curr_abs_transition + t] = DProbs[t] * scaleDip;
	curr_abs_transition --;
	return underflow_recovered;
}
