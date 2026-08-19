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

#include <models/hmm_scaffold/hmm_scaffold_header.h>
#include <models/hmm_scaffold/hmm_scaffold_vector.h>

using namespace std;
namespace hsv = hmm_scaffold_vector;

hmm_scaffold::hmm_scaffold(variant_map & _V, genotype_set & _G, conditioning_set & _C, hmm_parameters & _M) : V(_V), G(_G), C(_C), M(_M){
	match_prob[0] = 1.0f; match_prob[1] = M.ed/M.ee;

	unsigned max_nstates = 0;
	for (int32_t h = 0 ; h < C.n_haplotypes ; h ++) if (C.indexes_pbwt_neighbour[h].size() > max_nstates) max_nstates = C.indexes_pbwt_neighbour[h].size();
	alpha = vector < aligned_vector32 < float > > (C.n_scaffold_variants, aligned_vector32 < float > (max_nstates, 0.0f));
	beta = aligned_vector32 < float > (max_nstates, 1.0f);
	Hvar.allocate(C.n_scaffold_variants, max_nstates);
	Hhap.allocate(max_nstates, C.n_scaffold_variants);
}

hmm_scaffold::~hmm_scaffold() {
	alpha.clear();
	beta.clear();
}

void hmm_scaffold::setup(uint32_t _hap) {
	hap = _hap;
	nstates = C.indexes_pbwt_neighbour[hap].size();

	Hvar.reallocateFast(C.n_scaffold_variants, nstates);
	Hhap.reallocateFast(nstates, C.n_scaffold_variants);

	Hhap.subset(C.Hhap, C.indexes_pbwt_neighbour[hap]);
	Hhap.transpose(Hvar);
}

double hmm_scaffold::forward() {
	float sum;
	double loglik = 0.0;
	const uint32_t nstates_vectorized = (nstates / hsv::width) * hsv::width;
	for (int32_t vs = 0 ; vs < C.n_scaffold_variants ; vs ++) {
		const std::array<float,2> emit = {match_prob[C.Hhap.get(hap, vs)], match_prob[1-C.Hhap.get(hap, vs)]};
		const hsv::float8 _emit0 = hsv::broadcast(emit[0]);
		const hsv::float8 _emit1 = hsv::broadcast(emit[1]);

		if (!vs) {
			const float f0 = 1.0f / nstates;
			const hsv::float8 _f0 = hsv::broadcast(f0);
			hsv::float8 _sum = hsv::broadcast(0.0f);
			int32_t offset = 0;
			for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
				const hsv::float8 _emiss = hsv::select_packed_byte(Hvar.getByte(vs, k), _emit0, _emit1);
				const hsv::float8 _prob_curr = hsv::multiply(_emiss, _f0);
				_sum = hsv::add(_sum, _prob_curr);
				hsv::store_aligned(&alpha[vs][k], _prob_curr);
				offset += hsv::width;
			}
			sum = (offset > 0)?hsv::horizontal_add(_sum):0.0f;
			for (; offset < nstates ; offset ++) {
				alpha[vs][offset] = f0 * emit[Hvar.get(vs, offset)];
				sum += alpha[vs][offset];
			}
			loglik += log(sum);
		} else {
			const float f0 = M.t[vs-1] / nstates;
			const float f1 = M.nt[vs-1] / sum;
			const hsv::float8 _f0 = hsv::broadcast(f0);
			const hsv::float8 _f1 = hsv::broadcast(f1);
			hsv::float8 _sum = hsv::broadcast(0.0f);
			int32_t offset = 0;
			for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
				const hsv::float8 _emiss = hsv::select_packed_byte(Hvar.getByte(vs, k), _emit0, _emit1);
				const hsv::float8 _prob_prev = hsv::load_aligned(&alpha[vs-1][k]);
				const hsv::float8 _prob_temp = hsv::multiply_add(_prob_prev, _f1, _f0);
				const hsv::float8 _prob_curr = hsv::multiply(_prob_temp, _emiss);
				_sum = hsv::add(_sum, _prob_curr);
				hsv::store_aligned(&alpha[vs][k], _prob_curr);
				offset += hsv::width;
			}
			sum = (offset > 0)?hsv::horizontal_add(_sum):0.0f;
			for (; offset < nstates ; offset ++) {
				alpha[vs][offset] = (alpha[vs-1][offset]*f1+f0)*emit[Hvar.get(vs, offset)];
				sum += alpha[vs][offset];
			}
			loglik += log(sum);
		}
	}
	return loglik;
}

void hmm_scaffold::backward(vector < vector < uint32_t > > & cevents, vector < int32_t > & vpath) {
	float sum = 0.0f, scale = 0.0f;
	const uint32_t nstates_vectorized = (nstates / hsv::width) * hsv::width;
	aligned_vector32 < float > alphaXbeta_curr = aligned_vector32 < float >(nstates, 0.0f);
	aligned_vector32 < float > alphaXbeta_prev = aligned_vector32 < float >(nstates, 0.0f);

	//vpath = vector < int32_t > (C.n_scaffold_variants, -1);

	for (int32_t vs = C.n_scaffold_variants - 1 ; vs >= 0 ; vs --) {

		//
		const std::array<float,2> emit = {match_prob[C.Hhap.get(hap, vs)], match_prob[1-C.Hhap.get(hap, vs)]};
		const hsv::float8 _emit0 = hsv::broadcast(emit[0]);
		const hsv::float8 _emit1 = hsv::broadcast(emit[1]);

		//Transitions
		if (vs == C.n_scaffold_variants - 1) fill (beta.begin(), beta.end(), 1.0f / nstates);
		else {
			const float f0 = M.t[vs] / nstates;
			const float f1 = M.nt[vs] / sum;
			const hsv::float8 _f0 = hsv::broadcast(f0);
			const hsv::float8 _f1 = hsv::broadcast(f1);
			int32_t offset = 0;
			for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
				const hsv::float8 _prob_prev = hsv::load_aligned(&beta[k]);
				const hsv::float8 _prob_curr = hsv::multiply_add(_prob_prev, _f1, _f0);
				hsv::store_aligned(&beta[k], _prob_curr);
				offset += hsv::width;
			}
			for (; offset < nstates ; offset ++) beta[offset] = (beta[offset]*f1+f0);
		}

		//Products
		hsv::float8 _scale = hsv::broadcast(0.0f);
		int32_t offset = 0;
		for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
			const hsv::float8 _prob_temp = hsv::multiply(hsv::load_aligned(&alpha[vs][k]), hsv::load_aligned(&beta[k]));
			hsv::store_aligned(&alphaXbeta_curr[k], _prob_temp);
			_scale = hsv::add(_scale, _prob_temp);
			offset += hsv::width;
		}
		scale = (offset > 0)?hsv::horizontal_add(_scale):0.0f;
		for (; offset < nstates ; offset ++) {
			alphaXbeta_curr[offset] = alpha[vs][offset] * beta[offset];
			scale += alphaXbeta_curr[offset];
		}
		scale = 1.0f / scale;
		_scale = hsv::broadcast(scale);
		offset = 0;
		for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
			const hsv::float8 _prob_temp = hsv::multiply(hsv::load_aligned(&alphaXbeta_curr[k]), _scale);
			hsv::store_aligned(&alphaXbeta_curr[k], _prob_temp);
			offset += hsv::width;
		}
		for (; offset < nstates ; offset ++) alphaXbeta_curr[offset] *= scale;

		//Emission
		hsv::float8 _sum = hsv::broadcast(0.0f);
		offset = 0;
		for (int32_t k = 0 ; k < nstates_vectorized ; k += hsv::width) {
			const hsv::float8 _emiss = hsv::select_packed_byte(Hvar.getByte(vs, k), _emit0, _emit1);
			const hsv::float8 _prob_prev = hsv::load_aligned(&beta[k]);
			const hsv::float8 _prob_curr = hsv::multiply(_prob_prev, _emiss);
			_sum = hsv::add(_sum, _prob_curr);
			hsv::store_aligned(&beta[k], _prob_curr);
			offset += hsv::width;
		}
		sum = (offset > 0)?hsv::horizontal_add(_sum):0.0f;
		for (; offset < nstates ; offset ++) {
			beta[offset] *= emit[Hvar.get(vs, offset)];
			sum += beta[offset];
		}

		//vpath[vs] = std::distance(alphaXbeta_curr.begin(),std::max_element(alphaXbeta_curr.begin(), alphaXbeta_curr.end()));

		//Storage
		if (cevents[vs+1].size()) {
			if (vs == C.n_scaffold_variants-1) copy(alphaXbeta_curr.begin(), alphaXbeta_curr.begin() + nstates, alphaXbeta_prev.begin());

			//Impute from full conditioning set
			for (int32_t vr = 0 ; vr < cevents[vs+1].size() ; vr ++) {
				G.phaseLiAndStephens(cevents[vs+1][vr], hap, alphaXbeta_prev, alphaXbeta_curr, C.indexes_pbwt_neighbour[hap], 0.5001f);
			}
		}

		//Saving products
		copy(alphaXbeta_curr.begin(), alphaXbeta_curr.begin() + nstates, alphaXbeta_prev.begin());
	}

	if (cevents[0].size()) {

		//Impute from full conditioning set
		for (int32_t vr = 0 ; vr < cevents[0].size() ; vr ++) {
			G.phaseLiAndStephens(cevents[0][vr], hap, alphaXbeta_curr, alphaXbeta_curr, C.indexes_pbwt_neighbour[hap], 0.5001f);
		}
	}
}

void hmm_scaffold::viterbi(vector < int32_t > & path) {
	float sum, scale, maxv_prev, maxv_curr;
	int32_t maxi_curr, maxi_prev;
	vector < vector < int32_t > > _viterbi_paths = vector < vector < int32_t > > (C.n_scaffold_variants, vector < int32_t > (nstates, 0));
	vector < float > _viterbi_probs = vector < float > (nstates, 0.0f);

	//FORWARD PASS
	for (int32_t vs = 0 ; vs < C.n_scaffold_variants ; vs ++) {
		const std::array < float, 2 > emit = { match_prob[C.Hhap.get(hap, vs)], match_prob[1-C.Hhap.get(hap, vs)] };

		if (!vs) {
			maxi_curr = -1;
			sum = maxv_curr = 0.0f;
			for (int32_t k = 0; k < nstates ; k ++) {
				_viterbi_probs[k] = emit[Hvar.get(vs, k)];
				if (_viterbi_probs[k] > maxv_curr) {
					maxv_curr = _viterbi_probs[k];
					maxi_curr = k;
				}
				sum += _viterbi_probs[k];
			}
		} else {
			maxi_curr = -1;
			scale = 1.0f / sum;
			sum = maxv_curr = 0.0f;

			for (int32_t k = 0 ; k < nstates ; k ++) {

				float prob_yrecomb = M.t[vs-1] * maxv_prev * scale;
				float prob_nrecomb = M.nt[vs-1] * _viterbi_probs[k] * scale;

				if (prob_yrecomb > prob_nrecomb) {		// I switch copying from the most likely state
					_viterbi_probs[k] = prob_yrecomb;
					_viterbi_paths[vs][k] = maxi_prev;
				} else {								// I stay copying from the same state
					_viterbi_probs[k] = prob_nrecomb;
					_viterbi_paths[vs][k] = k;
				}

				_viterbi_probs[k] *= emit[Hvar.get(vs, k)];

				if (_viterbi_probs[k] > maxv_curr) {
					maxv_curr = _viterbi_probs[k];
					maxi_curr = k;
				}

				sum += _viterbi_probs[k];
			}
		}

		maxi_prev = maxi_curr;
		maxv_prev = maxv_curr;
	}

	//BACKTRACKING PASS
	path = vector < int32_t > (C.n_scaffold_variants, maxi_curr);
	for (int32_t vs = C.n_scaffold_variants - 1 ; vs > 0; vs --)
		path[vs-1] = _viterbi_paths[vs][path[vs]];
}
