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

#ifndef _HAPLOTYPE_SEGMENT_SINGLE_H
#define _HAPLOTYPE_SEGMENT_SINGLE_H

#include <utils/otools.h>
#include <objects/compute_job.h>
#include <objects/hmm_parameters.h>

#include <immintrin.h>
#include <boost/align/aligned_allocator.hpp>

template <typename T>
using aligned_vector32 = std::vector<T, boost::alignment::aligned_allocator < T, 32 > >;

inline __m256 haplotype_sum8_ps(const __m256 (&sums)[8]) {
	const __m256 sum01 = _mm256_add_ps(sums[0], sums[1]);
	const __m256 sum23 = _mm256_add_ps(sums[2], sums[3]);
	const __m256 sum45 = _mm256_add_ps(sums[4], sums[5]);
	const __m256 sum67 = _mm256_add_ps(sums[6], sums[7]);
	return _mm256_add_ps(_mm256_add_ps(sum01, sum23), _mm256_add_ps(sum45, sum67));
}

class haplotype_segment_single {
private:
	//EXTERNAL DATA
	hmm_parameters & M;
	genotype * G;
	bitmatrix Hvar;

	//COORDINATES & CONSTANTS
	int segment_first;
	int segment_last;
	int locus_first;
	int locus_last;
	int ambiguous_first;
	int ambiguous_last;
	int missing_first;
	int missing_last;
	int transition_first;
	int transition_last;
	unsigned int n_cond_haps;
	unsigned int n_missing;

	//CURSORS
	int curr_segment_index;
	int curr_segment_locus;
	int curr_abs_locus;
	int prev_abs_locus;
	int curr_rel_locus;
	int curr_rel_locus_offset;
	int curr_abs_ambiguous;
	int curr_abs_transition;
	int curr_abs_missing;
	int curr_rel_missing;


	//DYNAMIC ARRAYS
	float probSumT;
	aligned_vector32 < float > prob;
	aligned_vector32 < float > probSumK;
	aligned_vector32 < float > probSumH;
	std::vector < aligned_vector32 < float > > Alpha;
	std::vector < aligned_vector32 < float > > AlphaSum;
	std::vector < int > AlphaLocus;
	aligned_vector32 < float > AlphaSumSum;
	std::vector < aligned_vector32 < float > > AlphaMissing;
	std::vector < aligned_vector32 < float > > AlphaSumMissing;
	float HProbs [HAP_NUMBER * HAP_NUMBER] __attribute__ ((aligned(32)));
	double DProbs [HAP_NUMBER * HAP_NUMBER * HAP_NUMBER * HAP_NUMBER] __attribute__ ((aligned(32)));

	//STATIC ARRAYS
	float sumHProbs;
	double sumDProbs;
	float g0[HAP_NUMBER], g1[HAP_NUMBER];
	float nt, yt;

	//INLINED AND UNROLLED ROUTINES
	void INIT_HOM();
	void INIT_AMB();
	void INIT_MIS();
	bool RUN_HOM(char);
	void RUN_AMB();
	void RUN_MIS();
	void COLLAPSE_HOM();
	void COLLAPSE_AMB();
	void COLLAPSE_MIS();
	void SUMK();
	void IMPUTE(std::vector < float > & );
	bool TRANS_HAP();
	bool TRANS_DIP_MULT();
	bool TRANS_DIP_ADD();
	void SET_FIRST_TRANS(std::vector < double > & );
	int SET_OTHER_TRANS(std::vector < double > & );

public:
	//CONSTRUCTOR/DESTRUCTOR
	haplotype_segment_single(genotype *, bitmatrix &, std::vector < unsigned int > &, window &, hmm_parameters &);
	~haplotype_segment_single();

	//void fetch();
	void forward();
	int backward(std::vector < double > &, std::vector < float > &);
};

/*******************************************************************************/
/*****************			HOMOZYGOUS GENOTYPE			************************/
/*******************************************************************************/


inline
void haplotype_segment_single::INIT_HOM() {
	bool ag = VAR_GET_HAP0(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
	bitmatrix_allele_cursor alleles(Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3));
	__m256 _sum = _mm256_set1_ps(0.0f);
	__m256 _emission[2];
	_emission[0] = _mm256_set1_ps(1.0f);
	_emission[1] = _mm256_set1_ps(M.ed/M.ee);
	for(int k = 0, i = 0 ; k != n_cond_haps ; ++k, i += HAP_NUMBER) {
		bool ah = alleles.next();

		//std::cout << curr_rel_locus << " " << curr_rel_locus_offset << " " << k << " " << n_cond_haps << std::endl;


		__m256 _prob = _emission[ag!=ah];
		_sum = _mm256_add_ps(_sum, _prob);
		_mm256_store_ps(&prob[i], _prob);
	}
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

inline
bool haplotype_segment_single::RUN_HOM(char rare_allele) {
	bool ag = VAR_GET_HAP0(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
	if (rare_allele < 0 || ag == rare_allele) {
		const unsigned char * allele_row = Hvar.bytes +
			static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
		__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
			_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
		__m256 _factor = _mm256_set1_ps(yt / (n_cond_haps * probSumT));
		__m256 _tFreq = _mm256_load_ps(&probSumH[0]);
		_tFreq = _mm256_mul_ps(_tFreq, _factor);
		__m256 _nt = _mm256_set1_ps(nt / probSumT);
		__m256 _mismatch = _mm256_set1_ps(M.ed/M.ee);
		//Avoid an unpredictable branch on the conditioning haplotype allele.
		__m256 _emission[2];
		_emission[0] = _mm256_set1_ps(1.0f);
		_emission[1] = _mismatch;
		const unsigned char * alleles = allele_row;
		float * __restrict prob_data = prob.data();
		int k = 0, i = 0;
		for( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			const unsigned char mismatches = ag ? static_cast<unsigned char>(~packed) : packed;
			if (mismatches == 0) {
				#pragma GCC unroll 8
				for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
					__m256 _prob = _mm256_load_ps(prob_data + i);
					_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
					_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
					_mm256_store_ps(prob_data + i, _prob);
				}
			} else if (mismatches == 0xff) {
				#pragma GCC unroll 8
				for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
					__m256 _prob = _mm256_load_ps(prob_data + i);
					_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
					_prob = _mm256_mul_ps(_prob, _mismatch);
					_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
					_mm256_store_ps(prob_data + i, _prob);
				}
			} else {
				#pragma GCC unroll 8
				for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
					const bool mismatch = (mismatches >> (7 - lane)) & 1;
					__m256 _prob = _mm256_load_ps(prob_data + i);
					_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
					_prob = _mm256_mul_ps(_prob, _emission[mismatch]);
					_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
					_mm256_store_ps(prob_data + i, _prob);
				}
			}
		}
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			__m256 _prob = _mm256_load_ps(prob_data + i);
			_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
			_prob = _mm256_mul_ps(_prob, _emission[ag!=ah]);
			_sums[0] = _mm256_add_ps(_sums[0], _prob);
			_mm256_store_ps(prob_data + i, _prob);
		}
		__m256 _sum = haplotype_sum8_ps(_sums);
		_mm256_store_ps(&probSumH[0], _sum);
		probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
		return true;
	}
	return false;
}

inline
void haplotype_segment_single::COLLAPSE_HOM() {
	bool ag = VAR_GET_HAP0(MOD2(curr_abs_locus), G->Variants[DIV2(curr_abs_locus)]);
	__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	__m256 _tFreq = _mm256_set1_ps(yt / n_cond_haps);					//Check divide by probSumT here!
	__m256 _nt = _mm256_set1_ps(nt / probSumT);
	__m256 _mismatch = _mm256_set1_ps(M.ed/M.ee);
	//Avoid an unpredictable branch on the conditioning haplotype allele.
	__m256 _emission[2];
	_emission[0] = _mm256_set1_ps(1.0f);
	_emission[1] = _mismatch;
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float * __restrict prob_sum_k = probSumK.data();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8) {
		const unsigned char packed = *alleles++;
		const unsigned char mismatches = ag ? static_cast<unsigned char>(~packed) : packed;
		if (mismatches == 0 || mismatches == 0xff) {
			const bool mismatch = mismatches != 0;
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				__m256 _prob = _mm256_set1_ps(prob_sum_k[k + lane]);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				if (mismatch) _prob = _mm256_mul_ps(_prob, _mismatch);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		} else {
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				const bool mismatch = (mismatches >> (7 - lane)) & 1;
				__m256 _prob = _mm256_set1_ps(prob_sum_k[k + lane]);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				_prob = _mm256_mul_ps(_prob, _emission[mismatch]);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		}
	}
	const unsigned char packed = k < n_cond_haps ? *alleles : 0;
	for ( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
		const bool ah = (packed >> (7 - (k & 7))) & 1;
		__m256 _prob = _mm256_set1_ps(prob_sum_k[k]);
		_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
		_prob = _mm256_mul_ps(_prob, _emission[ag!=ah]);
		_sums[0] = _mm256_add_ps(_sums[0], _prob);
		_mm256_store_ps(prob_data + i, _prob);
	}
	__m256 _sum = haplotype_sum8_ps(_sums);
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

/*******************************************************************************/
/*****************			HETEROZYGOUS GENOTYPE			********************/
/*******************************************************************************/

inline
void haplotype_segment_single::INIT_AMB() {
	unsigned char amb_code = G->Ambiguous[curr_abs_ambiguous];
	for (int h = 0 ; h < HAP_NUMBER ; h ++) {
		g0[h] = HAP_GET(amb_code,h)?M.ed/M.ee:1.0f;
		g1[h] = HAP_GET(amb_code,h)?1.0f:M.ed/M.ee;
	}
	__m256 _sum = _mm256_set1_ps(0.0f);
	__m256 _emit[2]; _emit[0] = _mm256_loadu_ps(&g0[0]); _emit[1] = _mm256_loadu_ps(&g1[0]);
	bitmatrix_allele_cursor alleles(Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3));
	for(int k = 0, i = 0 ; k != n_cond_haps ; ++k, i += HAP_NUMBER) {
		bool ah = alleles.next();
		__m256 _prob = _emit[ah];
		_sum = _mm256_add_ps(_sum, _prob);
		_mm256_store_ps(&prob[i], _prob);
	}
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

inline
void haplotype_segment_single::RUN_AMB() {
	unsigned char amb_code = G->Ambiguous[curr_abs_ambiguous];
	for (int h = 0 ; h < HAP_NUMBER ; h ++) {
		g0[h] = HAP_GET(amb_code,h)?M.ed/M.ee:1.0f;
		g1[h] = HAP_GET(amb_code,h)?1.0f:M.ed/M.ee;
	}
	__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	__m256 _factor = _mm256_set1_ps(yt / (n_cond_haps * probSumT));
	__m256 _tFreq = _mm256_load_ps(&probSumH[0]);
	_tFreq = _mm256_mul_ps(_tFreq, _factor);
	__m256 _nt = _mm256_set1_ps(nt / probSumT);
	__m256 _emit[2]; _emit[0] = _mm256_loadu_ps(&g0[0]); _emit[1] = _mm256_loadu_ps(&g1[0]);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	int k = 0, i = 0;
	for( ; k + 7 < n_cond_haps ; k += 8) {
		const unsigned char packed = *alleles++;
		if (packed == 0 || packed == 0xff) {
			const __m256 emission = _emit[packed != 0];
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				__m256 _prob = _mm256_load_ps(prob_data + i);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				_prob = _mm256_mul_ps(_prob, emission);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		} else {
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				const bool ah = (packed >> (7 - lane)) & 1;
				__m256 _prob = _mm256_load_ps(prob_data + i);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				_prob = _mm256_mul_ps(_prob, _emit[ah]);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		}
	}
	const unsigned char packed = k < n_cond_haps ? *alleles : 0;
	for( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
		const bool ah = (packed >> (7 - (k & 7))) & 1;
		__m256 _prob = _mm256_load_ps(prob_data + i);
		_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
		_prob = _mm256_mul_ps(_prob, _emit[ah]);
		_sums[0] = _mm256_add_ps(_sums[0], _prob);
		_mm256_store_ps(prob_data + i, _prob);
	}
	__m256 _sum = haplotype_sum8_ps(_sums);
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

/*
inline
void haplotype_segment_single::RUN_AMB() {
	unsigned char amb_code = G->Ambiguous[curr_abs_ambiguous];
	for (int h = 0 ; h < HAP_NUMBER ; h ++) {
		g0[h] = HAP_GET(amb_code,h)?M.ed/M.ee:1.0f;
		g1[h] = HAP_GET(amb_code,h)?1.0f:M.ed/M.ee;
	}
	__m256d _sum0 = _mm256_set1_pd(0.0f);
	__m256d _sum1 = _mm256_set1_pd(0.0f);
	__m256d _factor = _mm256_set1_pd(yt / (n_cond_haps * probSumT));
	__m256d _tFreq0 = _mm256_load_pd(&probSumH[0]);
	__m256d _tFreq1 = _mm256_load_pd(&probSumH[4]);
	_tFreq0 = _mm256_mul_pd(_tFreq0, _factor);
	_tFreq1 = _mm256_mul_pd(_tFreq1, _factor);
	__m256d _nt = _mm256_set1_pd(nt / probSumT);
	__m256d _emit0[2], _emit1[2];
	_emit0[0] = _mm256_loadu_pd(&g0[0]);
	_emit0[1] = _mm256_loadu_pd(&g1[0]);
	_emit1[0] = _mm256_loadu_pd(&g0[4]);
	_emit1[1] = _mm256_loadu_pd(&g1[4]);
	for(int k = 0, i = 0 ; k != n_cond_haps ; ++k, i += HAP_NUMBER) {
		bool ah = Hvar.get(curr_rel_locus+curr_rel_locus_offset, k);
		__m256d _prob0 = _mm256_load_pd(&prob[i+0]);
		__m256d _prob1 = _mm256_load_pd(&prob[i+4]);
		_prob0 = _mm256_fmadd_pd(_prob0, _nt, _tFreq0);
		_prob1 = _mm256_fmadd_pd(_prob1, _nt, _tFreq1);
		_prob0 = _mm256_mul_pd(_prob0, _emit0[ah]);
		_prob1 = _mm256_mul_pd(_prob1, _emit1[ah]);
		_sum0 = _mm256_add_pd(_sum0, _prob0);
		_sum1 = _mm256_add_pd(_sum1, _prob1);
		_mm256_store_pd(&prob[i+0], _prob0);
		_mm256_store_pd(&prob[i+4], _prob1);
	}
	_mm256_store_pd(&probSumH[0], _sum0);
	_mm256_store_pd(&probSumH[4], _sum1);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}
*/

inline
void haplotype_segment_single::COLLAPSE_AMB() {
	unsigned char amb_code = G->Ambiguous[curr_abs_ambiguous];
	for (int h = 0 ; h < HAP_NUMBER ; h ++) {
		g0[h] = HAP_GET(amb_code,h)?M.ed/M.ee:1.0f;
		g1[h] = HAP_GET(amb_code,h)?1.0f:M.ed/M.ee;
	}
	__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	__m256 _tFreq = _mm256_set1_ps(yt / n_cond_haps);
	__m256 _nt = _mm256_set1_ps(nt / probSumT);
	__m256 _emit[2]; _emit[0] = _mm256_loadu_ps(&g0[0]); _emit[1] = _mm256_loadu_ps(&g1[0]);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float * __restrict prob_sum_k = probSumK.data();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8) {
		const unsigned char packed = *alleles++;
		if (packed == 0 || packed == 0xff) {
			const __m256 emission = _emit[packed != 0];
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				__m256 _prob = _mm256_set1_ps(prob_sum_k[k + lane]);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				_prob = _mm256_mul_ps(_prob, emission);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		} else {
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
				const bool ah = (packed >> (7 - lane)) & 1;
				__m256 _prob = _mm256_set1_ps(prob_sum_k[k + lane]);
				_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
				_prob = _mm256_mul_ps(_prob, _emit[ah]);
				_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
				_mm256_store_ps(prob_data + i, _prob);
			}
		}
	}
	const unsigned char packed = k < n_cond_haps ? *alleles : 0;
	for ( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
		const bool ah = (packed >> (7 - (k & 7))) & 1;
		__m256 _prob = _mm256_set1_ps(prob_sum_k[k]);
		_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
		_prob = _mm256_mul_ps(_prob, _emit[ah]);
		_sums[0] = _mm256_add_ps(_sums[0], _prob);
		_mm256_store_ps(prob_data + i, _prob);
	}
	__m256 _sum = haplotype_sum8_ps(_sums);
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

/*******************************************************************************/
/*****************			MISSING GENOTYPE			************************/
/*******************************************************************************/

inline
void haplotype_segment_single::INIT_MIS() {
	fill(prob.begin(), prob.end(), 1.0f/(HAP_NUMBER * n_cond_haps));
	fill(probSumH.begin(), probSumH.end(), 1.0f/HAP_NUMBER);
	probSumT = 1.0f;
}

inline
void haplotype_segment_single::RUN_MIS() {
	__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	__m256 _factor = _mm256_set1_ps(yt / (n_cond_haps * probSumT));
	__m256 _tFreq = _mm256_load_ps(&probSumH[0]);
	_tFreq = _mm256_mul_ps(_tFreq, _factor);
	__m256 _nt = _mm256_set1_ps(nt / probSumT);
	float * __restrict prob_data = prob.data();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8) {
		#pragma GCC unroll 8
		for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
			__m256 _prob = _mm256_load_ps(prob_data + i);
			_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
			_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
			_mm256_store_ps(prob_data + i, _prob);
		}
	}
	for ( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
		__m256 _prob = _mm256_load_ps(prob_data + i);
		_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
		_sums[0] = _mm256_add_ps(_sums[0], _prob);
		_mm256_store_ps(prob_data + i, _prob);
	}
	__m256 _sum = haplotype_sum8_ps(_sums);
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

inline
void haplotype_segment_single::COLLAPSE_MIS() {
	__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	__m256 _tFreq = _mm256_set1_ps(yt / n_cond_haps);
	__m256 _nt = _mm256_set1_ps(nt / probSumT);
	float * __restrict prob_data = prob.data();
	const float * __restrict prob_sum_k = probSumK.data();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8) {
		#pragma GCC unroll 8
		for (int lane = 0 ; lane < 8 ; ++lane, i += HAP_NUMBER) {
			__m256 _prob = _mm256_set1_ps(prob_sum_k[k + lane]);
			_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
			_sums[lane] = _mm256_add_ps(_sums[lane], _prob);
			_mm256_store_ps(prob_data + i, _prob);
		}
	}
	for ( ; k < n_cond_haps ; ++k, i += HAP_NUMBER) {
		__m256 _prob = _mm256_set1_ps(prob_sum_k[k]);
		_prob = _mm256_fmadd_ps(_prob, _nt, _tFreq);
		_sums[0] = _mm256_add_ps(_sums[0], _prob);
		_mm256_store_ps(prob_data + i, _prob);
	}
	__m256 _sum = haplotype_sum8_ps(_sums);
	_mm256_store_ps(&probSumH[0], _sum);
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] + probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

/*******************************************************************************/
/*****************					SUM Ks				************************/
/*******************************************************************************/

inline
void haplotype_segment_single::SUMK() {
	const float * __restrict prob_data = prob.data();
	float * __restrict prob_sum_k = probSumK.data();
	for(int k = 0, i = 0 ; k != n_cond_haps ; ++k, i += HAP_NUMBER) {
		prob_sum_k[k] = prob_data[i+0] + prob_data[i+1] + prob_data[i+2] + prob_data[i+3] + prob_data[i+4] + prob_data[i+5] + prob_data[i+6] + prob_data[i+7];
	}
}

/*******************************************************************************/
/*****************		TRANSITION COMPUTATIONS			************************/
/*******************************************************************************/

inline
bool haplotype_segment_single::TRANS_HAP() {
	sumHProbs = 0.0f;
	unsigned int  curr_rel_segment_index = curr_segment_index-segment_first;
	yt = M.getForwardTransProb(AlphaLocus[curr_rel_segment_index - 1], prev_abs_locus);
	nt = 1.0f - yt;
	const float alpha_total = AlphaSumSum[curr_rel_segment_index - 1];
	const float fact1 = nt / alpha_total;
	const float * __restrict alpha_data = Alpha[curr_rel_segment_index-1].data();
	const float * __restrict alpha_sums = AlphaSum[curr_rel_segment_index-1].data();
	const float * __restrict beta_data = prob.data();
	for (int h1 = 0 ; h1 < HAP_NUMBER ; h1++) {
		__m256 _sums[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
			_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
		const float fact2 = (alpha_sums[h1] / alpha_total) * yt / n_cond_haps;
		int k = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			#pragma GCC unroll 8
			for (int lane = 0 ; lane < 8 ; ++lane) {
				__m256 _alpha = _mm256_set1_ps(alpha_data[(k+lane)*HAP_NUMBER + h1] * fact1 + fact2);
				__m256 _beta = _mm256_load_ps(beta_data + (k+lane)*HAP_NUMBER);
				_sums[lane] = _mm256_add_ps(_sums[lane], _mm256_mul_ps(_alpha, _beta));
			}
		}
		for ( ; k < n_cond_haps ; ++k) {
			__m256 _alpha = _mm256_set1_ps(alpha_data[k*HAP_NUMBER + h1] * fact1 + fact2);
			__m256 _beta = _mm256_load_ps(beta_data + k*HAP_NUMBER);
			_sums[0] = _mm256_add_ps(_sums[0], _mm256_mul_ps(_alpha, _beta));
		}
		__m256 _sum = haplotype_sum8_ps(_sums);
		_mm256_store_ps(&HProbs[h1*HAP_NUMBER], _sum);
		sumHProbs += HProbs[h1*HAP_NUMBER+0]+HProbs[h1*HAP_NUMBER+1]+HProbs[h1*HAP_NUMBER+2]+HProbs[h1*HAP_NUMBER+3]+HProbs[h1*HAP_NUMBER+4]+HProbs[h1*HAP_NUMBER+5]+HProbs[h1*HAP_NUMBER+6]+HProbs[h1*HAP_NUMBER+7];
	}
	return (std::isnan(sumHProbs) || std::isinf(sumHProbs) || sumHProbs < std::numeric_limits<float>::min());
}

inline
bool haplotype_segment_single::TRANS_DIP_MULT() {
	sumDProbs= 0.0f;
	double scaling = 1.0 / sumHProbs;
	int t = 0;
	for (unsigned long prev = G->Diplotypes[curr_segment_index-1]; prev; prev &= prev - 1) {
		const int pd = std::countr_zero(prev);
		for (unsigned long next = G->Diplotypes[curr_segment_index]; next; next &= next - 1) {
			const int nd = std::countr_zero(next);
			DProbs[t] = (((double)HProbs[DIP_HAP0(pd)*HAP_NUMBER+DIP_HAP0(nd)]) * scaling) * ((double)(HProbs[DIP_HAP1(pd)*HAP_NUMBER+DIP_HAP1(nd)]) * scaling);
			sumDProbs += DProbs[t];
			t++;
		}
	}
	return (std::isnan(sumDProbs) || std::isinf(sumDProbs) || sumDProbs < std::numeric_limits<double>::min());
}

inline
bool haplotype_segment_single::TRANS_DIP_ADD() {
	sumDProbs = 0.0f;
	double scaling = 1.0 / sumHProbs;
	int t = 0;
	for (unsigned long prev = G->Diplotypes[curr_segment_index-1]; prev; prev &= prev - 1) {
		const int pd = std::countr_zero(prev);
		for (unsigned long next = G->Diplotypes[curr_segment_index]; next; next &= next - 1) {
			const int nd = std::countr_zero(next);
			DProbs[t] = (((double)HProbs[DIP_HAP0(pd)*HAP_NUMBER+DIP_HAP0(nd)]) * scaling) + ((double)(HProbs[DIP_HAP1(pd)*HAP_NUMBER+DIP_HAP1(nd)]) * scaling);
			sumDProbs += DProbs[t];
			t++;
		}
	}
	return (std::isnan(sumDProbs) || std::isinf(sumDProbs) || sumDProbs < std::numeric_limits<double>::min());
}

inline
void haplotype_segment_single::IMPUTE(std::vector < float > & missing_probabilities) {
	__m256 _sum = _mm256_set1_ps(0.0f);
	__m256 _sumA [2]; _sumA[0] = _mm256_set1_ps(0.0f); _sumA[1] = _mm256_set1_ps(0.0f);
	__m256 _alphaSum = _mm256_load_ps(&AlphaSumMissing[curr_rel_missing][0]);
	__m256 _ones = _mm256_set1_ps(1.0f);
	_alphaSum = _mm256_div_ps(_ones, _alphaSum);
	bitmatrix_allele_cursor alleles(Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3));
	for(int k = 0, i = 0 ; k != n_cond_haps ; ++k, i += HAP_NUMBER) {
		bool ah = alleles.next();
		__m256 _prob = _mm256_load_ps(&prob[i]);
		__m256 _alpha = _mm256_load_ps(&AlphaMissing[curr_rel_missing][i]);
		_sum = _mm256_mul_ps(_mm256_mul_ps(_alpha, _alphaSum), _prob);
		_sumA[ah] = _mm256_add_ps(_sumA[ah], _sum);
	}
	float * prob0 = (float*)&_sumA[0];
	float * prob1 = (float*)&_sumA[1];
	for (int h = 0 ; h < HAP_NUMBER ; h ++) {
		missing_probabilities[curr_abs_missing * HAP_NUMBER + h] = prob1[h] / (prob0[h]+prob1[h]);
	}
}

#endif
