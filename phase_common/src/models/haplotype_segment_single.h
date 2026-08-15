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
#include <array>
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

constexpr std::array < unsigned short, 256 > make_haplotype_allele_masks2() {
	std::array < unsigned short, 256 > masks = {};
	for (unsigned int value = 0 ; value < masks.size() ; ++value)
		for (unsigned int k = 0 ; k < 8 ; ++k)
			if ((value >> (7-k)) & 1U) masks[value] |= 3U << (2*k);
	return masks;
}

inline constexpr auto haplotype_allele_masks2 = make_haplotype_allele_masks2();

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
	unsigned char prob_haps;
	bool use_avx512;


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
	std::vector < unsigned char > SegmentHapCount;
	std::vector < unsigned short > SegmentFirstAmbiguous;
	std::vector < unsigned short > SegmentLastAmbiguous;
	float HProbs [HAP_NUMBER * HAP_NUMBER] __attribute__ ((aligned(32)));
	double HProbsDouble [HAP_NUMBER * HAP_NUMBER] __attribute__ ((aligned(32)));
	double DProbs [HAP_NUMBER * HAP_NUMBER * HAP_NUMBER * HAP_NUMBER] __attribute__ ((aligned(32)));

	//STATIC ARRAYS
	float sumHProbs;
	double sumHProbsDouble;
	double sumDProbs;
	float g0[HAP_NUMBER], g1[HAP_NUMBER];
	float nt, yt;

	//INLINED AND UNROLLED ROUTINES
	void INIT_HOM();
	void INIT_AMB();
	void INIT_MIS();
	bool RUN_HOM(char);
	template <unsigned int N_HAPS> bool RUN_HOM_REDUCED(bool);
	__attribute__((target("avx512f,fma"))) bool RUN_HOM_REDUCED2_AVX512(bool);
	void RUN_AMB();
	template <unsigned int N_HAPS> void RUN_AMB_REDUCED(unsigned char);
	__attribute__((target("avx512f,fma"))) void RUN_AMB_REDUCED2_AVX512(unsigned char);
	void RUN_MIS();
	void COLLAPSE_HOM();
	void COLLAPSE_AMB();
	void COLLAPSE_MIS();
	void SUMK();
	void RESHAPE_HAPS(unsigned int);
	void EXPAND_HAPS();
	void IMPUTE(std::vector < float > & );
	bool TRANS_HAP();
	bool TRANS_HAP_DOUBLE();
	bool TRANS_DIP_MULT();
	bool TRANS_DIP_MULT_DOUBLE();
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
		const unsigned int n_haps = prob_haps;
		if (n_haps == 1) return RUN_HOM_REDUCED<1>(ag);
		if (n_haps == 2) return use_avx512 ? RUN_HOM_REDUCED2_AVX512(ag) : RUN_HOM_REDUCED<2>(ag);
		if (n_haps == 4) return RUN_HOM_REDUCED<4>(ag);
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

template <unsigned int N_HAPS>
inline
bool haplotype_segment_single::RUN_HOM_REDUCED(bool ag) {
	static_assert(N_HAPS == 1 || N_HAPS == 2 || N_HAPS == 4);
	assert(prob_haps == N_HAPS);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float factor = yt / (n_cond_haps * probSumT);
	const float nt_factor = nt / probSumT;
	const float mismatch = M.ed/M.ee;
	float unique_sums[N_HAPS] = {};
	if constexpr (N_HAPS == 1) {
		__m256 sums = _mm256_setzero_ps();
		const __m256 t_freq = _mm256_set1_ps(probSumH[0] * factor);
		const __m256 nt_vec = _mm256_set1_ps(nt_factor);
		const __m256 mismatch_vec = _mm256_set1_ps(mismatch);
		const __m256 ones = _mm256_set1_ps(1.0f);
		const __m256i shifts = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		const __m256i one = _mm256_set1_epi32(1);
		int k = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			const unsigned char mismatches = ag ? static_cast<unsigned char>(~packed) : packed;
			__m256 value = _mm256_load_ps(prob_data + k);
			value = _mm256_fmadd_ps(value, nt_vec, t_freq);
			if (mismatches == 0xff) {
				value = _mm256_mul_ps(value, mismatch_vec);
			} else if (mismatches != 0) {
				__m256i mask = _mm256_srlv_epi32(_mm256_set1_epi32(mismatches), shifts);
				mask = _mm256_slli_epi32(_mm256_and_si256(mask, one), 31);
				value = _mm256_mul_ps(value, _mm256_blendv_ps(ones, mismatch_vec, _mm256_castsi256_ps(mask)));
			}
			sums = _mm256_add_ps(sums, value);
			_mm256_store_ps(prob_data + k, value);
		}
		alignas(32) float sum_lanes[8];
		_mm256_store_ps(sum_lanes, sums);
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for ( ; k < n_cond_haps ; ++k) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			float value = __builtin_fmaf(prob_data[k], nt_factor, probSumH[0] * factor);
			if (ag != ah) value *= mismatch;
			sum_lanes[0] += value;
			prob_data[k] = value;
		}
		const float sum01 = sum_lanes[0] + sum_lanes[1];
		const float sum23 = sum_lanes[2] + sum_lanes[3];
		const float sum45 = sum_lanes[4] + sum_lanes[5];
		const float sum67 = sum_lanes[6] + sum_lanes[7];
		unique_sums[0] = (sum01 + sum23) + (sum45 + sum67);
	} else if constexpr (N_HAPS == 2) {
		__m256 sums03 = _mm256_setzero_ps();
		__m256 sums47 = _mm256_setzero_ps();
		const __m256 t_freq = _mm256_mul_ps(
			_mm256_setr_ps(probSumH[0], probSumH[1], probSumH[0], probSumH[1],
				probSumH[0], probSumH[1], probSumH[0], probSumH[1]), _mm256_set1_ps(factor));
		const __m256 nt_vec = _mm256_set1_ps(nt_factor);
		const __m256 mismatch_vec = _mm256_set1_ps(mismatch);
		const __m256 ones = _mm256_set1_ps(1.0f);
		const __m256i shifts03 = _mm256_setr_epi32(7, 7, 6, 6, 5, 5, 4, 4);
		const __m256i shifts47 = _mm256_setr_epi32(3, 3, 2, 2, 1, 1, 0, 0);
		const __m256i one = _mm256_set1_epi32(1);
		int k = 0, i = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			const unsigned char mismatches = ag ? static_cast<unsigned char>(~packed) : packed;
			__m256 value03 = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i), nt_vec, t_freq);
			__m256 value47 = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i + 8), nt_vec, t_freq);
			if (mismatches == 0xff) {
				value03 = _mm256_mul_ps(value03, mismatch_vec);
				value47 = _mm256_mul_ps(value47, mismatch_vec);
			} else if (mismatches != 0) {
				__m256i mask03 = _mm256_srlv_epi32(_mm256_set1_epi32(mismatches), shifts03);
				__m256i mask47 = _mm256_srlv_epi32(_mm256_set1_epi32(mismatches), shifts47);
				mask03 = _mm256_slli_epi32(_mm256_and_si256(mask03, one), 31);
				mask47 = _mm256_slli_epi32(_mm256_and_si256(mask47, one), 31);
				value03 = _mm256_mul_ps(value03, _mm256_blendv_ps(ones, mismatch_vec, _mm256_castsi256_ps(mask03)));
				value47 = _mm256_mul_ps(value47, _mm256_blendv_ps(ones, mismatch_vec, _mm256_castsi256_ps(mask47)));
			}
			sums03 = _mm256_add_ps(sums03, value03);
			sums47 = _mm256_add_ps(sums47, value47);
			_mm256_store_ps(prob_data + i, value03);
			_mm256_store_ps(prob_data + i + 8, value47);
			i += 16;
		}
		alignas(32) float sum_lanes[16];
		_mm256_store_ps(sum_lanes, sums03);
		_mm256_store_ps(sum_lanes + 8, sums47);
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for ( ; k < n_cond_haps ; ++k, i += N_HAPS) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
				float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
				if (ag != ah) value *= mismatch;
				sum_lanes[h] += value;
				prob_data[i + h] = value;
			}
		}
		for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
			const float sum01 = sum_lanes[h] + sum_lanes[2 + h];
			const float sum23 = sum_lanes[4 + h] + sum_lanes[6 + h];
			const float sum45 = sum_lanes[8 + h] + sum_lanes[10 + h];
			const float sum67 = sum_lanes[12 + h] + sum_lanes[14 + h];
			unique_sums[h] = (sum01 + sum23) + (sum45 + sum67);
		}
	} else {
		__m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
		const __m256 t_freq = _mm256_mul_ps(
			_mm256_setr_ps(probSumH[0], probSumH[1], probSumH[2], probSumH[3],
				probSumH[0], probSumH[1], probSumH[2], probSumH[3]), _mm256_set1_ps(factor));
		const __m256 nt_vec = _mm256_set1_ps(nt_factor);
		const __m256 mismatch_vec = _mm256_set1_ps(mismatch);
		int k = 0, i = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			const unsigned char mismatches = ag ? static_cast<unsigned char>(~packed) : packed;
			#pragma GCC unroll 4
			for (int pair = 0 ; pair < 4 ; ++pair, i += 8) {
				__m256 value = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i), nt_vec, t_freq);
				if (mismatches == 0xff) {
					value = _mm256_mul_ps(value, mismatch_vec);
				} else if (mismatches != 0) {
					const float e0 = ((mismatches >> (7 - 2*pair)) & 1) ? mismatch : 1.0f;
					const float e1 = ((mismatches >> (6 - 2*pair)) & 1) ? mismatch : 1.0f;
					value = _mm256_mul_ps(value, _mm256_setr_ps(e0, e0, e0, e0, e1, e1, e1, e1));
				}
				sums[pair] = _mm256_add_ps(sums[pair], value);
				_mm256_store_ps(prob_data + i, value);
			}
		}
		alignas(32) float sum_lanes[32];
		for (int pair = 0 ; pair < 4 ; ++pair) _mm256_store_ps(sum_lanes + pair*8, sums[pair]);
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for ( ; k < n_cond_haps ; ++k, i += N_HAPS) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
				float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
				if (ag != ah) value *= mismatch;
				sum_lanes[h] += value;
				prob_data[i + h] = value;
			}
		}
		for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
			const float sum01 = sum_lanes[h] + sum_lanes[4 + h];
			const float sum23 = sum_lanes[8 + h] + sum_lanes[12 + h];
			const float sum45 = sum_lanes[16 + h] + sum_lanes[20 + h];
			const float sum67 = sum_lanes[24 + h] + sum_lanes[28 + h];
			unique_sums[h] = (sum01 + sum23) + (sum45 + sum67);
		}
	}
	for (unsigned int h = 0 ; h < N_HAPS ; ++h) probSumH[h] = unique_sums[h];
	for (unsigned int h = N_HAPS ; h < HAP_NUMBER ; ++h) probSumH[h] = probSumH[h % N_HAPS];
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] +
		probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
	return true;
}

inline __attribute__((target("avx512f,fma")))
bool haplotype_segment_single::RUN_HOM_REDUCED2_AVX512(bool ag) {
	assert(prob_haps == 2);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float factor = yt / (n_cond_haps * probSumT);
	const float nt_factor = nt / probSumT;
	const float mismatch = M.ed/M.ee;
	alignas(64) const float t_freq_lanes[16] = {
		probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1],
		probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1]};
	const __m512 t_freq = _mm512_mul_ps(_mm512_load_ps(t_freq_lanes), _mm512_set1_ps(factor));
	const __m512 nt_vec = _mm512_set1_ps(nt_factor);
	const __m512 mismatch_vec = _mm512_set1_ps(mismatch);
	__m512 sums = _mm512_setzero_ps();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8, i += 16) {
		const unsigned char packed = *alleles++;
		const __mmask16 mismatch_mask = haplotype_allele_masks2[packed] ^ (ag ? 0xffffU : 0U);
		__m512 value = _mm512_fmadd_ps(_mm512_loadu_ps(prob_data + i), nt_vec, t_freq);
		value = _mm512_mask_mul_ps(value, mismatch_mask, value, mismatch_vec);
		sums = _mm512_add_ps(sums, value);
		_mm512_storeu_ps(prob_data + i, value);
	}
	alignas(64) float sum_lanes[16];
	_mm512_store_ps(sum_lanes, sums);
	const unsigned char packed = k < n_cond_haps ? *alleles : 0;
	for ( ; k < n_cond_haps ; ++k, i += 2) {
		const bool ah = (packed >> (7 - (k & 7))) & 1;
		for (unsigned int h = 0 ; h < 2 ; ++h) {
			float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
			if (ag != ah) value *= mismatch;
			sum_lanes[h] += value;
			prob_data[i + h] = value;
		}
	}
	for (unsigned int h = 0 ; h < 2 ; ++h) {
		const float sum01 = sum_lanes[h] + sum_lanes[2 + h];
		const float sum23 = sum_lanes[4 + h] + sum_lanes[6 + h];
		const float sum45 = sum_lanes[8 + h] + sum_lanes[10 + h];
		const float sum67 = sum_lanes[12 + h] + sum_lanes[14 + h];
		probSumH[h] = (sum01 + sum23) + (sum45 + sum67);
	}
	for (unsigned int h = 2 ; h < HAP_NUMBER ; ++h) probSumH[h] = probSumH[h % 2];
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] +
		probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
	return true;
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
	const unsigned int n_haps = prob_haps;
	if (n_haps == 1) {
		RUN_HOM_REDUCED<1>(HAP_GET(amb_code, 0));
		return;
	}
	if (n_haps == 2) return use_avx512 ? RUN_AMB_REDUCED2_AVX512(amb_code) : RUN_AMB_REDUCED<2>(amb_code);
	if (n_haps == 4) return RUN_AMB_REDUCED<4>(amb_code);
	EXPAND_HAPS();
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

template <unsigned int N_HAPS>
inline
void haplotype_segment_single::RUN_AMB_REDUCED(unsigned char amb_code) {
	static_assert(N_HAPS == 2 || N_HAPS == 4);
	assert(prob_haps == N_HAPS);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float factor = yt / (n_cond_haps * probSumT);
	const float nt_factor = nt / probSumT;
	const float mismatch = M.ed/M.ee;
	float unique_sums[N_HAPS] = {};
	if constexpr (N_HAPS == 2) {
		__m256 sums03 = _mm256_setzero_ps();
		__m256 sums47 = _mm256_setzero_ps();
		const __m256 t_freq = _mm256_mul_ps(
			_mm256_setr_ps(probSumH[0], probSumH[1], probSumH[0], probSumH[1],
				probSumH[0], probSumH[1], probSumH[0], probSumH[1]), _mm256_set1_ps(factor));
		const __m256 nt_vec = _mm256_set1_ps(nt_factor);
		const __m256 mismatch_vec = _mm256_set1_ps(mismatch);
		const __m256 ones = _mm256_set1_ps(1.0f);
		const __m256i shifts03 = _mm256_setr_epi32(7, 7, 6, 6, 5, 5, 4, 4);
		const __m256i shifts47 = _mm256_setr_epi32(3, 3, 2, 2, 1, 1, 0, 0);
		const __m256i graph_bits = _mm256_setr_epi32(
			HAP_GET(amb_code, 0), HAP_GET(amb_code, 1), HAP_GET(amb_code, 0), HAP_GET(amb_code, 1),
			HAP_GET(amb_code, 0), HAP_GET(amb_code, 1), HAP_GET(amb_code, 0), HAP_GET(amb_code, 1));
		const __m256i one = _mm256_set1_epi32(1);
		int k = 0, i = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			__m256 value03 = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i), nt_vec, t_freq);
			__m256 value47 = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i + 8), nt_vec, t_freq);
			__m256i mask03 = _mm256_srlv_epi32(_mm256_set1_epi32(packed), shifts03);
			__m256i mask47 = _mm256_srlv_epi32(_mm256_set1_epi32(packed), shifts47);
			mask03 = _mm256_slli_epi32(_mm256_xor_si256(_mm256_and_si256(mask03, one), graph_bits), 31);
			mask47 = _mm256_slli_epi32(_mm256_xor_si256(_mm256_and_si256(mask47, one), graph_bits), 31);
			value03 = _mm256_mul_ps(value03, _mm256_blendv_ps(ones, mismatch_vec, _mm256_castsi256_ps(mask03)));
			value47 = _mm256_mul_ps(value47, _mm256_blendv_ps(ones, mismatch_vec, _mm256_castsi256_ps(mask47)));
			sums03 = _mm256_add_ps(sums03, value03);
			sums47 = _mm256_add_ps(sums47, value47);
			_mm256_store_ps(prob_data + i, value03);
			_mm256_store_ps(prob_data + i + 8, value47);
			i += 16;
		}
		alignas(32) float sum_lanes[16];
		_mm256_store_ps(sum_lanes, sums03);
		_mm256_store_ps(sum_lanes + 8, sums47);
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for ( ; k < n_cond_haps ; ++k, i += N_HAPS) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
				float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
				if (ah != HAP_GET(amb_code, h)) value *= mismatch;
				sum_lanes[h] += value;
				prob_data[i + h] = value;
			}
		}
		for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
			const float sum01 = sum_lanes[h] + sum_lanes[2 + h];
			const float sum23 = sum_lanes[4 + h] + sum_lanes[6 + h];
			const float sum45 = sum_lanes[8 + h] + sum_lanes[10 + h];
			const float sum67 = sum_lanes[12 + h] + sum_lanes[14 + h];
			unique_sums[h] = (sum01 + sum23) + (sum45 + sum67);
		}
	} else {
		__m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
		const __m256 t_freq = _mm256_mul_ps(
			_mm256_setr_ps(probSumH[0], probSumH[1], probSumH[2], probSumH[3],
				probSumH[0], probSumH[1], probSumH[2], probSumH[3]), _mm256_set1_ps(factor));
		const __m256 nt_vec = _mm256_set1_ps(nt_factor);
		const __m128 emit0 = _mm_setr_ps(
			HAP_GET(amb_code, 0) ? mismatch : 1.0f, HAP_GET(amb_code, 1) ? mismatch : 1.0f,
			HAP_GET(amb_code, 2) ? mismatch : 1.0f, HAP_GET(amb_code, 3) ? mismatch : 1.0f);
		const __m128 emit1 = _mm_setr_ps(
			HAP_GET(amb_code, 0) ? 1.0f : mismatch, HAP_GET(amb_code, 1) ? 1.0f : mismatch,
			HAP_GET(amb_code, 2) ? 1.0f : mismatch, HAP_GET(amb_code, 3) ? 1.0f : mismatch);
		int k = 0, i = 0;
		for ( ; k + 7 < n_cond_haps ; k += 8) {
			const unsigned char packed = *alleles++;
			#pragma GCC unroll 4
			for (int pair = 0 ; pair < 4 ; ++pair, i += 8) {
				const bool ah0 = (packed >> (7 - 2*pair)) & 1;
				const bool ah1 = (packed >> (6 - 2*pair)) & 1;
				__m256 emission = _mm256_castps128_ps256(ah0 ? emit1 : emit0);
				emission = _mm256_insertf128_ps(emission, ah1 ? emit1 : emit0, 1);
				__m256 value = _mm256_fmadd_ps(_mm256_load_ps(prob_data + i), nt_vec, t_freq);
				value = _mm256_mul_ps(value, emission);
				sums[pair] = _mm256_add_ps(sums[pair], value);
				_mm256_store_ps(prob_data + i, value);
			}
		}
		alignas(32) float sum_lanes[32];
		for (int pair = 0 ; pair < 4 ; ++pair) _mm256_store_ps(sum_lanes + pair*8, sums[pair]);
		const unsigned char packed = k < n_cond_haps ? *alleles : 0;
		for ( ; k < n_cond_haps ; ++k, i += N_HAPS) {
			const bool ah = (packed >> (7 - (k & 7))) & 1;
			for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
				float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
				if (ah != HAP_GET(amb_code, h)) value *= mismatch;
				sum_lanes[h] += value;
				prob_data[i + h] = value;
			}
		}
		for (unsigned int h = 0 ; h < N_HAPS ; ++h) {
			const float sum01 = sum_lanes[h] + sum_lanes[4 + h];
			const float sum23 = sum_lanes[8 + h] + sum_lanes[12 + h];
			const float sum45 = sum_lanes[16 + h] + sum_lanes[20 + h];
			const float sum67 = sum_lanes[24 + h] + sum_lanes[28 + h];
			unique_sums[h] = (sum01 + sum23) + (sum45 + sum67);
		}
	}
	for (unsigned int h = 0 ; h < N_HAPS ; ++h) probSumH[h] = unique_sums[h];
	for (unsigned int h = N_HAPS ; h < HAP_NUMBER ; ++h) probSumH[h] = probSumH[h % N_HAPS];
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] +
		probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
}

inline __attribute__((target("avx512f,fma")))
void haplotype_segment_single::RUN_AMB_REDUCED2_AVX512(unsigned char amb_code) {
	assert(prob_haps == 2);
	const unsigned char * alleles = Hvar.bytes +
		static_cast<unsigned long>(curr_rel_locus + curr_rel_locus_offset) * (Hvar.n_cols >> 3);
	float * __restrict prob_data = prob.data();
	const float factor = yt / (n_cond_haps * probSumT);
	const float nt_factor = nt / probSumT;
	const float mismatch = M.ed/M.ee;
	alignas(64) const float t_freq_lanes[16] = {
		probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1],
		probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1], probSumH[0], probSumH[1]};
	const __m512 t_freq = _mm512_mul_ps(_mm512_load_ps(t_freq_lanes), _mm512_set1_ps(factor));
	const __m512 nt_vec = _mm512_set1_ps(nt_factor);
	const __m512 mismatch_vec = _mm512_set1_ps(mismatch);
	const unsigned short graph_mask = (HAP_GET(amb_code, 0) ? 0x5555U : 0U) |
		(HAP_GET(amb_code, 1) ? 0xaaaaU : 0U);
	__m512 sums = _mm512_setzero_ps();
	int k = 0, i = 0;
	for ( ; k + 7 < n_cond_haps ; k += 8, i += 16) {
		const unsigned char packed = *alleles++;
		const __mmask16 mismatch_mask = haplotype_allele_masks2[packed] ^ graph_mask;
		__m512 value = _mm512_fmadd_ps(_mm512_loadu_ps(prob_data + i), nt_vec, t_freq);
		value = _mm512_mask_mul_ps(value, mismatch_mask, value, mismatch_vec);
		sums = _mm512_add_ps(sums, value);
		_mm512_storeu_ps(prob_data + i, value);
	}
	alignas(64) float sum_lanes[16];
	_mm512_store_ps(sum_lanes, sums);
	const unsigned char packed = k < n_cond_haps ? *alleles : 0;
	for ( ; k < n_cond_haps ; ++k, i += 2) {
		const bool ah = (packed >> (7 - (k & 7))) & 1;
		for (unsigned int h = 0 ; h < 2 ; ++h) {
			float value = __builtin_fmaf(prob_data[i + h], nt_factor, probSumH[h] * factor);
			if (ah != HAP_GET(amb_code, h)) value *= mismatch;
			sum_lanes[h] += value;
			prob_data[i + h] = value;
		}
	}
	for (unsigned int h = 0 ; h < 2 ; ++h) {
		const float sum01 = sum_lanes[h] + sum_lanes[2 + h];
		const float sum23 = sum_lanes[4 + h] + sum_lanes[6 + h];
		const float sum45 = sum_lanes[8 + h] + sum_lanes[10 + h];
		const float sum67 = sum_lanes[12 + h] + sum_lanes[14 + h];
		probSumH[h] = (sum01 + sum23) + (sum45 + sum67);
	}
	for (unsigned int h = 2 ; h < HAP_NUMBER ; ++h) probSumH[h] = probSumH[h % 2];
	probSumT = probSumH[0] + probSumH[1] + probSumH[2] + probSumH[3] +
		probSumH[4] + probSumH[5] + probSumH[6] + probSumH[7];
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
	EXPAND_HAPS();
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
void haplotype_segment_single::RESHAPE_HAPS(unsigned int n_haps) {
	assert(n_haps == 1 || n_haps == 2 || n_haps == 4 || n_haps == HAP_NUMBER);
	if (prob_haps == n_haps) return;
	float * __restrict prob_data = prob.data();
	if (prob_haps == HAP_NUMBER && n_haps == 1) {
		for (unsigned int k = 1 ; k < n_cond_haps ; ++k) prob_data[k] = prob_data[k*HAP_NUMBER];
	} else if (prob_haps == HAP_NUMBER && n_haps == 2) {
		for (unsigned int k = 1 ; k < n_cond_haps ; ++k) {
			prob_data[k*2] = prob_data[k*HAP_NUMBER];
			prob_data[k*2 + 1] = prob_data[k*HAP_NUMBER + 1];
		}
	} else if (prob_haps == HAP_NUMBER && n_haps == 4) {
		for (unsigned int k = 1 ; k < n_cond_haps ; ++k)
			_mm_store_ps(prob_data + k*4, _mm_load_ps(prob_data + k*HAP_NUMBER));
	} else if (prob_haps == 1 && n_haps == 2) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k) {
			const float value = prob_data[k];
			prob_data[k*2] = value;
			prob_data[k*2 + 1] = value;
		}
	} else if (prob_haps == 1 && n_haps == 4) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k)
			_mm_store_ps(prob_data + k*4, _mm_set1_ps(prob_data[k]));
	} else if (prob_haps == 1 && n_haps == HAP_NUMBER) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k)
			_mm256_store_ps(prob_data + k*HAP_NUMBER, _mm256_set1_ps(prob_data[k]));
	} else if (prob_haps == 2 && n_haps == 4) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k) {
			const __m128 value = _mm_setr_ps(prob_data[k*2], prob_data[k*2 + 1],
				prob_data[k*2], prob_data[k*2 + 1]);
			_mm_store_ps(prob_data + k*4, value);
		}
	} else if (prob_haps == 2 && n_haps == HAP_NUMBER) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k) {
			const __m128 value = _mm_setr_ps(prob_data[k*2], prob_data[k*2 + 1],
				prob_data[k*2], prob_data[k*2 + 1]);
			__m256 row = _mm256_castps128_ps256(value);
			row = _mm256_insertf128_ps(row, value, 1);
			_mm256_store_ps(prob_data + k*HAP_NUMBER, row);
		}
	} else if (prob_haps == 4 && n_haps == HAP_NUMBER) {
		for (int k = n_cond_haps - 1 ; k >= 0 ; --k) {
			const __m128 value = _mm_load_ps(prob_data + k*4);
			_mm_store_ps(prob_data + k*HAP_NUMBER, value);
			_mm_store_ps(prob_data + k*HAP_NUMBER + 4, value);
		}
	} else {
		assert(false);
	}
	prob_haps = n_haps;
}

inline
void haplotype_segment_single::EXPAND_HAPS() {
	RESHAPE_HAPS(HAP_NUMBER);
}

inline
void haplotype_segment_single::SUMK() {
	const float * __restrict prob_data = prob.data();
	float * __restrict prob_sum_k = probSumK.data();
	if (prob_haps < HAP_NUMBER) {
		const unsigned int n_haps = prob_haps;
		const float multiplicity = static_cast<float>(HAP_NUMBER / n_haps);
		for (unsigned int k = 0 ; k < n_cond_haps ; ++k) {
			float sum = prob_data[k*n_haps];
			for (unsigned int h = 1 ; h < n_haps ; ++h) sum += prob_data[k*n_haps + h];
			prob_sum_k[k] = sum * multiplicity;
		}
		return;
	}
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
	const unsigned int prev_haps = SegmentHapCount[curr_rel_segment_index - 1];
	const unsigned int curr_haps = SegmentHapCount[curr_rel_segment_index];
	yt = M.getForwardTransProb(AlphaLocus[curr_rel_segment_index - 1], prev_abs_locus);
	nt = 1.0f - yt;
	const float alpha_total = AlphaSumSum[curr_rel_segment_index - 1];
	const float fact1 = nt / alpha_total;
	const float * __restrict alpha_data = Alpha[curr_rel_segment_index-1].data();
	const float * __restrict alpha_sums = AlphaSum[curr_rel_segment_index-1].data();
	const float * __restrict beta_data = prob.data();
	if (prev_haps < HAP_NUMBER || curr_haps < HAP_NUMBER) {
		for (unsigned int h1 = 0 ; h1 < prev_haps ; ++h1) {
			const float fact2 = (alpha_sums[h1] / alpha_total) * yt / n_cond_haps;
			float sums[HAP_NUMBER] = {};
			for (unsigned int k = 0 ; k < n_cond_haps ; ++k) {
				const float alpha = alpha_data[k*prev_haps + h1] * fact1 + fact2;
				for (unsigned int h2 = 0 ; h2 < curr_haps ; ++h2)
					sums[h2] = __builtin_fmaf(alpha, beta_data[k*curr_haps + h2], sums[h2]);
			}
			for (unsigned int h2 = 0 ; h2 < curr_haps ; ++h2) {
				HProbs[h1*HAP_NUMBER + h2] = sums[h2];
			}
		}
		//The graph retains all eight lane labels because their multiplicity is
		//part of the transition measure. Expand only the tiny 8x8 contraction;
		//the O(K) recurrence above remains quotient-compressed.
		sumHProbs = 0.0f;
		for (unsigned int h1 = 0 ; h1 < HAP_NUMBER ; ++h1)
			for (unsigned int h2 = 0 ; h2 < HAP_NUMBER ; ++h2) {
				const float value = HProbs[(h1 % prev_haps)*HAP_NUMBER + (h2 % curr_haps)];
				HProbs[h1*HAP_NUMBER + h2] = value;
				sumHProbs += value;
			}
		return (std::isnan(sumHProbs) || std::isinf(sumHProbs) ||
			sumHProbs < std::numeric_limits<float>::min());
	}
	__m256 _sums[HAP_NUMBER] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
		_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
	float fact2[HAP_NUMBER];
	for (int h1 = 0 ; h1 < HAP_NUMBER ; ++h1)
		fact2[h1] = (alpha_sums[h1] / alpha_total) * yt / n_cond_haps;
	for (int k = 0 ; k < n_cond_haps ; ++k) {
		const __m256 _beta = _mm256_load_ps(beta_data + k*HAP_NUMBER);
		#pragma GCC unroll 8
		for (int h1 = 0 ; h1 < HAP_NUMBER ; ++h1) {
			const __m256 _alpha = _mm256_set1_ps(alpha_data[k*HAP_NUMBER + h1] * fact1 + fact2[h1]);
			_sums[h1] = _mm256_fmadd_ps(_alpha, _beta, _sums[h1]);
		}
	}
	for (int h1 = 0 ; h1 < HAP_NUMBER ; ++h1) {
		_mm256_store_ps(&HProbs[h1*HAP_NUMBER], _sums[h1]);
		sumHProbs += HProbs[h1*HAP_NUMBER+0]+HProbs[h1*HAP_NUMBER+1]+HProbs[h1*HAP_NUMBER+2]+HProbs[h1*HAP_NUMBER+3]+HProbs[h1*HAP_NUMBER+4]+HProbs[h1*HAP_NUMBER+5]+HProbs[h1*HAP_NUMBER+6]+HProbs[h1*HAP_NUMBER+7];
	}
	return (std::isnan(sumHProbs) || std::isinf(sumHProbs) || sumHProbs < std::numeric_limits<float>::min());
}

inline
bool haplotype_segment_single::TRANS_HAP_DOUBLE() {
	sumHProbsDouble = 0.0;
	const unsigned int curr_rel_segment_index = curr_segment_index-segment_first;
	const unsigned int prev_haps = SegmentHapCount[curr_rel_segment_index - 1];
	const unsigned int curr_haps = SegmentHapCount[curr_rel_segment_index];
	yt = M.getForwardTransProb(AlphaLocus[curr_rel_segment_index - 1], prev_abs_locus);
	nt = 1.0f - yt;
	const double alpha_total = AlphaSumSum[curr_rel_segment_index - 1];
	const double fact1 = nt / alpha_total;
	const float * __restrict alpha_data = Alpha[curr_rel_segment_index-1].data();
	const float * __restrict alpha_sums = AlphaSum[curr_rel_segment_index-1].data();
	const float * __restrict beta_data = prob.data();
	if (prev_haps < HAP_NUMBER || curr_haps < HAP_NUMBER) {
		for (unsigned int h1 = 0 ; h1 < prev_haps ; ++h1) {
			const double fact2 = (alpha_sums[h1] / alpha_total) * yt / n_cond_haps;
			for (unsigned int h2 = 0 ; h2 < curr_haps ; ++h2) {
				double sum = 0.0;
				for (unsigned int k = 0 ; k < n_cond_haps ; ++k) {
					const double alpha = alpha_data[k*prev_haps + h1] * fact1 + fact2;
					sum = __builtin_fma(alpha, beta_data[k*curr_haps + h2], sum);
				}
				HProbsDouble[h1*HAP_NUMBER + h2] = sum;
			}
		}
		sumHProbsDouble = 0.0;
		for (unsigned int h1 = 0 ; h1 < HAP_NUMBER ; ++h1)
			for (unsigned int h2 = 0 ; h2 < HAP_NUMBER ; ++h2) {
				const double value = HProbsDouble[(h1 % prev_haps)*HAP_NUMBER + (h2 % curr_haps)];
				HProbsDouble[h1*HAP_NUMBER + h2] = value;
				sumHProbsDouble += value;
			}
		return (std::isnan(sumHProbsDouble) || std::isinf(sumHProbsDouble) ||
			sumHProbsDouble < std::numeric_limits<double>::min());
	}
	for (int h1 = 0 ; h1 < HAP_NUMBER ; ++h1) {
		__m256d sum0 = _mm256_setzero_pd();
		__m256d sum1 = _mm256_setzero_pd();
		const double fact2 = (alpha_sums[h1] / alpha_total) * yt / n_cond_haps;
		for (int k = 0 ; k < n_cond_haps ; ++k) {
			const double alpha = alpha_data[k*HAP_NUMBER + h1] * fact1 + fact2;
			const __m256d alpha4 = _mm256_set1_pd(alpha);
			const __m256d beta0 = _mm256_cvtps_pd(_mm_load_ps(beta_data + k*HAP_NUMBER));
			const __m256d beta1 = _mm256_cvtps_pd(_mm_load_ps(beta_data + k*HAP_NUMBER + 4));
			sum0 = _mm256_fmadd_pd(alpha4, beta0, sum0);
			sum1 = _mm256_fmadd_pd(alpha4, beta1, sum1);
		}
		_mm256_store_pd(HProbsDouble + h1*HAP_NUMBER, sum0);
		_mm256_store_pd(HProbsDouble + h1*HAP_NUMBER + 4, sum1);
		for (int h2 = 0 ; h2 < HAP_NUMBER ; ++h2)
			sumHProbsDouble += HProbsDouble[h1*HAP_NUMBER + h2];
	}
	return (std::isnan(sumHProbsDouble) || std::isinf(sumHProbsDouble) ||
		sumHProbsDouble < std::numeric_limits<double>::min());
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
bool haplotype_segment_single::TRANS_DIP_MULT_DOUBLE() {
	sumDProbs = 0.0;
	const double scaling = 1.0 / sumHProbsDouble;
	int t = 0;
	for (unsigned long prev = G->Diplotypes[curr_segment_index-1]; prev; prev &= prev - 1) {
		const int pd = std::countr_zero(prev);
		for (unsigned long next = G->Diplotypes[curr_segment_index]; next; next &= next - 1) {
			const int nd = std::countr_zero(next);
			DProbs[t] = (HProbsDouble[DIP_HAP0(pd)*HAP_NUMBER+DIP_HAP0(nd)] * scaling) *
				(HProbsDouble[DIP_HAP1(pd)*HAP_NUMBER+DIP_HAP1(nd)] * scaling);
			sumDProbs += DProbs[t];
			++t;
		}
	}
	return (std::isnan(sumDProbs) || std::isinf(sumDProbs) ||
		sumDProbs < std::numeric_limits<double>::min());
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
