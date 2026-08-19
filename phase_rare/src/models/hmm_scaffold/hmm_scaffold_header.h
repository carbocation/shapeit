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

#ifndef _HMM_SCAFFOLD_H
#define _HMM_SCAFFOLD_H

#include <utils/otools.h>
#include <containers/conditioning_set/conditioning_set_header.h>
#include <containers/genotype_set/genotype_set_header.h>
#include <objects/hmm_parameters.h>
#include <containers/state_set.h>

class hmm_scaffold {
public:
	//DATA
	conditioning_set & C;
	genotype_set & G;
	variant_map & V;
	hmm_parameters & M;

	//CONSTANT
	uint32_t hap;
	float match_prob[2];
	uint32_t nstates;

	//
	bitmatrix Hvar;
	bitmatrix Hhap;

	//ARRAYS
	std::vector < aligned_vector32 < float > > alpha;
	aligned_vector32 < float > beta;

public:
	//CONSTRUCTOR/DESTRUCTOR
	hmm_scaffold(variant_map & _V, genotype_set & _G, conditioning_set & _C, hmm_parameters & _M);
	~hmm_scaffold();

	void setup(uint32_t _hap);
	double forward();
	void backward(std::vector < std::vector < uint32_t > > & cevents, std::vector < int32_t > & vpath);
	void viterbi(std::vector < int32_t > & path);

};

#endif







