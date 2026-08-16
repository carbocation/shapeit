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

#ifndef _GENOTYPE_H
#define _GENOTYPE_H

#include <utils/otools.h>
#include <shapeit_genotype.h>

//Macros for packing/unpacking variants
#define VAR_GET_HOM(e,v)	((((v)>>((e)<<2))&3)==0)
#define VAR_GET_MIS(e,v)	((((v)>>((e)<<2))&3)==1)
#define VAR_GET_HET(e,v)	((((v)>>((e)<<2))&3)==2)
#define VAR_GET_SCA(e,v)	((((v)>>((e)<<2))&3)==3)
//#define VAR_GET_AMB(e,v)	((((v)>>((e)<<2))&3)!=0)
#define VAR_GET_AMB(e,v)	((((v)>>((e)<<2))&3)>1)
#define VAR_SET_HOM(e,v)	((e)?((v)&=207):((v)&=252))
#define VAR_SET_MIS(e,v)	((v)|=(1<<((e)<<2)))
#define VAR_SET_HET(e,v)	((v)|=(2<<((e)<<2)))
#define VAR_SET_SCA(e,v)	((v)|=(3<<((e)<<2)))

#define VAR_GET_HAP0(e,v)	(((v)&(4<<((e)<<2)))!=0)
#define VAR_SET_HAP0(e,v)	((v)|=(4<<((e)<<2)))
#define VAR_CLR_HAP0(e,v)	((e)?((v)&=191):((v)&=251))
#define VAR_GET_HAP1(e,v)	(((v)&(8<<((e)<<2)))!=0)
#define VAR_SET_HAP1(e,v)	((v)|=(8<<((e)<<2)))
#define VAR_CLR_HAP1(e,v)	((e)?((v)&=127):((v)&=247))
class genotype {
public:
	// INTERNAL DATA
	std::string name;
	unsigned int index;						// Index in containers
	unsigned int n_segments;				// Number of segments
	unsigned int n_variants;				// Number of variants	(to iterate over Variants)
	unsigned int n_ambiguous;				// Number of ambiguous variants
	unsigned int n_missing;					// Number of missing
	unsigned int n_transitions;				// Number of transitions
	unsigned int n_stored_transitionProbs;	// Number of transition probabilities stored in memory
	unsigned int n_storage_events;			// Number of storage having been done
	bool double_precision;					//If I get underflows using float, move to double
	bool haploid;							//Is this sample haploid?

	// Packed input exists in C++ only until build(); the live graph is Rust-owned.
	std::vector < unsigned char > Variants;
	shapeit_genotype_graph_v1 * Graph;

	//METHODS
	genotype(unsigned int);
	~genotype();
	void free();
	void build();
	void sample(std::vector < double > &, std::vector < float > &, random_number_generator &);
	void prune(std::vector < double > &, double);
	void store(std::vector < double > &, std::vector < float > &);
	void solve();
	void scaffoldTrio(genotype *, genotype *, std::vector < unsigned int > &);
	void scaffoldDuoFather(genotype *, std::vector < unsigned int > &);
	void scaffoldDuoMother(genotype *, std::vector < unsigned int > &);
	uint32_t setHetsAsMissing();
	shapeit_genotype_graph_view_v1 graphView() const;
	std::span < const unsigned char > packedVariants() const;
};

#endif
