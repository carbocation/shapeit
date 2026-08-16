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

#ifndef _COMPUTE_THREAD_H
#define _COMPUTE_THREAD_H

#include <cstdint>

#include <utils/otools.h>

#include <containers/conditioning_set/conditioning_set_header.h>
#include <containers/genotype_set.h>

struct shapeit_conditioning_job_v1;
class hmm_parameters;

class compute_job {
public:

	//DATA
	conditioning_set & H;
	shapeit_conditioning_job_v1 * Conditioning;
	std::vector < uint8_t > Haploid;

	compute_job(genotype_set &, conditioning_set &);
	compute_job(const compute_job &);
	~compute_job();

	void free();
	int run(unsigned int, genotype *, hmm_parameters &, double, unsigned int, double,
		random_number_generator &, random_number_generator &, random_number_generator &,
		int &, int &);
	void windowStats(size_t, int &, int &, size_t &, bool &);
	size_t size();
};

#endif
