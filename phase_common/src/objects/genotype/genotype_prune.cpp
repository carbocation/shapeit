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

#include <cstdint>
#include <stdexcept>

using namespace std;

void genotype::prune(vector < double > & probabilities, double threshold_probability_mass) {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));
	vector < unsigned char > ambiguous2(Ambiguous.size(), 0);
	vector < unsigned long > diplotypes2(n_segments, 0);
	vector < unsigned short > lengths2(n_segments, 0);
	size_t segment_count2 = 0;
	uint32_t transition_count2 = 0;
	uint32_t status = shapeit_genotype_prune_v1(
		Variants.data(), Variants.size(), n_variants,
		Ambiguous.data(), Ambiguous.size(),
		reinterpret_cast<const uint64_t *>(Diplotypes.data()), Diplotypes.size(),
		Lengths.data(), Lengths.size(), probabilities.data(), n_transitions,
		threshold_probability_mass,
		ambiguous2.data(), ambiguous2.size(),
		reinterpret_cast<uint64_t *>(diplotypes2.data()), diplotypes2.size(),
		lengths2.data(), lengths2.size(), &segment_count2, &transition_count2);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype pruning rejected its graph (status " +
			to_string(status) + ")");
	}

	diplotypes2.resize(segment_count2);
	lengths2.resize(segment_count2);
	Ambiguous.swap(ambiguous2);
	Diplotypes.swap(diplotypes2);
	Lengths.swap(lengths2);
	n_segments = segment_count2;
	n_transitions = transition_count2;
}
