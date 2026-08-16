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

void genotype::build() {
	static_assert(sizeof(unsigned long) == sizeof(uint64_t));

	size_t segment_count = 0;
	size_t ambiguous_count = 0;
	size_t missing_count = 0;
	uint32_t status = shapeit_genotype_graph_sizes_v1(
		Variants.data(), Variants.size(), n_variants,
		&segment_count, &ambiguous_count, &missing_count);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype graph sizing rejected the packed variants (status " +
			to_string(status) + ")");
	}

	Lengths.resize(segment_count);
	Ambiguous.resize(ambiguous_count);
	Diplotypes.resize(segment_count);
	status = shapeit_genotype_graph_build_v1(
		Variants.data(), Variants.size(), n_variants,
		Lengths.data(), Lengths.size(), Ambiguous.data(), Ambiguous.size(),
		reinterpret_cast<uint64_t *>(Diplotypes.data()), Diplotypes.size(),
		&n_transitions);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype graph builder rejected its output layout (status " +
			to_string(status) + ")");
	}

	n_segments = segment_count;
	n_ambiguous = ambiguous_count;
	n_missing = missing_count;
}
