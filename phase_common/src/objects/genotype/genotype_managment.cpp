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

#include <stdexcept>

using namespace std;

genotype::genotype(unsigned int _index) {
	index = _index;
	n_variants = 0;
	Variants = {};
	Graph = nullptr;
	this->name = "";
}

genotype::~genotype() {
	free();
}

void genotype::free() {
	Variants = {};
	if (Graph != nullptr) {
		shapeit_genotype_graph_free_v1(Graph);
		Graph = nullptr;
	}
	name = "";
}

void genotype::allocateVariants(unsigned int variant_count) {
	if (Graph != nullptr) throw runtime_error("Genotype variants have already been allocated");
	n_variants = variant_count;
	uint32_t status = shapeit_genotype_graph_allocate_v1(n_variants, &Graph);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype variant allocation failed (status " +
			to_string(status) + ")");
	}
	uint8_t * variants = nullptr;
	size_t variants_length = 0;
	status = shapeit_genotype_graph_variants_mut_v1(Graph, &variants, &variants_length);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		shapeit_genotype_graph_free_v1(Graph);
		Graph = nullptr;
		throw runtime_error("Rust genotype variant borrow failed (status " +
			to_string(status) + ")");
	}
	Variants = span < unsigned char > (variants, variants_length);
}

shapeit_genotype_graph_view_v1 genotype::graphView() const {
	if (Graph == nullptr) throw runtime_error("Genotype graph has not been built");
	shapeit_genotype_graph_view_v1 view = {};
	const uint32_t status = shapeit_genotype_graph_borrow_v1(Graph, &view);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype graph returned an invalid view (status " +
			to_string(status) + ")");
	}
	return view;
}

span < const unsigned char > genotype::packedVariants() const {
	return span < const unsigned char > (Variants.data(), Variants.size());
}

bool genotype::isHaploid() const {
	uint8_t haploid = 0, double_precision = 0;
	const uint32_t status = shapeit_genotype_graph_flags_v1(Graph, &haploid, &double_precision);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype graph flags are unavailable (status " +
			to_string(status) + ")");
	}
	return haploid != 0;
}

void genotype::setHaploid() {
	const uint32_t status = shapeit_genotype_graph_set_haploid_v1(Graph, 1);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype haploid flag update failed (status " +
			to_string(status) + ")");
	}
}

bool genotype::requiresDoublePrecision() const {
	uint8_t haploid = 0, double_precision = 0;
	const uint32_t status = shapeit_genotype_graph_flags_v1(Graph, &haploid, &double_precision);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype graph flags are unavailable (status " +
			to_string(status) + ")");
	}
	return double_precision != 0;
}

void genotype::requireDoublePrecision() {
	const uint32_t status = shapeit_genotype_graph_require_double_v1(Graph);
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust genotype precision flag update failed (status " +
			to_string(status) + ")");
	}
}
