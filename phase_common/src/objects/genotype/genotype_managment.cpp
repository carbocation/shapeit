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
	n_segments = 0;
	n_variants = 0;
	n_ambiguous = 0;
	n_missing = 0;
	n_transitions = 0;
	n_stored_transitionProbs = 0;
	n_storage_events = 0;
	Graph = nullptr;
	this->name = "";
	double_precision = false;
	haploid = false;
}

genotype::~genotype() {
	free();
}

void genotype::free() {
	if (Graph != nullptr) {
		shapeit_genotype_graph_free_v1(Graph);
		Graph = nullptr;
	}
	name = "";
	vector < unsigned char > ().swap(Variants);
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
	if (Graph == nullptr) return span < const unsigned char > (Variants.data(), Variants.size());
	const shapeit_genotype_graph_view_v1 view = graphView();
	return span < const unsigned char > (view.variants, view.variants_length);
}
