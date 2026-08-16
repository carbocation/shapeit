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

#include <io/graph_writer.h>
#include <shapeit_genotype.h>

#include <stdexcept>

using namespace std;

graph_writer::graph_writer(genotype_set & _G, variant_map & _V): G(_G), V(_V) {
}

graph_writer::~graph_writer() {
}

void graph_writer::string_write(output_file & fout, string & x) {
	size_t size_str = x.size();
	fout.write(reinterpret_cast<char*>(&size_str), sizeof(size_str));
	fout.write(reinterpret_cast<char*>(&x[0]), size_str);
}

void graph_writer::writeGraphs(string fname) {
	// Init
	tac.clock();
	output_file fd (fname);

	//Write variant map
	int n_variants = V.vec_pos.size();
	fd.write(reinterpret_cast<char*>(&n_variants), sizeof(n_variants));
	for (int l = 0 ; l < n_variants ; l ++) {
		string_write(fd, V.vec_pos[l]->chr);
		fd.write(reinterpret_cast<char*>(&V.vec_pos[l]->bp), sizeof(V.vec_pos[l]->bp));
		string_write(fd, V.vec_pos[l]->id);
		string_write(fd, V.vec_pos[l]->ref);
		string_write(fd, V.vec_pos[l]->alt);
		fd.write(reinterpret_cast<char*>(&V.vec_pos[l]->idx), sizeof(V.vec_pos[l]->idx));
	}

	//Write genotype graphs
	fd.write(reinterpret_cast<char*>(&G.n_ind), sizeof(G.n_ind));
	for (int g  = 0 ; g < G.n_ind ; g++) {
		static_assert(sizeof(unsigned long) == sizeof(uint64_t));
		const shapeit_genotype_graph_view_v1 graph = G.vecG[g]->graphView();
		if (graph.variant_count != G.vecG[g]->n_variants ||
			graph.segment_lengths_length != G.vecG[g]->n_segments ||
			graph.ambiguous_length != G.vecG[g]->n_ambiguous ||
			graph.missing_count != G.vecG[g]->n_missing ||
			graph.transition_count != G.vecG[g]->n_transitions) {
			throw runtime_error("Rust genotype graph returned inconsistent writer metadata");
		}
		// name
		string_write(fd, G.vecG[g]->name);
		// integers
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->index), sizeof(G.vecG[g]->index));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_segments), sizeof(G.vecG[g]->n_segments));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_variants), sizeof(G.vecG[g]->n_variants));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_ambiguous), sizeof(G.vecG[g]->n_ambiguous));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_missing), sizeof(G.vecG[g]->n_missing));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_transitions), sizeof(G.vecG[g]->n_transitions));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_stored_transitionProbs), sizeof(G.vecG[g]->n_stored_transitionProbs));
		fd.write(reinterpret_cast<char*>(&G.vecG[g]->n_storage_events), sizeof(G.vecG[g]->n_storage_events));

		// vectors
		if (graph.variants_length)
			fd.write(reinterpret_cast<const char *>(graph.variants), graph.variants_length);
		if (graph.ambiguous_length)
			fd.write(reinterpret_cast<const char *>(graph.ambiguous), graph.ambiguous_length);
		if (graph.diplotypes_length)
			fd.write(reinterpret_cast<const char *>(graph.diplotypes), graph.diplotypes_length * sizeof(uint64_t));
		if (graph.segment_lengths_length)
			fd.write(reinterpret_cast<const char *>(graph.segment_lengths), graph.segment_lengths_length * sizeof(uint16_t));
		shapeit_genotype_storage_view_v1 view = {};
		const uint32_t status = shapeit_genotype_graph_storage_borrow_v1(G.vecG[g]->Graph, &view);
		if (status != SHAPEIT_GENOTYPE_STATUS_OK ||
			view.transition_count != G.vecG[g]->n_transitions ||
			view.transition_mask_length != (view.transition_count + 7) / 8 ||
			view.transition_probabilities_length != G.vecG[g]->n_stored_transitionProbs ||
			view.missing_probabilities_length != G.vecG[g]->n_missing * HAP_NUMBER ||
			view.storage_events != G.vecG[g]->n_storage_events) {
			throw runtime_error("Rust genotype storage returned an invalid graph-writer view");
		}
		const vector<bool>::size_type transition_count = view.transition_count;
		fd.write(reinterpret_cast<const char *>(&transition_count), sizeof(transition_count));
		if (view.transition_mask_length)
			fd.write(reinterpret_cast<const char *>(view.transition_mask), view.transition_mask_length);
		if (view.transition_probabilities_length)
			fd.write(reinterpret_cast<const char *>(view.transition_probabilities), view.transition_probabilities_length * sizeof(float));
		if (view.missing_probabilities_length)
			fd.write(reinterpret_cast<const char *>(view.missing_probabilities), view.missing_probabilities_length * sizeof(float));
	}
	vrb.bullet("BIN writing [Compressed / N=" + stb.str(G.n_ind) + " / L=" + stb.str(V.size()) + "] (" + stb.str(tac.rel_time()*0.001, 2) + "s)");
}
