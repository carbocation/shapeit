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

#include <phaser/phaser_header.h>
#include <shapeit_common.h>

#include <io/genotype_reader/genotype_reader_header.h>
#include <io/haplotype_writer.h>
#include <io/gmap_reader.h>
#include <io/pedigree_reader.h>
#include <io/haploid_reader.h>

using namespace std;

void phaser::read_files_and_initialise() {
	//step0: Initialize seed
	rng.setSeed(options["seed"].as < int > ());

	//step1: Set up the genotype reader
	vrb.title("Reading genotype data:");
	genotype_reader readerG(H, G, V);
	readerG.setThreads(options["thread"].as < int > ());
	readerG.setRegion(options["region"].as < string > ());
	readerG.setMainFilename(options["input"].as < string > ());
	if (options.count("reference")) readerG.addReferenceFilename(options["reference"].as < string > ());
	if (options.count("scaffold")) readerG.addScaffoldFilename(options["scaffold"].as < string > ());
	if (options.count("filter-snp")) readerG.setFilterSNP();
	if (!options["filter-maf"].defaulted()) readerG.setFilterMAF(options["filter-maf"].as < double > ());

	//step2: Read the genotype data
	readerG.scanGenotypes();
	readerG.allocateGenotypes();
	readerG.readGenotypes();

	//step3: Read haploid samples
	if (options.count("haploids")) {
		haploid_reader readerH;
		readerH.readHaploidFile(options["haploids"].as < string > ());
		G.resetHaploidHeterozgotes(readerH.samples);
	}

	//step4: Read pedigrees and scaffold diploid kids
	if (options.count("pedigree")) {
		pedigree_reader readerP;
		readerP.readPedigreeFile(options["pedigree"].as < string > ());
		G.scaffoldUsingPedigrees(readerP);
	}

	//step5: Read and initialise genetic map
	vrb.title("Setting up genetic map:");
	if (options.count("map")) {
		gmap_reader readerGM;
		readerGM.readGeneticMapFile(options["map"].as < string > ());
		V.setGeneticMap(readerGM);
	} else V.setGeneticMap();
	M.initialise(V, options["hmm-ne"].as < int > (), (readerG.n_main_samples+readerG.n_ref_samples)*2);

	//step6: Initialize haplotype set
	vrb.title("Initializing data structures:");
	G.imputeMonomorphic(V);
	H.updateHaplotypes(G, true);
	H.transposeHaplotypes_H2V(true);

	//step7: Initialize PBWT for selecting states
	if (pbwt_auto) {
		unsigned int cumulative_sample_size = readerG.n_main_samples + readerG.n_ref_samples;
		pbwt_depth = max(min((int)round(9-log10(cumulative_sample_size)), 8), 2);
		pbwt_modulo = max(min((log(cumulative_sample_size) - log(50) + 1) * 0.01, 0.15), 0.005);
		vrb.bullet("PBWT parameters auto setting : [modulo = " + stb.str(pbwt_modulo, 3) + " / depth = " + stb.str(pbwt_depth, 3) + "]");
	} else {
		pbwt_depth = options["pbwt-depth"].as < int > ();
		pbwt_modulo = options["pbwt-modulo"].as < double > ();
	}

	H.initialize(V,	pbwt_modulo,
					options["pbwt-window"].as < double > (),
					options["pbwt-mdr"].as < double > (),
					pbwt_depth,
					options["pbwt-mac"].as < int > (),
					options["thread"].as < int > ());

	if (!options.count("pbwt-disable-init")) H.solve(&G);

	//step8: Initialize genotype structures
	G.build(options["thread"].as < int > ());

	//step9: Allocate persistent Rust workers for common-phasing iterations
	vector < shapeit_genotype_graph_v1 * > graphs(G.n_ind, nullptr);
	vector < uint8_t > haploid(G.n_ind, 0);
	for (int ind = 0 ; ind < G.n_ind ; ind ++) {
		graphs[ind] = G.vecG[ind]->Graph;
		haploid[ind] = G.vecG[ind]->isHaploid();
	}
	const uint32_t status = shapeit_common_workers_create_v1(
		options["thread"].as < int > (), graphs.data(), graphs.size(),
		haploid.data(), haploid.size(), &phase_workers);
	if (status != SHAPEIT_COMMON_STATUS_OK) {
		throw runtime_error("Rust common worker pool rejected initialization (status " +
			to_string(status) + ")");
	}
}
