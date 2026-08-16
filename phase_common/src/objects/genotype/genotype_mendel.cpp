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

namespace {

void scaffoldPedigree(
	genotype * child,
	genotype * father,
	genotype * mother,
	uint32_t mode,
	vector < unsigned int > & counts) {
	static_assert(sizeof(unsigned int) == sizeof(uint32_t));
	if (counts.size() < 4) throw runtime_error("Pedigree count buffer must contain four values");

	const uint8_t * father_variants = father ? father->Variants.data() : nullptr;
	const size_t father_length = father ? father->Variants.size() : 0;
	const uint8_t * mother_variants = mother ? mother->Variants.data() : nullptr;
	const size_t mother_length = mother ? mother->Variants.size() : 0;
	uint32_t status = shapeit_genotype_pedigree_scaffold_v1(
		child->Variants.data(), child->Variants.size(), child->n_variants,
		father_variants, father_length, mother_variants, mother_length, mode,
		reinterpret_cast<uint32_t *>(counts.data()), counts.size());
	if (status != SHAPEIT_GENOTYPE_STATUS_OK) {
		throw runtime_error("Rust pedigree scaffolding rejected its inputs (status " +
			to_string(status) + ")");
	}
}

}

// counts[0]: observed Mendelian errors
// counts[1]: possible Mendelian errors
// counts[2]: heterozygotes scaffolded
// counts[3]: heterozygotes not scaffolded
void genotype::scaffoldTrio(genotype * gfather, genotype * gmother, vector < unsigned int > & counts) {
	scaffoldPedigree(this, gfather, gmother, SHAPEIT_GENOTYPE_PEDIGREE_TRIO, counts);
}

void genotype::scaffoldDuoFather(genotype * gfather, vector < unsigned int > & counts) {
	scaffoldPedigree(this, gfather, nullptr, SHAPEIT_GENOTYPE_PEDIGREE_FATHER, counts);
}

void genotype::scaffoldDuoMother(genotype * gmother, vector < unsigned int > & counts) {
	scaffoldPedigree(this, nullptr, gmother, SHAPEIT_GENOTYPE_PEDIGREE_MOTHER, counts);
}
