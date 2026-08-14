#include <htslib/hts.h>
#include <htslib/vcf.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
	std::cerr << "subset_bcf: " << message << '\n';
	std::exit(EXIT_FAILURE);
}

std::vector<size_t> selected_indexes(size_t total, size_t maximum) {
	if (total == 0) return {};
	const size_t count = std::min(total, maximum);
	std::vector<size_t> selected;
	selected.reserve(count);
	if (count == 1) {
		selected.push_back(total / 2);
		return selected;
	}
	for (size_t i = 0; i < count; ++i) {
		selected.push_back(static_cast<size_t>(std::llround(
			static_cast<double>(i) * (total - 1) / (count - 1))));
	}
	return selected;
}

double update_allele_counts(bcf_hdr_t* header, bcf1_t* record) {
	int32_t* genotypes = nullptr;
	int capacity = 0;
	const int values = bcf_get_genotypes(header, record, &genotypes, &capacity);
	if (values < 0) fail("record has no readable GT field");

	int32_t ac = 0;
	int32_t an = 0;
	for (int i = 0; i < values; ++i) {
		if (genotypes[i] == bcf_int32_vector_end) continue;
		if (bcf_gt_is_missing(genotypes[i])) continue;
		ac += bcf_gt_allele(genotypes[i]) == 1;
		++an;
	}
	std::free(genotypes);

	bcf_update_info_int32(header, record, "AC", &ac, 1);
	bcf_update_info_int32(header, record, "AN", &an, 1);
	if (an == 0) return -1.0;
	const double af = static_cast<double>(ac) / an;
	return std::min(af, 1.0 - af);
}

bool subset_and_test(
	const bcf_hdr_t* input_header,
	bcf_hdr_t* output_header,
	bcf1_t* record,
	int sample_count,
	int* sample_map,
	double min_maf) {
	bcf_unpack(record, BCF_UN_STR);
	if (record->n_allele != 2) return false;
	if (bcf_subset(input_header, record, sample_count, sample_map) < 0)
		fail("cannot subset a record");
	return update_allele_counts(output_header, record) >= min_maf;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 7) {
		std::cerr
			<< "usage: subset_bcf INPUT.bcf OUTPUT.bcf REGION SAMPLE_COUNT "
			   "MAX_RECORDS MIN_MAF\n";
		return EXIT_FAILURE;
	}

	const std::string input_path = argv[1];
	const std::string output_path = argv[2];
	const std::string region = argv[3];
	const int requested_samples = std::stoi(argv[4]);
	const size_t maximum_records = std::stoul(argv[5]);
	const double min_maf = std::stod(argv[6]);
	if (requested_samples < 1) fail("SAMPLE_COUNT must be positive");
	if (maximum_records < 1) fail("MAX_RECORDS must be positive");

	htsFile* input = bcf_open(input_path.c_str(), "r");
	if (!input) fail("cannot open " + input_path);
	bcf_hdr_t* input_header = bcf_hdr_read(input);
	if (!input_header) fail("cannot read the input header");
	hts_idx_t* index = bcf_index_load(input_path.c_str());
	if (!index) fail("cannot load the input BCF index");

	const int sample_count = std::min(requested_samples, bcf_hdr_nsamples(input_header));
	std::vector<char*> samples(sample_count);
	for (int i = 0; i < sample_count; ++i) samples[i] = input_header->samples[i];
	std::vector<int> sample_map(sample_count);
	bcf_hdr_t* output_header =
		bcf_hdr_subset(input_header, sample_count, samples.data(), sample_map.data());
	if (!output_header) fail("cannot construct the subset header");

	hts_itr_t* iterator = bcf_itr_querys(index, input_header, region.c_str());
	if (!iterator) fail("cannot query region " + region);
	bcf1_t* record = bcf_init();
	size_t eligible_count = 0;
	while (bcf_itr_next(input, iterator, record) >= 0) {
		eligible_count += subset_and_test(
			input_header, output_header, record, sample_count, sample_map.data(), min_maf);
	}
	bcf_itr_destroy(iterator);
	if (eligible_count == 0) fail("no eligible records in " + region);

	const std::vector<size_t> selected = selected_indexes(eligible_count, maximum_records);
	htsFile* output = bcf_open(output_path.c_str(), "wb");
	if (!output) fail("cannot open " + output_path + " for writing");
	if (bcf_hdr_write(output, output_header) < 0) fail("cannot write the output header");

	iterator = bcf_itr_querys(index, input_header, region.c_str());
	if (!iterator) fail("cannot query region on the second pass");
	size_t eligible_index = 0;
	size_t selected_index = 0;
	while (selected_index < selected.size() && bcf_itr_next(input, iterator, record) >= 0) {
		if (!subset_and_test(
				input_header, output_header, record, sample_count, sample_map.data(), min_maf))
			continue;
		if (eligible_index == selected[selected_index]) {
			if (bcf_write(output, output_header, record) < 0) fail("cannot write a record");
			++selected_index;
		}
		++eligible_index;
	}

	bcf_destroy(record);
	bcf_itr_destroy(iterator);
	bcf_hdr_destroy(output_header);
	bcf_hdr_destroy(input_header);
	hts_idx_destroy(index);
	if (bcf_close(input) < 0) fail("cannot close the input");
	if (bcf_close(output) < 0) fail("cannot close the output");
	if (bcf_index_build3(output_path.c_str(), nullptr, 14, 1) < 0)
		fail("cannot index the output");

	std::cout << output_path << ": " << sample_count << " samples, "
			  << selected.size() << " of " << eligible_count << " eligible records\n";
	return EXIT_SUCCESS;
}
