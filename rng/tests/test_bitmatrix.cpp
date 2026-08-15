#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <shapeit_bitmatrix.h>

static std::vector < uint8_t > reference_transpose(
	const std::vector < uint8_t > & source,
	size_t source_stride,
	const std::vector < uint32_t > & rows,
	size_t source_byte_first,
	size_t source_byte_count) {
	const size_t padded_rows = (rows.size() + 7) & ~size_t(7);
	const size_t target_stride = padded_rows >> 3;
	std::vector < uint8_t > target(source_byte_count * 8 * target_stride, 0);
	for (size_t byte = 0 ; byte < source_byte_count ; ++byte) {
		for (size_t row = 0 ; row < padded_rows ; row += 8) {
			for (size_t bit = 0 ; bit < 8 ; ++bit) {
				uint8_t value = 0;
				for (size_t lane = 0 ; lane < 8 ; ++lane) {
					if (row + lane >= rows.size()) continue;
					const uint8_t source_byte = source[
						static_cast<size_t>(rows[row + lane]) * source_stride +
						source_byte_first + byte];
					value |= ((source_byte >> (7 - bit)) & 1) << (7 - lane);
				}
				target[(byte * 8 + bit) * target_stride + (row >> 3)] = value;
			}
		}
	}
	return target;
}

static std::vector < uint8_t > reference_full_transpose(
	const std::vector < uint8_t > & source,
	size_t source_stride,
	size_t max_rows,
	size_t max_cols,
	size_t target_stride,
	size_t target_length) {
	std::vector < uint8_t > target(target_length, 0xa5);
	for (size_t col = 0 ; col < max_cols ; ++col) {
		for (size_t row = 0 ; row < max_rows ; row += 8) {
			uint8_t value = 0;
			for (size_t lane = 0 ; lane < 8 ; ++lane) {
				const uint8_t source_byte = source[
					(row + lane) * source_stride + (col >> 3)];
				value |= ((source_byte >> (7 - (col & 7))) & 1) << (7 - lane);
			}
			target[col * target_stride + (row >> 3)] = value;
		}
	}
	return target;
}

static void test_reference_equivalence() {
	constexpr size_t source_rows = 137;
	constexpr size_t source_stride = 19;
	std::vector < uint8_t > source(source_rows * source_stride);
	for (size_t index = 0 ; index < source.size() ; ++index)
		source[index] = static_cast<uint8_t>(index * 73 + 41);

	for (const size_t row_count : {1U, 7U, 8U, 9U, 63U, 64U, 65U, 129U}) {
		std::vector < uint32_t > rows(row_count);
		for (size_t index = 0 ; index < row_count ; ++index)
			rows[index] = static_cast<uint32_t>((index * 37 + 11) % source_rows);
		for (const auto [source_byte_first, source_byte_count] :
			{std::pair<size_t, size_t>{0, 1}, {1, 8}, {7, 12}}) {
			const std::vector < uint8_t > expected = reference_transpose(
				source, source_stride, rows, source_byte_first, source_byte_count);
			const size_t target_stride = ((rows.size() + 7) & ~size_t(7)) >> 3;
			std::vector < uint8_t > actual(expected.size(), 0xa5);
			const uint32_t status = shapeit_bitmatrix_subset_transpose_v1(
				source.data(), source.size(), source_stride, rows.data(), rows.size(),
				source_byte_first, source_byte_count, actual.data(), actual.size(),
				target_stride);
			assert(status == SHAPEIT_BITMATRIX_STATUS_OK);
			assert(actual == expected);
		}
	}
}

static void test_full_reference_equivalence() {
	struct test_case {
		size_t source_rows;
		size_t source_stride;
		size_t max_rows;
		size_t max_cols;
	};
	for (const test_case test : {
		test_case{8, 1, 8, 8}, {16, 3, 8, 16},
		{72, 9, 64, 56}, {72, 9, 72, 72}}) {
		std::vector < uint8_t > source(test.source_rows * test.source_stride);
		for (size_t index = 0 ; index < source.size() ; ++index)
			source[index] = static_cast<uint8_t>(index * 73 + 41);
		const size_t target_stride = test.source_rows >> 3;
		const size_t target_length = test.source_stride * 8 * target_stride;
		const std::vector < uint8_t > expected = reference_full_transpose(
			source, test.source_stride, test.max_rows, test.max_cols,
			target_stride, target_length);
		std::vector < uint8_t > actual(target_length, 0xa5);
		const uint32_t status = shapeit_bitmatrix_transpose_v1(
			source.data(), source.size(), test.source_rows, test.source_stride,
			test.max_rows, test.max_cols, actual.data(), actual.size(), target_stride);
		assert(status == SHAPEIT_BITMATRIX_STATUS_OK);
		assert(actual == expected);
	}
}

static void test_invalid_layouts() {
	std::vector < uint8_t > source(32, 0);
	std::vector < uint8_t > target(16, 0);
	const std::vector < uint32_t > rows {0, 4};
	assert(shapeit_bitmatrix_subset_transpose_v1(
		source.data(), source.size(), 8, rows.data(), rows.size(), 0, 2,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS);
	assert(shapeit_bitmatrix_subset_transpose_v1(
		source.data(), source.size(), 8, rows.data(), rows.size(), 7, 2,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS);
	assert(shapeit_bitmatrix_subset_transpose_v1(
		source.data(), source.size(), 8, rows.data(), rows.size(), 0, 2,
		target.data(), target.size() - 1, 1) == SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS);
	assert(shapeit_bitmatrix_subset_transpose_v1(
		nullptr, source.size(), 8, rows.data(), rows.size(), 0, 2,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_NULL_POINTER);

	std::fill(target.begin(), target.end(), 0xa5);
	assert(shapeit_bitmatrix_transpose_v1(
		source.data(), source.size(), 8, 4, 8, 32,
		target.data(), target.size() - 1, 1) == SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS);
	assert(target == std::vector < uint8_t >(target.size(), 0xa5));
	assert(shapeit_bitmatrix_transpose_v1(
		source.data(), source.size(), 7, 4, 8, 32,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_INVALID_DIMENSIONS);
	assert(shapeit_bitmatrix_transpose_v1(
		source.data(), source.size(), 8, 4, 16, 32,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_OUT_OF_BOUNDS);
	assert(shapeit_bitmatrix_transpose_v1(
		nullptr, source.size(), 8, 4, 8, 32,
		target.data(), target.size(), 1) == SHAPEIT_BITMATRIX_STATUS_NULL_POINTER);
}

int main() {
	assert(shapeit_bitmatrix_abi_version() == SHAPEIT_BITMATRIX_ABI_VERSION);
	test_reference_equivalence();
	test_full_reference_equivalence();
	test_invalid_layouts();
	return 0;
}
