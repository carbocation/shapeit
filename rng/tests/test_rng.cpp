#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <shapeit_rng.h>
#include <utils/random_number.h>

static void test_known_answer_vectors() {
	struct vector {
		std::array < uint32_t, 4 > counter;
		std::array < uint32_t, 2 > key;
		std::array < uint32_t, 4 > expected;
	};
	const std::array < vector, 3 > vectors {{
		{{0, 0, 0, 0}, {0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}, {UINT32_MAX, UINT32_MAX}, {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0}, {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}}
	}};

	for (const vector & test : vectors) {
		std::array < uint32_t, 4 > actual {};
		shapeit_rng_philox4x32_10_raw(test.counter.data(), test.key.data(), actual.data());
		assert(actual == test.expected);
	}
}

static void test_cpp_adapter() {
	assert(random_number_generator::abiVersion() == SHAPEIT_RNG_ABI_VERSION);

	random_number_generator pinned(42, RNG_DOMAIN_PHASE_COMMON_MCMC, 3, 5);
	const std::array < unsigned int, 5 > expected_ints {{2, 8, 3, 6, 5}};
	const std::array < uint64_t, 5 > expected_doubles {{
		0x3fe7bdbf090e5a49, 0x3fdf489671b8700c, 0x3fdd496dbef692f4,
		0x3fd9122674d2b246, 0x3fef6dbc0765fa4d
	}};
	for (int i = 0 ; i < expected_ints.size() ; ++i) {
		assert(pinned.getInt(17) == expected_ints[i]);
		assert(std::bit_cast < uint64_t >(pinned.getDouble()) == expected_doubles[i]);
	}

	random_number_generator left(42, RNG_DOMAIN_PHASE_COMMON_MCMC, 3, 5);
	random_number_generator right(42, RNG_DOMAIN_PHASE_COMMON_MCMC, 3, 5);
	for (int i = 0 ; i < 1000 ; ++i) {
		assert(left.getInt(17) == right.getInt(17));
		assert(std::bit_cast < uint64_t >(left.getDouble()) == std::bit_cast < uint64_t >(right.getDouble()));
	}

	random_number_generator full_range(91);
	bool saw_nonzero = false;
	for (int i = 0 ; i < 100 ; ++i)
		saw_nonzero |= full_range.getInt(0, std::numeric_limits < unsigned int >::max()) != 0;
	assert(saw_nonzero);

	std::vector < unsigned int > first(32), second(32);
	std::iota(first.begin(), first.end(), 0);
	std::iota(second.begin(), second.end(), 0);
	random_number_generator shuffle_left(1234, RNG_DOMAIN_PHASE_RARE_SELECTION_ORDER, 0, 0);
	random_number_generator shuffle_right(1234, RNG_DOMAIN_PHASE_RARE_SELECTION_ORDER, 0, 0);
	shuffle_left.shuffle(first.begin(), first.end());
	shuffle_right.shuffle(second.begin(), second.end());
	assert(first == second);
	const std::vector < unsigned int > expected_shuffle {
		4, 5, 12, 22, 14, 24, 7, 2, 1, 30, 16, 18, 11, 3, 27, 23,
		6, 25, 26, 31, 0, 13, 21, 8, 10, 20, 19, 29, 17, 9, 15, 28
	};
	assert(first == expected_shuffle);
	std::sort(first.begin(), first.end());
	for (unsigned int i = 0 ; i < first.size() ; ++i) assert(first[i] == i);
}

int main() {
	test_known_answer_vectors();
	test_cpp_adapter();
	return 0;
}
