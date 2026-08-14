/*******************************************************************************
 * Copyright (C) 2023-2025 Simone Rubinacci
 * Copyright (C) 2023-2025 Olivier Delaneau
 *
 * MIT Licence
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

#ifndef _RANDOM_NUMBER_H
#define _RANDOM_NUMBER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

#include <shapeit_rng.h>

enum random_number_domain : uint32_t {
	RNG_DOMAIN_SERIAL = SHAPEIT_RNG_DOMAIN_SERIAL,
	RNG_DOMAIN_PHASE_COMMON_PBWT_SITE = SHAPEIT_RNG_DOMAIN_PHASE_COMMON_PBWT_SITE,
	RNG_DOMAIN_PHASE_COMMON_WINDOW = SHAPEIT_RNG_DOMAIN_PHASE_COMMON_WINDOW,
	RNG_DOMAIN_PHASE_COMMON_MCMC = SHAPEIT_RNG_DOMAIN_PHASE_COMMON_MCMC,
	RNG_DOMAIN_PHASE_RARE_SELECTION_ORDER = SHAPEIT_RNG_DOMAIN_PHASE_RARE_SELECTION_ORDER,
	RNG_DOMAIN_PHASE_RARE_PBWT_SITE = SHAPEIT_RNG_DOMAIN_PHASE_RARE_PBWT_SITE,
	RNG_DOMAIN_PHASE_RARE_FALLBACK = SHAPEIT_RNG_DOMAIN_PHASE_RARE_FALLBACK,
	RNG_DOMAIN_PHASE_RARE_SOLVE_ORDER = SHAPEIT_RNG_DOMAIN_PHASE_RARE_SOLVE_ORDER
};

class random_number_generator {
private:
	uint64_t seed;
	uint32_t domain;
	uint32_t iteration;
	uint64_t item;
	uint64_t next_block;
	std::array < uint32_t, 4 > words;
	uint32_t next_word;

	void refill() {
		shapeit_rng_block_v1(seed, domain, iteration, item, next_block++, words.data());
		next_word = 0;
	}

	uint32_t nextUInt() {
		if (next_word == words.size()) refill();
		return words[next_word++];
	}

	void reset(uint64_t _seed, uint32_t _domain, uint32_t _iteration, uint64_t _item) {
		seed = _seed;
		domain = _domain;
		iteration = _iteration;
		item = _item;
		next_block = 0;
		next_word = words.size();
		words.fill(0);
	}

public:
	random_number_generator(uint64_t _seed = 15052011) {
		reset(_seed, RNG_DOMAIN_SERIAL, 0, 0);
	}

	random_number_generator(uint64_t _seed, random_number_domain _domain, uint32_t _iteration, uint64_t _item) {
		reset(_seed, _domain, _iteration, _item);
	}

	void setSeed(uint64_t _seed) {
		reset(_seed, RNG_DOMAIN_SERIAL, 0, 0);
	}

	uint64_t getSeed() const {
		return seed;
	}

	random_number_generator fork(random_number_domain _domain, uint32_t _iteration, uint64_t _item) const {
		return random_number_generator(seed, _domain, _iteration, _item);
	}

	static uint32_t abiVersion() {
		return shapeit_rng_abi_version();
	}

	unsigned int getInt(unsigned int imin, unsigned int imax) {
		static_assert(sizeof(unsigned int) == sizeof(uint32_t));
		assert(imin <= imax);
		const uint64_t range64 = uint64_t(imax) - uint64_t(imin) + 1;
		if (range64 == (uint64_t(1) << 32)) return nextUInt();

		const uint32_t range = uint32_t(range64);
		uint32_t value = nextUInt();
		uint64_t product = uint64_t(value) * uint64_t(range);
		uint32_t low = uint32_t(product);
		if (low < range) {
			const uint32_t threshold = uint32_t(0U - range) % range;
			while (low < threshold) {
				value = nextUInt();
				product = uint64_t(value) * uint64_t(range);
				low = uint32_t(product);
			}
		}
		return imin + unsigned(product >> 32);
	}

	unsigned int getInt(unsigned int isize) {
		assert(isize > 0);
		return getInt(0, isize - 1);
	}

	double getDouble(double fmin, double fmax) {
		return fmin + (fmax - fmin) * getDouble();
	}

	double getDouble() {
		const uint64_t bits = (uint64_t(nextUInt()) << 32) | uint64_t(nextUInt());
		return double(bits >> 11) * 0x1.0p-53;
	}

	bool flipCoin() {
		return getDouble() < 0.5;
	}

	int sample(std::vector < float > & vec, float sum) {
		float csum = vec[0];
		float u = getDouble() * sum;
		for (int i = 0; i < vec.size() - 1; ++i) {
			if ( u <= csum ) return i;
			csum += vec[i+1];
		}
		return vec.size() - 1;
	}

	int sample(std::vector < double > & vec, double sum) {
		double csum = vec[0];
		double u = getDouble() * sum;
		for (int i = 0; i < vec.size() - 1; ++i) {
			if ( u < csum ) return i;
			csum += vec[i+1];
		}
		return vec.size() - 1;
	}

	int sample4(const double * vec, double sum) {
		double csum = vec[0];
		double u = getDouble() * sum;
		for (int i = 0; i < 3; ++i) {
			if ( u < csum ) return i;
			csum += vec[i+1];
		}
		return 3;
	}

	template < class RandomIt >
	void shuffle(RandomIt first, RandomIt last) {
		typename std::iterator_traits < RandomIt >::difference_type size = std::distance(first, last);
		assert(size >= 0);
		assert(uint64_t(size) <= uint64_t(std::numeric_limits < unsigned int >::max()));
		for (decltype(size) i = size - 1 ; i > 0 ; --i)
			std::iter_swap(first + i, first + getInt(unsigned(i + 1)));
	}
};

#endif
