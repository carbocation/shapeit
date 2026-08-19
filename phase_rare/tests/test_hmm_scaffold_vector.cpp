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

#include <models/hmm_scaffold/hmm_scaffold_vector.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace hsv = hmm_scaffold_vector;

bool same_bits(const float left, const float right) {
	return std::bit_cast < uint32_t >(left) == std::bit_cast < uint32_t >(right);
}

int fail(const char * message) {
	std::cerr << message << '\n';
	return 1;
}

int main() {
	alignas(32) float zero[hsv::width];
	alignas(32) float one[hsv::width];
	alignas(32) float output[hsv::width];
	for (int32_t lane = 0 ; lane < hsv::width ; ++lane) {
		zero[lane] = 100.0f + lane;
		one[lane] = 200.0f + lane;
	}
	const hsv::float8 zero8 = hsv::load_aligned(zero);
	const hsv::float8 one8 = hsv::load_aligned(one);
	for (uint32_t packed = 0 ; packed < 256 ; ++packed) {
		hsv::store_aligned(output, hsv::select_packed_byte(static_cast < uint8_t >(packed), zero8, one8));
		for (int32_t lane = 0 ; lane < hsv::width ; ++lane) {
			const float expected = packed & (0x80u >> lane) ? one[lane] : zero[lane];
			if (!same_bits(output[lane], expected)) return fail("Packed-byte lane order differs from AVX");
		}
	}

	const hsv::float8 two = hsv::broadcast(2.0f);
	const hsv::float8 three = hsv::broadcast(3.0f);
	hsv::store_aligned(output, hsv::add(two, three));
	for (const float value : output)
		if (!same_bits(value, 5.0f)) return fail("Vector addition failed");
	hsv::store_aligned(output, hsv::multiply(two, three));
	for (const float value : output)
		if (!same_bits(value, 6.0f)) return fail("Vector multiplication failed");

	const float fma_left = std::bit_cast < float >(0x3f800001u);
	const float fma_right = fma_left;
	const float fma_addend = -std::bit_cast < float >(0x3f800002u);
	const float fma_expected = std::fma(fma_left, fma_right, fma_addend);
	const hsv::float8 fma_result = hsv::multiply_add(
		hsv::broadcast(fma_left), hsv::broadcast(fma_right), hsv::broadcast(fma_addend));
	hsv::store_aligned(output, fma_result);
	for (const float value : output)
		if (!same_bits(value, fma_expected)) return fail("Vector multiply-add is not fused");

	alignas(32) const float reduction_lanes[hsv::width] = {
		1.0e20f, 1.0f, -1.0e20f, 3.0f,
		-1.0e20f, 2.0f, 1.0e20f, 4.0f
	};
	const float cross0 = reduction_lanes[0] + reduction_lanes[4];
	const float cross1 = reduction_lanes[1] + reduction_lanes[5];
	const float cross2 = reduction_lanes[2] + reduction_lanes[6];
	const float cross3 = reduction_lanes[3] + reduction_lanes[7];
	const float reduction_expected = (cross0 + cross1) + (cross2 + cross3);
	const float reduction_result = hsv::horizontal_add(hsv::load_aligned(reduction_lanes));
	if (!same_bits(reduction_result, reduction_expected))
		return fail("Horizontal reduction order differs from AVX");

	return 0;
}
