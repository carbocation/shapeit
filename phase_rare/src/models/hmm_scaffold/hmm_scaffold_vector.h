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

#ifndef SHAPEIT_HMM_SCAFFOLD_VECTOR_H
#define SHAPEIT_HMM_SCAFFOLD_VECTOR_H

#include <cstdint>

#if !defined(SHAPEIT_HMM_SCAFFOLD_FORCE_SCALAR) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif !defined(SHAPEIT_HMM_SCAFFOLD_FORCE_SCALAR) && defined(__aarch64__)
#include <arm_neon.h>
#else
#include <array>
#include <cmath>
#endif

namespace hmm_scaffold_vector {

constexpr int32_t width = 8;

#if !defined(SHAPEIT_HMM_SCAFFOLD_FORCE_SCALAR) && defined(__AVX2__) && defined(__FMA__)

using float8 = __m256;

inline float8 broadcast(const float value) {
	return _mm256_set1_ps(value);
}

inline float8 load_aligned(const float * source) {
	return _mm256_load_ps(source);
}

inline void store_aligned(float * target, const float8 value) {
	_mm256_store_ps(target, value);
}

inline float8 add(const float8 left, const float8 right) {
	return _mm256_add_ps(left, right);
}

inline float8 multiply(const float8 left, const float8 right) {
	return _mm256_mul_ps(left, right);
}

inline float8 multiply_add(const float8 left, const float8 right, const float8 addend) {
	return _mm256_fmadd_ps(left, right, addend);
}

inline float8 select_packed_byte(const uint8_t packed, const float8 when_zero, const float8 when_one) {
	const __m256i shift_count = _mm256_set_epi32(31, 30, 29, 28, 27, 26, 25, 24);
	const __m256i mask = _mm256_sllv_epi32(_mm256_set1_epi32(static_cast<uint32_t>(packed)), shift_count);
	return _mm256_blendv_ps(when_zero, when_one, _mm256_castsi256_ps(mask));
}

inline float horizontal_add(const float8 value) {
	__m128 low = _mm256_castps256_ps128(value);
	const __m128 high = _mm256_extractf128_ps(value, 1);
	low = _mm_add_ps(low, high);
	__m128 shuffle = _mm_movehdup_ps(low);
	__m128 sums = _mm_add_ps(low, shuffle);
	shuffle = _mm_movehl_ps(shuffle, sums);
	sums = _mm_add_ss(sums, shuffle);
	return _mm_cvtss_f32(sums);
}

#elif !defined(SHAPEIT_HMM_SCAFFOLD_FORCE_SCALAR) && defined(__aarch64__)

struct float8 {
	float32x4_t low;
	float32x4_t high;
};

inline float8 broadcast(const float value) {
	const float32x4_t lanes = vdupq_n_f32(value);
	return {lanes, lanes};
}

inline float8 load_aligned(const float * source) {
	return {vld1q_f32(source), vld1q_f32(source + 4)};
}

inline void store_aligned(float * target, const float8 value) {
	vst1q_f32(target, value.low);
	vst1q_f32(target + 4, value.high);
}

inline float8 add(const float8 left, const float8 right) {
	return {vaddq_f32(left.low, right.low), vaddq_f32(left.high, right.high)};
}

inline float8 multiply(const float8 left, const float8 right) {
	return {vmulq_f32(left.low, right.low), vmulq_f32(left.high, right.high)};
}

inline float8 multiply_add(const float8 left, const float8 right, const float8 addend) {
	return {
		vfmaq_f32(addend.low, left.low, right.low),
		vfmaq_f32(addend.high, left.high, right.high)
	};
}

inline float8 select_packed_byte(const uint8_t packed, const float8 when_zero, const float8 when_one) {
	const uint32x4_t packed_lanes = vdupq_n_u32(static_cast<uint32_t>(packed));
	const uint32x4_t low_bits = {0x80u, 0x40u, 0x20u, 0x10u};
	const uint32x4_t high_bits = {0x08u, 0x04u, 0x02u, 0x01u};
	return {
		vbslq_f32(vtstq_u32(packed_lanes, low_bits), when_one.low, when_zero.low),
		vbslq_f32(vtstq_u32(packed_lanes, high_bits), when_one.high, when_zero.high)
	};
}

inline float horizontal_add(const float8 value) {
	// Match the AVX reduction exactly: combine corresponding 128-bit lanes,
	// then add lanes 0+1 and 2+3 before combining those two partial sums.
	const float32x4_t cross_halves = vaddq_f32(value.low, value.high);
	const float32x2_t pairs = vpadd_f32(vget_low_f32(cross_halves), vget_high_f32(cross_halves));
	return vget_lane_f32(vpadd_f32(pairs, pairs), 0);
}

#else

struct float8 {
	std::array < float, width > lanes;
};

inline float8 broadcast(const float value) {
	float8 output;
	output.lanes.fill(value);
	return output;
}

inline float8 load_aligned(const float * source) {
	float8 output;
	for (int32_t lane = 0 ; lane < width ; ++lane) output.lanes[lane] = source[lane];
	return output;
}

inline void store_aligned(float * target, const float8 value) {
	for (int32_t lane = 0 ; lane < width ; ++lane) target[lane] = value.lanes[lane];
}

inline float8 add(const float8 left, const float8 right) {
	float8 output;
	for (int32_t lane = 0 ; lane < width ; ++lane) output.lanes[lane] = left.lanes[lane] + right.lanes[lane];
	return output;
}

inline float8 multiply(const float8 left, const float8 right) {
	float8 output;
	for (int32_t lane = 0 ; lane < width ; ++lane) output.lanes[lane] = left.lanes[lane] * right.lanes[lane];
	return output;
}

inline float8 multiply_add(const float8 left, const float8 right, const float8 addend) {
	float8 output;
	for (int32_t lane = 0 ; lane < width ; ++lane)
		output.lanes[lane] = std::fma(left.lanes[lane], right.lanes[lane], addend.lanes[lane]);
	return output;
}

inline float8 select_packed_byte(const uint8_t packed, const float8 when_zero, const float8 when_one) {
	float8 output;
	for (int32_t lane = 0 ; lane < width ; ++lane)
		output.lanes[lane] = packed & (0x80u >> lane) ? when_one.lanes[lane] : when_zero.lanes[lane];
	return output;
}

inline float horizontal_add(const float8 value) {
	const float cross0 = value.lanes[0] + value.lanes[4];
	const float cross1 = value.lanes[1] + value.lanes[5];
	const float cross2 = value.lanes[2] + value.lanes[6];
	const float cross3 = value.lanes[3] + value.lanes[7];
	return (cross0 + cross1) + (cross2 + cross3);
}

#endif

}

#endif
