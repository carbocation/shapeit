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

#include <containers/ibd2_tracks.h>

#include <cstdint>
#include <stdexcept>

using namespace std;

ibd2_tracks::ibd2_tracks () {
	Handle = nullptr;
}

ibd2_tracks::~ibd2_tracks() {
	clear();
}

void ibd2_tracks::clear() {
	if (Handle != nullptr) {
		shapeit_ibd2_tracks_free_v1(Handle);
		Handle = nullptr;
	}
}

void ibd2_tracks::initialize(int n_ind, variant_map & V) {
	clear();
	vector < float > centimorgans(V.size(), 0.0f);
	for (int l = 0 ; l < centimorgans.size() ; l ++) centimorgans[l] = V.vec_pos[l]->cm;
	const uint32_t status = shapeit_ibd2_tracks_create_v1(
		n_ind, centimorgans.data(), centimorgans.size(), &Handle);
	if (status != SHAPEIT_IBD2_STATUS_OK) {
		throw runtime_error("Rust IBD2 registry rejected initialization (status " +
			to_string(status) + ")");
	}
}
