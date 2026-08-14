#!/usr/bin/env bash
set -euo pipefail

benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$benchmark_dir/../.." && pwd)
tool_dir="$benchmark_dir/tools"
temp_root=${SHAPEIT5_BENCHMARK_TMP:-/tmp/shapeit5-benchmarks-$(id -u)}
source_dir="$temp_root/source"
fixture_dir="$temp_root/fixtures"
binary_dir="$temp_root/bin"
public_base="https://raw.githubusercontent.com/carbocation/shapeit/c34d4db3e99a2f7e23deb727671ae260901a5886"

mkdir -p "$source_dir" "$fixture_dir" "$binary_dir"

if [[ -z "${HTSLIB_PREFIX:-}" ]]; then
	if command -v brew >/dev/null 2>&1; then
		HTSLIB_PREFIX=$(brew --prefix htslib)
	else
		HTSLIB_PREFIX=/usr/local
	fi
fi

subset="$binary_dir/subset_bcf"
make -C "$tool_dir" HTSLIB_PREFIX="$HTSLIB_PREFIX" OUTPUT="$subset"

resolve_bcf() {
	local relative_path=$1
	local local_path="$repository/$relative_path"
	local cached_path="$source_dir/${relative_path//\//_}"
	if [[ -f "$local_path" && -f "$local_path.csi" ]]; then
		printf '%s\n' "$local_path"
		return
	fi
	if [[ ! -f "$cached_path" ]]; then
		curl --fail --location --silent --show-error \
			"$public_base/$relative_path" --output "$cached_path"
	fi
	if [[ ! -f "$cached_path.csi" ]]; then
		curl --fail --location --silent --show-error \
			"$public_base/$relative_path.csi" --output "$cached_path.csi"
	fi
	printf '%s\n' "$cached_path"
}

array_source=$(resolve_bcf test/array/target.unrelated.bcf)
wgs_source=$(resolve_bcf test/wgs/target.unrelated.bcf)

"$subset" "$array_source" \
	"$fixture_dir/common.truth.bcf" 1 128 1200 0.01
"$subset" "$fixture_dir/common.truth.bcf" \
	"$fixture_dir/common.scaffold.bcf" 1 128 300 0.01
"$subset" "$wgs_source" \
	"$fixture_dir/rare.truth.bcf" 1:1-500000 128 1500 0.0001
"$subset" "$fixture_dir/rare.truth.bcf" \
	"$fixture_dir/rare.scaffold.bcf" 1:1-500000 128 120 0.05

printf 'Generated benchmark fixtures in %s\n' "$fixture_dir"
