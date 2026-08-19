---
layout: default
title: Compile SHAPEIT5
parent: Build from source
grand_parent: Installation
permalink: /docs/installation/build_from_source/compile_shapeit5
---
# Compile SHAPEIT5
{: .no_toc .text-center }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Download the source

Download the latest SHAPEIT5 source using:

<div class="code-example" markdown="1">
```bash
git clone --recurse-submodules https://github.com/odelaneau/shapeit.git
cd shapeit
```
</div>

Adding `--recurse-submodules` also initializes any required submodules.

The checkout contains the following software packages and utility folders:

- **docker**: all scripts needed to build a docker file comprising all binaries
- **docs**: documentation in html
- **ligate**: ligate multiple phased BCF/VCF files into a chromosome length file
- **phase_common**: phase common sites, typically SNP array data
- **phase_rare**: phase rare variants onto a scaffold of common variants
- **rng**: versioned Rust phasing kernels, built automatically
- **resources**: genetics maps [b37/b38] and coordinates for [5/20] cM chunks
- **simulate**: simulate simple family and haploid datasets
- **static_bins**: static binaries of all executables
- **switch**: compute switch error rate and genotyping error rate given simulated or trio data
- **tasks**: scripts used to phase large datasets, good base to start pipelining
- **test**: simulated data for first-step testing of the method
- **versions**: versioning
- **xcftools**: tool to handle XCF file format

## Native Apple Silicon build

These instructions build all six executables as native arm64 Mach-O programs.
They do not use Rosetta. Install the command-line developer tools, Homebrew
dependencies, and a Rust toolchain first:

<div class="code-example" markdown="1">
```bash
xcode-select --install
brew install boost htslib rust
```
</div>

If Rust was installed through `rustup` instead, it only needs to be version
1.85 or newer. From the repository root, build the complete suite with:

<div class="code-example" markdown="1">
```bash
make -j"$(sysctl -n hw.logicalcpu)" macos-arm64
```
</div>

The `macos-arm64` target requires a Darwin arm64 host and discovers the native
Homebrew prefixes automatically. Its outputs are kept separate from the
established Linux build artifacts. Executables are written to:

- `phase_common/bin/macos-arm64/phase_common`
- `phase_rare/bin/macos-arm64/phase_rare`
- `switch/bin/macos-arm64/switch`
- `ligate/bin/macos-arm64/ligate`
- `simulate/bin/macos-arm64/simulate`
- `xcftools/bin/macos-arm64/xcftools`

Confirm that the resulting files are native before running a large job:

<div class="code-example" markdown="1">
```bash
file phase_common/bin/macos-arm64/phase_common \
  phase_rare/bin/macos-arm64/phase_rare
```
</div>

Both lines should identify an `arm64` Mach-O executable.

The reported build provenance should also match the checkout:

<div class="code-example" markdown="1">
```bash
phase_common/bin/macos-arm64/phase_common --help 2>&1 \
  | grep "commit = $(git rev-parse --short HEAD)"
```
</div>

### Validate the build

The repository includes Rust unit tests, C++/Rust ABI tests, deterministic
phasing regressions, and thread-scheduling checks. Run the complete local suite
with:

<div class="code-example" markdown="1">
```bash
make rng-test
make -C phase_rare vector-test
make benchmark-unit

native_bin="$(mktemp -d)"
for tool in phase_common phase_rare switch ligate simulate xcftools; do
  ln -s "$PWD/$tool/bin/macos-arm64/$tool" "$native_bin/$tool"
done

python3 test/benchmarks/run.py --bin-dir "$native_bin"
python3 test/benchmarks/thread_determinism.py --bin-dir "$native_bin"
python3 test/benchmarks/edge_regressions.py --bin-dir "$native_bin"
```
</div>

The benchmark fixtures are generated under `/tmp` and require `htsfile`, which
is included in the Homebrew HTSlib package. The temporary directory above only
collects symlinks so the cross-platform benchmark runner can find all six
per-project executables in one place.

### Dynamic library dependency

The native executables link dynamically to the Homebrew installations of
HTSlib, Boost, and their dependencies. Keep those packages installed and do
not copy the executables to another Mac as standalone files. To inspect one
executable's library paths, run:

<div class="code-example" markdown="1">
```bash
otool -L phase_common/bin/macos-arm64/phase_common
```
</div>

macOS does not support the Linux-style fully static `static_exe` target. A
redistributable macOS package would need to bundle or otherwise provide its
non-system dynamic libraries.

## Linux and custom library prefixes

Each software in the suite contains the same folder structure:

- `bin`: folder for the compiled binary.
- `obj`: folder with all binary objects.
- `src`: folder with source code.
- `makefile`: Makefile to compile the program.

On Linux, `make` at the repository root builds all tools. To compile only one
tool, for example `phase_rare`, run `make -C phase_rare`. If dependencies are
installed under a custom prefix, provide the relevant variables on the command
line:

- `HTSSRC`: path to the root of the HTSlib library, the prefix for HTSLIB_INC and HTSLIB_LIB paths.
- `HTSLIB_INC`: path to the HTSlib header files
- `HTSLIB_LIB`: path to the HTSlib library
- `BOOST_INC`: path to the Boost header files
- `BOOST_LIB_IO`: path to the Boost iostreams library
- `BOOST_LIB_PO`: path to the Boost program_options library

On Linux, system static libraries can be located with:

<div class="code-example" markdown="1">
```bash
locate libboost_program_options.a libboost_iostreams.a libhts.a
```
</div>

Cargo builds the Rust phasing-kernel library automatically. Build products are
placed in each tool's `bin/` folder. Use `make clean-macos-arm64` to remove the
native macOS objects and executables without touching Linux artifacts; use
`make clean` for the established build paths.
