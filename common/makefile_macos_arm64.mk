# Native Apple Silicon build.  This is intentionally isolated from the
# established Linux/x86 build so that its flags and artifact layout remain
# unchanged.

MACOS_ARM64_TAG ?= macos-arm64
MACOS_ARM64_OBJDIR ?= obj/$(MACOS_ARM64_TAG)
MACOS_ARM64_BINDIR ?= bin/$(MACOS_ARM64_TAG)
MACOS_ARM64_BFILE = $(MACOS_ARM64_BINDIR)/$(NAME)

MACOS_ARM64_CXX ?= clang++
MACOS_ARM64_CXXFLAGS ?= -O3 -std=c++20 -arch arm64
MACOS_ARM64_LDFLAGS ?= -O3 -arch arm64

MACOS_ARM64_RUST_TARGET ?= aarch64-apple-darwin
MACOS_ARM64_RNG_LIB = $(RNG_DIR)/target/$(MACOS_ARM64_RUST_TARGET)/release/libshapeit_rng.a
MACOS_ARM64_CARGO ?= cargo

MACOS_ARM64_BREW ?= $(shell command -v brew 2>/dev/null)
MACOS_ARM64_HTSLIB_PREFIX ?= $(shell $(MACOS_ARM64_BREW) --prefix htslib 2>/dev/null)
MACOS_ARM64_BOOST_PREFIX ?= $(shell $(MACOS_ARM64_BREW) --prefix boost 2>/dev/null)

MACOS_ARM64_HTSLIB_INC ?= $(MACOS_ARM64_HTSLIB_PREFIX)/include
MACOS_ARM64_HTSLIB_LIB ?= $(MACOS_ARM64_HTSLIB_PREFIX)/lib/libhts.dylib
MACOS_ARM64_BOOST_INC ?= $(MACOS_ARM64_BOOST_PREFIX)/include
MACOS_ARM64_BOOST_LIB_IO ?= $(MACOS_ARM64_BOOST_PREFIX)/lib/libboost_iostreams.dylib
MACOS_ARM64_BOOST_LIB_PO ?= $(MACOS_ARM64_BOOST_PREFIX)/lib/libboost_program_options.dylib
MACOS_ARM64_EXTRA_LIBS ?=

MACOS_ARM64_OFILE = $(shell for file in `find src -name '*.cpp'`; do echo $(MACOS_ARM64_OBJDIR)/$$(basename $$file .cpp).o; done)
MACOS_ARM64_VERSION_HEADER = $(MACOS_ARM64_OBJDIR)/build_version.h
MACOS_ARM64_RNG_SOURCE = $(RNG_SOURCE) $(wildcard $(RNG_DIR)/.cargo/config.toml)

MACOS_ARM64_COMMIT_VERS := $(strip $(COMMIT_VERS))
MACOS_ARM64_COMMIT_DATE := $(strip $(COMMIT_DATE))
ifeq ($(MACOS_ARM64_COMMIT_VERS),)
MACOS_ARM64_COMMIT_VERS := unknown
endif
ifeq ($(MACOS_ARM64_COMMIT_DATE),)
MACOS_ARM64_COMMIT_DATE := unknown
endif

.PHONY: macos-arm64 macos-arm64-check macos-arm64-version-force clean-macos-arm64

macos-arm64: macos-arm64-check $(MACOS_ARM64_BFILE)

macos-arm64-check:
	@if [ "$(shell uname -s)" != "Darwin" ] || [ "$(shell uname -m)" != "arm64" ]; then \
		echo "macos-arm64 requires a native arm64 macOS shell" >&2; \
		exit 1; \
	fi
	@for file in \
		"$(MACOS_ARM64_HTSLIB_INC)/htslib/hts.h" \
		"$(MACOS_ARM64_HTSLIB_LIB)" \
		"$(MACOS_ARM64_BOOST_INC)/boost/program_options.hpp" \
		"$(MACOS_ARM64_BOOST_LIB_IO)" \
		"$(MACOS_ARM64_BOOST_LIB_PO)"; do \
		if [ ! -e "$$file" ]; then \
			echo "Missing macOS arm64 dependency: $$file" >&2; \
			exit 1; \
		fi; \
	done
	@if [ -n "$(strip $(MACOS_ARM64_EXTRA_LIBS))" ]; then \
		for file in $(MACOS_ARM64_EXTRA_LIBS); do \
			if [ ! -e "$$file" ]; then \
				echo "Missing macOS arm64 dependency: $$file" >&2; \
				exit 1; \
			fi; \
		done; \
	fi

$(MACOS_ARM64_OBJDIR) $(MACOS_ARM64_BINDIR):
	mkdir -p $@

macos-arm64-version-force:

$(MACOS_ARM64_VERSION_HEADER): macos-arm64-version-force | $(MACOS_ARM64_OBJDIR)
	@printf '%s\n' \
		'#ifndef SHAPEIT_BUILD_VERSION_H' \
		'#define SHAPEIT_BUILD_VERSION_H' \
		'#ifndef __COMMIT_ID__' \
		'#define __COMMIT_ID__ "$(MACOS_ARM64_COMMIT_VERS)"' \
		'#endif' \
		'#ifndef __COMMIT_DATE__' \
		'#define __COMMIT_DATE__ "$(MACOS_ARM64_COMMIT_DATE)"' \
		'#endif' \
		'#endif' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm -f $@.tmp; fi

$(MACOS_ARM64_BFILE): $(MACOS_ARM64_OFILE) $(MACOS_ARM64_RNG_LIB) | $(MACOS_ARM64_BINDIR)
	$(MACOS_ARM64_CXX) $(MACOS_ARM64_LDFLAGS) $^ -o $@ \
		$(MACOS_ARM64_HTSLIB_LIB) $(MACOS_ARM64_BOOST_LIB_IO) $(MACOS_ARM64_BOOST_LIB_PO) \
		$(MACOS_ARM64_EXTRA_LIBS)

$(MACOS_ARM64_RNG_LIB): $(MACOS_ARM64_RNG_SOURCE)
	cd $(RNG_DIR) && $(MACOS_ARM64_CARGO) build --release --target $(MACOS_ARM64_RUST_TARGET)

$(MACOS_ARM64_OBJDIR)/%.o: %.cpp $(HFILE) $(RNG_HEADERS) $(MACOS_ARM64_VERSION_HEADER) | $(MACOS_ARM64_OBJDIR)
	$(MACOS_ARM64_CXX) $(MACOS_ARM64_CXXFLAGS) -include $(MACOS_ARM64_VERSION_HEADER) \
		-c $< -o $@ -Isrc -I$(RNG_DIR)/include \
		-I$(MACOS_ARM64_HTSLIB_INC) -I$(MACOS_ARM64_BOOST_INC)

clean-macos-arm64:
	rm -f $(MACOS_ARM64_OBJDIR)/*.o $(MACOS_ARM64_VERSION_HEADER) \
		$(MACOS_ARM64_VERSION_HEADER).tmp $(MACOS_ARM64_BFILE)
