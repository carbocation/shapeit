CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20

RNG_LIB = ../target/release/libshapeit_rng.a
RNG_TEST_BIN = ../target/test_shapeit_rng
BITMATRIX_TEST_BIN = ../target/test_shapeit_bitmatrix
TEST_BINS = $(RNG_TEST_BIN) $(BITMATRIX_TEST_BIN)

.PHONY: test clean

test: $(TEST_BINS)
	$(RNG_TEST_BIN)
	$(BITMATRIX_TEST_BIN)

$(RNG_LIB): ../Cargo.toml ../Cargo.lock $(shell find ../src -name '*.rs')
	cargo build --release --manifest-path ../Cargo.toml

$(RNG_TEST_BIN): test_rng.cpp ../include/shapeit_rng.h ../../xcftools/common/src/utils/random_number.h $(RNG_LIB)
	$(CXX) $(CXXFLAGS) -I../include -I../../xcftools/common/src $< $(RNG_LIB) -o $@

$(BITMATRIX_TEST_BIN): test_bitmatrix.cpp ../include/shapeit_bitmatrix.h $(RNG_LIB)
	$(CXX) $(CXXFLAGS) -I../include $< $(RNG_LIB) -o $@

clean:
	rm -f $(TEST_BINS)
