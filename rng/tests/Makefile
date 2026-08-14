CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20

RNG_LIB = ../target/release/libshapeit_rng.a
TEST_BIN = ../target/test_shapeit_rng

.PHONY: test clean

test: $(TEST_BIN)
	$(TEST_BIN)

$(RNG_LIB): ../Cargo.toml ../Cargo.lock $(shell find ../src -name '*.rs')
	cargo build --release --manifest-path ../Cargo.toml

$(TEST_BIN): test_rng.cpp ../include/shapeit_rng.h ../../xcftools/common/src/utils/random_number.h $(RNG_LIB)
	$(CXX) $(CXXFLAGS) -I../include -I../../xcftools/common/src $< $(RNG_LIB) -o $@

clean:
	rm -f $(TEST_BIN)
