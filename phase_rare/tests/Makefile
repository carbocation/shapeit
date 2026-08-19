CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic

TEST_ARCH := $(shell uname -m)
NATIVE_SIMD_FLAGS =
ifeq ($(TEST_ARCH),x86_64)
NATIVE_SIMD_FLAGS = -mavx2 -mfma
endif

VECTOR_HEADER = ../src/models/hmm_scaffold/hmm_scaffold_vector.h
OBJ_DIR = ../obj/tests-$(TEST_ARCH)
NATIVE_TEST_BIN = $(OBJ_DIR)/test_hmm_scaffold_vector
SCALAR_TEST_BIN = $(OBJ_DIR)/test_hmm_scaffold_vector_scalar
TEST_BINS = $(NATIVE_TEST_BIN) $(SCALAR_TEST_BIN)

.PHONY: test clean

test: $(TEST_BINS)
	$(NATIVE_TEST_BIN)
	$(SCALAR_TEST_BIN)

$(NATIVE_TEST_BIN): test_hmm_scaffold_vector.cpp $(VECTOR_HEADER) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_SIMD_FLAGS) -I../src $< -o $@

$(SCALAR_TEST_BIN): test_hmm_scaffold_vector.cpp $(VECTOR_HEADER) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DSHAPEIT_HMM_SCAFFOLD_FORCE_SCALAR -I../src $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -f $(TEST_BINS)
