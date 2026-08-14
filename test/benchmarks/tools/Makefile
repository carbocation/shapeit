CXX ?= c++
HTSLIB_PREFIX ?= /usr/local
OUTPUT ?= subset_bcf
CPPFLAGS += -I$(HTSLIB_PREFIX)/include
CXXFLAGS += -O2 -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS += -L$(HTSLIB_PREFIX)/lib
LDLIBS += -lhts

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): subset_bcf.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(OUTPUT)
