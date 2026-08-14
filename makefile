projects = phase_common phase_rare switch ligate simulate xcftools

.PHONY: all $(projects) benchmark benchmark-threads benchmark-edge benchmark-unit rng-test

all: $(projects)

$(projects):
	$(MAKE) -C $@

clean:
	for dir in $(projects); do \
	$(MAKE) $@ -C $$dir; \
	done
	cargo clean --manifest-path rng/Cargo.toml
	rm -f static_bins/*
	rm -f docker/resources/*
	rm -f docker/shapeit5*.tar.gz

static_exe:
	for dir in $(projects); do \
	$(MAKE) $@ -C $$dir; \
	done

rgc:
	for dir in $(projects); do \
	$(MAKE) $@ -C $$dir; \
	done

benchmark:
	python3 test/benchmarks/run.py --bin-dir .

benchmark-threads:
	python3 test/benchmarks/thread_determinism.py --bin-dir .

benchmark-edge:
	python3 test/benchmarks/edge_regressions.py --bin-dir .

benchmark-unit: rng-test
	python3 -m unittest discover -s test/benchmarks/tests -v

rng-test:
	cargo test --manifest-path rng/Cargo.toml
	$(MAKE) -C rng/tests test
