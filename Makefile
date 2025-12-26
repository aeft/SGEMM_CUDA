.PHONY: all build debug clean profile bench cuobjdump test practice compare

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)

FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

bench: build
	@bash gen_benchmark_results.sh

# Usage: make test KERNEL=1
# Test original implementation
test: build
	@$(BUILD_DIR)/sgemm $(KERNEL)

# Usage: make practice KERNEL=1
# Test practice implementation
practice: build
	@$(BUILD_DIR)/sgemm_practice $(KERNEL)

# Usage: make compare KERNEL=1
# Compare original vs practice
# Or: make compare KERNEL="1 2 3" to compare multiple kernels
# Or: make compare KERNEL=1 PRACTICE_ONLY=1 to run only practice version
compare: build
	@bash compare_kernels.sh $(KERNEL) $(if $(PRACTICE_ONLY),--practice,)
