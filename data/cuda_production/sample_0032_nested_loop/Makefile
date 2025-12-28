
# Makefile for nested_bxkbgg

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
nested_bxkbgg: nested_bxkbgg.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
nested_bxkbgg.ptx: nested_bxkbgg.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f nested_bxkbgg nested_bxkbgg.ptx

.PHONY: clean
