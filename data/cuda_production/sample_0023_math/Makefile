
# Makefile for math_ciiams

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
math_ciiams: math_ciiams.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
math_ciiams.ptx: math_ciiams.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f math_ciiams math_ciiams.ptx

.PHONY: clean
