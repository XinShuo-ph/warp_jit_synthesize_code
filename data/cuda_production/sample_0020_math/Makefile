
# Makefile for math_xidpcr

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
math_xidpcr: math_xidpcr.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
math_xidpcr.ptx: math_xidpcr.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f math_xidpcr math_xidpcr.ptx

.PHONY: clean
