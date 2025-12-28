
# Makefile for scalar_tqjcbz

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
scalar_tqjcbz: scalar_tqjcbz.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
scalar_tqjcbz.ptx: scalar_tqjcbz.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f scalar_tqjcbz scalar_tqjcbz.ptx

.PHONY: clean
