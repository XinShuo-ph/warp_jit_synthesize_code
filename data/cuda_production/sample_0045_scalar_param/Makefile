
# Makefile for scalar_nehqkc

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
scalar_nehqkc: scalar_nehqkc.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
scalar_nehqkc.ptx: scalar_nehqkc.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f scalar_nehqkc scalar_nehqkc.ptx

.PHONY: clean
