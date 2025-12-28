
# Makefile for scalar_dttsrd

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
scalar_dttsrd: scalar_dttsrd.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
scalar_dttsrd.ptx: scalar_dttsrd.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f scalar_dttsrd scalar_dttsrd.ptx

.PHONY: clean
