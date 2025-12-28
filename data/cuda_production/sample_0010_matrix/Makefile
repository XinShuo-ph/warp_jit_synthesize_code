
# Makefile for mat_dwetwf

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
mat_dwetwf: mat_dwetwf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
mat_dwetwf.ptx: mat_dwetwf.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f mat_dwetwf mat_dwetwf.ptx

.PHONY: clean
