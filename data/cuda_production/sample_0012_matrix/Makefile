
# Makefile for mat_byhkii

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
mat_byhkii: mat_byhkii.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
mat_byhkii.ptx: mat_byhkii.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f mat_byhkii mat_byhkii.ptx

.PHONY: clean
