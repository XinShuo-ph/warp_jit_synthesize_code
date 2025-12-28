
# Makefile for mat_akfmrx

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
mat_akfmrx: mat_akfmrx.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
mat_akfmrx.ptx: mat_akfmrx.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f mat_akfmrx mat_akfmrx.ptx

.PHONY: clean
