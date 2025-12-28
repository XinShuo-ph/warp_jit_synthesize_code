
# Makefile for combined_nrvzok

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
combined_nrvzok: combined_nrvzok.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
combined_nrvzok.ptx: combined_nrvzok.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f combined_nrvzok combined_nrvzok.ptx

.PHONY: clean
