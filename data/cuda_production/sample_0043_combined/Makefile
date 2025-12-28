
# Makefile for combined_ldrywc

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
combined_ldrywc: combined_ldrywc.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
combined_ldrywc.ptx: combined_ldrywc.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f combined_ldrywc combined_ldrywc.ptx

.PHONY: clean
