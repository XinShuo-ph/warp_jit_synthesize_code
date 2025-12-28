
# Makefile for combined_iflttc

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
combined_iflttc: combined_iflttc.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
combined_iflttc.ptx: combined_iflttc.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f combined_iflttc combined_iflttc.ptx

.PHONY: clean
