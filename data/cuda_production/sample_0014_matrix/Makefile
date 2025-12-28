
# Makefile for mat_apjbjy

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
mat_apjbjy: mat_apjbjy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
mat_apjbjy.ptx: mat_apjbjy.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f mat_apjbjy mat_apjbjy.ptx

.PHONY: clean
