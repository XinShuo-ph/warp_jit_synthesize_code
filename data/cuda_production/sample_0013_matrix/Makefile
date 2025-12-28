
# Makefile for mat_ufrugc

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
mat_ufrugc: mat_ufrugc.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
mat_ufrugc.ptx: mat_ufrugc.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f mat_ufrugc mat_ufrugc.ptx

.PHONY: clean
