
# Makefile for vec_okuuez

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
vec_okuuez: vec_okuuez.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
vec_okuuez.ptx: vec_okuuez.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f vec_okuuez vec_okuuez.ptx

.PHONY: clean
