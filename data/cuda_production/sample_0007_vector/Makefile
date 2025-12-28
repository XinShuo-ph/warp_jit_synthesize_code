
# Makefile for vec_gyqaks

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
vec_gyqaks: vec_gyqaks.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
vec_gyqaks.ptx: vec_gyqaks.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f vec_gyqaks vec_gyqaks.ptx

.PHONY: clean
