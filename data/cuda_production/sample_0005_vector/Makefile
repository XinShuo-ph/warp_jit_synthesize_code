
# Makefile for vec_kgoohi

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
vec_kgoohi: vec_kgoohi.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
vec_kgoohi.ptx: vec_kgoohi.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f vec_kgoohi vec_kgoohi.ptx

.PHONY: clean
