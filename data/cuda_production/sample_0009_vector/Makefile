
# Makefile for vec_syjfja

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
vec_syjfja: vec_syjfja.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
vec_syjfja.ptx: vec_syjfja.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f vec_syjfja vec_syjfja.ptx

.PHONY: clean
