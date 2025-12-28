
# Makefile for vec_sipwqd

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
vec_sipwqd: vec_sipwqd.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
vec_sipwqd.ptx: vec_sipwqd.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f vec_sipwqd vec_sipwqd.ptx

.PHONY: clean
