
# Makefile for scalar_tlxqzy

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
scalar_tlxqzy: scalar_tlxqzy.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
scalar_tlxqzy.ptx: scalar_tlxqzy.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f scalar_tlxqzy scalar_tlxqzy.ptx

.PHONY: clean
