
# Makefile for nested_aigoig

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
nested_aigoig: nested_aigoig.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
nested_aigoig.ptx: nested_aigoig.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f nested_aigoig nested_aigoig.ptx

.PHONY: clean
