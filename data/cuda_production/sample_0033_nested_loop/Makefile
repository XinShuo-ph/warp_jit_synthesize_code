
# Makefile for nested_kfwusk

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
nested_kfwusk: nested_kfwusk.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
nested_kfwusk.ptx: nested_kfwusk.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f nested_kfwusk nested_kfwusk.ptx

.PHONY: clean
