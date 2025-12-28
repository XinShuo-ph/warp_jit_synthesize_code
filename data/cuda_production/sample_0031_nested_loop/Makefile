
# Makefile for nested_zptoks

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
nested_zptoks: nested_zptoks.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
nested_zptoks.ptx: nested_zptoks.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f nested_zptoks nested_zptoks.ptx

.PHONY: clean
