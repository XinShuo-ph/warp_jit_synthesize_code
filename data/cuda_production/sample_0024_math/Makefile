
# Makefile for math_wgnhxn

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
math_wgnhxn: math_wgnhxn.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
math_wgnhxn.ptx: math_wgnhxn.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f math_wgnhxn math_wgnhxn.ptx

.PHONY: clean
