
# Makefile for math_hnfauv

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
math_hnfauv: math_hnfauv.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
math_hnfauv.ptx: math_hnfauv.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f math_hnfauv math_hnfauv.ptx

.PHONY: clean
