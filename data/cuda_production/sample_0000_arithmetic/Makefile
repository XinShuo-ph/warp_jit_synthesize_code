
# Makefile for arith_ifwkrq

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
arith_ifwkrq: arith_ifwkrq.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
arith_ifwkrq.ptx: arith_ifwkrq.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f arith_ifwkrq arith_ifwkrq.ptx

.PHONY: clean
