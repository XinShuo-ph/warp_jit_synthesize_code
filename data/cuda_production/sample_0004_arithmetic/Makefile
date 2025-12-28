
# Makefile for arith_trrlkf

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
arith_trrlkf: arith_trrlkf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
arith_trrlkf.ptx: arith_trrlkf.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f arith_trrlkf arith_trrlkf.ptx

.PHONY: clean
