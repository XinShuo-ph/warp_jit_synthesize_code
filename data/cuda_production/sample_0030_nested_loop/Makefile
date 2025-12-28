
# Makefile for nested_hqlsqc

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
nested_hqlsqc: nested_hqlsqc.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
nested_hqlsqc.ptx: nested_hqlsqc.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f nested_hqlsqc nested_hqlsqc.ptx

.PHONY: clean
