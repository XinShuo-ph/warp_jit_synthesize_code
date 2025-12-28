
# Makefile for multicond_gyspox

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
multicond_gyspox: multicond_gyspox.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
multicond_gyspox.ptx: multicond_gyspox.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f multicond_gyspox multicond_gyspox.ptx

.PHONY: clean
