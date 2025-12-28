
# Makefile for multicond_tdzjrm

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
multicond_tdzjrm: multicond_tdzjrm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
multicond_tdzjrm.ptx: multicond_tdzjrm.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f multicond_tdzjrm multicond_tdzjrm.ptx

.PHONY: clean
