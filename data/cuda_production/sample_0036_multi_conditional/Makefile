
# Makefile for multicond_lxkjwf

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
multicond_lxkjwf: multicond_lxkjwf.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
multicond_lxkjwf.ptx: multicond_lxkjwf.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f multicond_lxkjwf multicond_lxkjwf.ptx

.PHONY: clean
