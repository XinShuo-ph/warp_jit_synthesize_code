
# Makefile for atom_jrimtx

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
atom_jrimtx: atom_jrimtx.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
atom_jrimtx.ptx: atom_jrimtx.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f atom_jrimtx atom_jrimtx.ptx

.PHONY: clean
