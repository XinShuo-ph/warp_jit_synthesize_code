
# Makefile for atom_sxoaqm

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
atom_sxoaqm: atom_sxoaqm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
atom_sxoaqm.ptx: atom_sxoaqm.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f atom_sxoaqm atom_sxoaqm.ptx

.PHONY: clean
