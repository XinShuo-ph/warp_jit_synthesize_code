
# Makefile for atom_bommgp

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
atom_bommgp: atom_bommgp.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
atom_bommgp.ptx: atom_bommgp.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f atom_bommgp atom_bommgp.ptx

.PHONY: clean
