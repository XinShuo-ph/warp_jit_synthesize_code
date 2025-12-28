
# Makefile for atom_kltzyb

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
atom_kltzyb: atom_kltzyb.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
atom_kltzyb.ptx: atom_kltzyb.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f atom_kltzyb atom_kltzyb.ptx

.PHONY: clean
