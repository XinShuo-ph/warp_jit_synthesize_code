
# Makefile for atom_qharmi

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
atom_qharmi: atom_qharmi.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
atom_qharmi.ptx: atom_qharmi.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f atom_qharmi atom_qharmi.ptx

.PHONY: clean
