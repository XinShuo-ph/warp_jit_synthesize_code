
# Makefile for combined_sfdqao

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
combined_sfdqao: combined_sfdqao.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
combined_sfdqao.ptx: combined_sfdqao.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f combined_sfdqao combined_sfdqao.ptx

.PHONY: clean
