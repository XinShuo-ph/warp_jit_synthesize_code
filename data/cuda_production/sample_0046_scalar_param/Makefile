
# Makefile for scalar_fcrmeo

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
scalar_fcrmeo: scalar_fcrmeo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
scalar_fcrmeo.ptx: scalar_fcrmeo.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f scalar_fcrmeo scalar_fcrmeo.ptx

.PHONY: clean
