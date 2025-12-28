
# Makefile for ctrl_kjyuat

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
ctrl_kjyuat: ctrl_kjyuat.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
ctrl_kjyuat.ptx: ctrl_kjyuat.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f ctrl_kjyuat ctrl_kjyuat.ptx

.PHONY: clean
