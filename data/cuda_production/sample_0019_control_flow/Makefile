
# Makefile for ctrl_scjoaa

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
ctrl_scjoaa: ctrl_scjoaa.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
ctrl_scjoaa.ptx: ctrl_scjoaa.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f ctrl_scjoaa ctrl_scjoaa.ptx

.PHONY: clean
