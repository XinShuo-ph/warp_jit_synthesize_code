
# Makefile for ctrl_wjujtb

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
ctrl_wjujtb: ctrl_wjujtb.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
ctrl_wjujtb.ptx: ctrl_wjujtb.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f ctrl_wjujtb ctrl_wjujtb.ptx

.PHONY: clean
