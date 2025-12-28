
# Makefile for ctrl_bqlwaq

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
ctrl_bqlwaq: ctrl_bqlwaq.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
ctrl_bqlwaq.ptx: ctrl_bqlwaq.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f ctrl_bqlwaq ctrl_bqlwaq.ptx

.PHONY: clean
