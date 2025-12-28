
# Makefile for multicond_eziiui

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
multicond_eziiui: multicond_eziiui.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
multicond_eziiui.ptx: multicond_eziiui.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f multicond_eziiui multicond_eziiui.ptx

.PHONY: clean
