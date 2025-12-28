
# Makefile for multicond_rmmlsj

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
multicond_rmmlsj: multicond_rmmlsj.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
multicond_rmmlsj.ptx: multicond_rmmlsj.cu
	$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
	rm -f multicond_rmmlsj multicond_rmmlsj.ptx

.PHONY: clean
