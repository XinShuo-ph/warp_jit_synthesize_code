
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA Kernel: multicond_gyspox
__global__ void multicond_gyspox_kernel(float* x, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        float val = x[idx];
        // TODO: if val < 1.08:
        out[idx] = val * 0.5;
        // TODO: elif val < 2.87:
        out[idx] = val * 1.0;
        // TODO: else:
        out[idx] = val * 2.0;
    }
}

// Host code for multicond_gyspox
void launch_multicond_gyspox(int n)
{
    // Allocate device memory
    float *d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    float *d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    multicond_gyspox_kernel<<<numBlocks, blockSize>>>(d_x, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_x);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching multicond_gyspox kernel with %d elements\n", n);
    launch_multicond_gyspox(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
