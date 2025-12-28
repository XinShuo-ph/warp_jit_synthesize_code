
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

// CUDA Kernel: scalar_nehqkc
__global__ void scalar_nehqkc_kernel(float* x, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = (x[idx] + scale) + offset;
    }
}

// Host code for scalar_nehqkc
void launch_scalar_nehqkc(int n)
{
    // Allocate device memory
    float *d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    float *d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    scalar_nehqkc_kernel<<<numBlocks, blockSize>>>(d_x, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_x);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching scalar_nehqkc kernel with %d elements\n", n);
    launch_scalar_nehqkc(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
