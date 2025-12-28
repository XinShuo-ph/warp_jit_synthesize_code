
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

// CUDA Kernel: math_wgnhxn
__global__ void math_wgnhxn_kernel(float* a, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = logf(fabsf(a[idx]) + 1.0);
    }
}

// Host code for math_wgnhxn
void launch_math_wgnhxn(int n)
{
    // Allocate device memory
    float *d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    float *d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    math_wgnhxn_kernel<<<numBlocks, blockSize>>>(d_a, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching math_wgnhxn kernel with %d elements\n", n);
    launch_math_wgnhxn(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
