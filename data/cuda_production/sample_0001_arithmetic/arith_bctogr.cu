
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

// CUDA Kernel: arith_bctogr
__global__ void arith_bctogr_kernel(float* a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        float var_0 = expf(a[idx]);
        float var_1 = var_0 - b[idx];
        float var_2 = expf(var_1);
        c[idx] = var_2;
    }
}

// Host code for arith_bctogr
void launch_arith_bctogr(int n)
{
    // Allocate device memory
    float *d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    float *d_b;
    cudaMalloc(&d_b, n * sizeof(float));
    float *d_c;
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    arith_bctogr_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    int n = 1024;
    
    printf("Launching arith_bctogr kernel with %d elements\n", n);
    launch_arith_bctogr(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
