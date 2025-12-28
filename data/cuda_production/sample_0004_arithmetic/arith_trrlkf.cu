
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

// CUDA Kernel: arith_trrlkf
__global__ void arith_trrlkf_kernel(float* a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        float var_0 = fabsf(a[idx]);
        float var_1 = fabsf(var_0);
        c[idx] = var_1;
    }
}

// Host code for arith_trrlkf
void launch_arith_trrlkf(int n)
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
    arith_trrlkf_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
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
    
    printf("Launching arith_trrlkf kernel with %d elements\n", n);
    launch_arith_trrlkf(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
