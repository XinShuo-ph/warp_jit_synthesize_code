
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

// CUDA Kernel: vec_okuuez
__global__ void vec_okuuez_kernel(float4* a, float4* b, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = wp.length(a[idx]);
    }
}

// Host code for vec_okuuez
void launch_vec_okuuez(int n)
{
    // Allocate device memory
    float4 *d_a;
    cudaMalloc(&d_a, n * sizeof(float4));
    float4 *d_b;
    cudaMalloc(&d_b, n * sizeof(float4));
    float *d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vec_okuuez_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching vec_okuuez kernel with %d elements\n", n);
    launch_vec_okuuez(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
