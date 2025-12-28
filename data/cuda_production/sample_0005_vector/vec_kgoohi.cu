
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

// CUDA Kernel: vec_kgoohi
__global__ void vec_kgoohi_kernel(float3* a, float3* b, float3* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = wp.cross(a[idx], b[idx]);
    }
}

// Host code for vec_kgoohi
void launch_vec_kgoohi(int n)
{
    // Allocate device memory
    float3 *d_a;
    cudaMalloc(&d_a, n * sizeof(float3));
    float3 *d_b;
    cudaMalloc(&d_b, n * sizeof(float3));
    float3 *d_out;
    cudaMalloc(&d_out, n * sizeof(float3));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vec_kgoohi_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_out, n);
    
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
    
    printf("Launching vec_kgoohi kernel with %d elements\n", n);
    launch_vec_kgoohi(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
