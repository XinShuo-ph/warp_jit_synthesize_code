
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

// CUDA Kernel: mat_akfmrx
__global__ void mat_akfmrx_kernel(float* m, float2* v, float2* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = m[idx] * v[idx];
    }
}

// Host code for mat_akfmrx
void launch_mat_akfmrx(int n)
{
    // Allocate device memory
    float *d_m;
    cudaMalloc(&d_m, n * sizeof(float));
    float2 *d_v;
    cudaMalloc(&d_v, n * sizeof(float2));
    float2 *d_out;
    cudaMalloc(&d_out, n * sizeof(float2));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mat_akfmrx_kernel<<<numBlocks, blockSize>>>(d_m, d_v, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching mat_akfmrx kernel with %d elements\n", n);
    launch_mat_akfmrx(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
