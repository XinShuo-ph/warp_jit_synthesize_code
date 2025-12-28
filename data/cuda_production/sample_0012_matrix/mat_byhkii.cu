
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

// CUDA Kernel: mat_byhkii
__global__ void mat_byhkii_kernel(float* m, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread ID computed from idx
        out[idx] = wp.transpose(m[idx]);
    }
}

// Host code for mat_byhkii
void launch_mat_byhkii(int n)
{
    // Allocate device memory
    float *d_m;
    cudaMalloc(&d_m, n * sizeof(float));
    float *d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mat_byhkii_kernel<<<numBlocks, blockSize>>>(d_m, d_out, n);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_m);
    cudaFree(d_out);
}

int main()
{
    int n = 1024;
    
    printf("Launching mat_byhkii kernel with %d elements\n", n);
    launch_mat_byhkii(n);
    
    printf("Kernel completed successfully\n");
    return 0;
}
