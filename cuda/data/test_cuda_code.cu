
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()



extern "C" __global__ void simple_add_9ad1d227_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_c)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        //---------
        // forward
        // def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):       <L 5>
        // tid = wp.tid()                                                                         <L 6>
        var_0 = builtin_tid1d();
        // c[tid] = a[tid] + b[tid]                                                               <L 7>
        var_1 = wp::address(var_a, var_0);
        var_2 = wp::address(var_b, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::add(var_4, var_5);
        wp::array_store(var_c, var_0, var_3);
    }
}

