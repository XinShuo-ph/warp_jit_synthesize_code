
#define WP_TILE_BLOCK_DIM 1
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

#define builtin_block_dim() wp::block_dim()

struct wp_args_kernel_3_b86dc842 {
    wp::array_t<wp::float32> a0;
    wp::array_t<wp::float32> a1;
    wp::array_t<wp::float32> a2;
    wp::array_t<wp::float32> b0;
    wp::array_t<wp::float32> b1;
    wp::float32 scale;
};


void kernel_3_b86dc842_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_3_b86dc842 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_a0 = _wp_args->a0;
    wp::array_t<wp::float32> var_a1 = _wp_args->a1;
    wp::array_t<wp::float32> var_a2 = _wp_args->a2;
    wp::array_t<wp::float32> var_b0 = _wp_args->b0;
    wp::array_t<wp::float32> var_b1 = _wp_args->b1;
    wp::float32 var_scale = _wp_args->scale;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32* var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    //---------
    // forward
    // def kernel_3(a0: wp.array(dtype=float), a1: wp.array(dtype=float), a2: wp.array(dtype=float), b0: wp.array(dtype=float), b1: wp.array(dtype=float), scale: float):       <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // b0[tid] = a0[tid] * scale                                                              <L 6>
    var_1 = wp::address(var_a0, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::mul(var_3, var_scale);
    wp::array_store(var_b0, var_0, var_2);
    // b1[tid] = a0[tid] * scale                                                              <L 7>
    var_4 = wp::address(var_a0, var_0);
    var_6 = wp::load(var_4);
    var_5 = wp::mul(var_6, var_scale);
    wp::array_store(var_b1, var_0, var_5);
}



void kernel_3_b86dc842_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_3_b86dc842 *_wp_args,
    wp_args_kernel_3_b86dc842 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_a0 = _wp_args->a0;
    wp::array_t<wp::float32> var_a1 = _wp_args->a1;
    wp::array_t<wp::float32> var_a2 = _wp_args->a2;
    wp::array_t<wp::float32> var_b0 = _wp_args->b0;
    wp::array_t<wp::float32> var_b1 = _wp_args->b1;
    wp::float32 var_scale = _wp_args->scale;
    wp::array_t<wp::float32> adj_a0 = _wp_adj_args->a0;
    wp::array_t<wp::float32> adj_a1 = _wp_adj_args->a1;
    wp::array_t<wp::float32> adj_a2 = _wp_adj_args->a2;
    wp::array_t<wp::float32> adj_b0 = _wp_adj_args->b0;
    wp::array_t<wp::float32> adj_b1 = _wp_adj_args->b1;
    wp::float32 adj_scale = _wp_adj_args->scale;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32* var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    //---------
    // forward
    // def kernel_3(a0: wp.array(dtype=float), a1: wp.array(dtype=float), a2: wp.array(dtype=float), b0: wp.array(dtype=float), b1: wp.array(dtype=float), scale: float):       <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // b0[tid] = a0[tid] * scale                                                              <L 6>
    var_1 = wp::address(var_a0, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::mul(var_3, var_scale);
    // wp::array_store(var_b0, var_0, var_2);
    // b1[tid] = a0[tid] * scale                                                              <L 7>
    var_4 = wp::address(var_a0, var_0);
    var_6 = wp::load(var_4);
    var_5 = wp::mul(var_6, var_scale);
    // wp::array_store(var_b1, var_0, var_5);
    //---------
    // reverse
    wp::adj_array_store(var_b1, var_0, var_5, adj_b1, adj_0, adj_5);
    wp::adj_mul(var_6, var_scale, adj_4, adj_scale, adj_5);
    wp::adj_address(var_a0, var_0, adj_a0, adj_0, adj_4);
    // adj: b1[tid] = a0[tid] * scale                                                         <L 7>
    wp::adj_array_store(var_b0, var_0, var_2, adj_b0, adj_0, adj_2);
    wp::adj_mul(var_3, var_scale, adj_1, adj_scale, adj_2);
    wp::adj_address(var_a0, var_0, adj_a0, adj_0, adj_1);
    // adj: b0[tid] = a0[tid] * scale                                                         <L 6>
    // adj: tid = wp.tid()                                                                    <L 5>
    // adj: def kernel_3(a0: wp.array(dtype=float), a1: wp.array(dtype=float), a2: wp.array(dtype=float), b0: wp.array(dtype=float), b1: wp.array(dtype=float), scale: float):  <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void kernel_3_b86dc842_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_kernel_3_b86dc842 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_3_b86dc842_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void kernel_3_b86dc842_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_kernel_3_b86dc842 *_wp_args,
    wp_args_kernel_3_b86dc842 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_3_b86dc842_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

