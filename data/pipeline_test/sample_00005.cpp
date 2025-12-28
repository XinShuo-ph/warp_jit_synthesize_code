
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

struct wp_args_kernel_5_8ed964ac {
    wp::array_t<wp::float32> x;
    wp::array_t<wp::float32> result;
};


void kernel_5_8ed964ac_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_5_8ed964ac *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_x = _wp_args->x;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def kernel_5(x: wp.array(dtype=float), result: wp.array(dtype=float)):                 <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // val = x[tid]                                                                           <L 6>
    var_1 = wp::address(var_x, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // result[tid] = wp.sin(val) + wp.cos(val * 2.0)                                          <L 7>
    var_4 = wp::sin(var_2);
    var_6 = wp::mul(var_2, var_5);
    var_7 = wp::cos(var_6);
    var_8 = wp::add(var_4, var_7);
    wp::array_store(var_result, var_0, var_8);
}



void kernel_5_8ed964ac_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_5_8ed964ac *_wp_args,
    wp_args_kernel_5_8ed964ac *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_x = _wp_args->x;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::array_t<wp::float32> adj_x = _wp_adj_args->x;
    wp::array_t<wp::float32> adj_result = _wp_adj_args->result;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    //---------
    // forward
    // def kernel_5(x: wp.array(dtype=float), result: wp.array(dtype=float)):                 <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // val = x[tid]                                                                           <L 6>
    var_1 = wp::address(var_x, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // result[tid] = wp.sin(val) + wp.cos(val * 2.0)                                          <L 7>
    var_4 = wp::sin(var_2);
    var_6 = wp::mul(var_2, var_5);
    var_7 = wp::cos(var_6);
    var_8 = wp::add(var_4, var_7);
    // wp::array_store(var_result, var_0, var_8);
    //---------
    // reverse
    wp::adj_array_store(var_result, var_0, var_8, adj_result, adj_0, adj_8);
    wp::adj_add(var_4, var_7, adj_4, adj_7, adj_8);
    wp::adj_cos(var_6, adj_6, adj_7);
    wp::adj_mul(var_2, var_5, adj_2, adj_5, adj_6);
    wp::adj_sin(var_2, adj_2, adj_4);
    // adj: result[tid] = wp.sin(val) + wp.cos(val * 2.0)                                     <L 7>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_x, var_0, adj_x, adj_0, adj_1);
    // adj: val = x[tid]                                                                      <L 6>
    // adj: tid = wp.tid()                                                                    <L 5>
    // adj: def kernel_5(x: wp.array(dtype=float), result: wp.array(dtype=float)):            <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void kernel_5_8ed964ac_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_kernel_5_8ed964ac *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_5_8ed964ac_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void kernel_5_8ed964ac_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_kernel_5_8ed964ac *_wp_args,
    wp_args_kernel_5_8ed964ac *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_5_8ed964ac_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

