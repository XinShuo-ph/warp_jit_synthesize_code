
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

struct wp_args_kernel_4_06582103 {
    wp::array_t<wp::float32> values;
    wp::array_t<wp::float32> result;
};


void kernel_4_06582103_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_4_06582103 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_values = _wp_args->values;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    //---------
    // forward
    // def kernel_4(values: wp.array(dtype=float), result: wp.array(dtype=float)):            <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // val = values[tid]                                                                      <L 6>
    var_1 = wp::address(var_values, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val > 0.0:                                                                          <L 8>
    var_5 = (var_2 > var_4);
    if (var_5) {
        // wp.atomic_add(result, 0, val)                                                      <L 9>
        var_7 = wp::atomic_add(var_result, var_6, var_2);
    }
}



void kernel_4_06582103_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_4_06582103 *_wp_args,
    wp_args_kernel_4_06582103 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_values = _wp_args->values;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::array_t<wp::float32> adj_values = _wp_adj_args->values;
    wp::array_t<wp::float32> adj_result = _wp_adj_args->result;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float32 adj_7 = {};
    //---------
    // forward
    // def kernel_4(values: wp.array(dtype=float), result: wp.array(dtype=float)):            <L 4>
    // tid = wp.tid()                                                                         <L 5>
    var_0 = builtin_tid1d();
    // val = values[tid]                                                                      <L 6>
    var_1 = wp::address(var_values, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val > 0.0:                                                                          <L 8>
    var_5 = (var_2 > var_4);
    if (var_5) {
        // wp.atomic_add(result, 0, val)                                                      <L 9>
        // var_7 = wp::atomic_add(var_result, var_6, var_2);
    }
    //---------
    // reverse
    if (var_5) {
        wp::adj_atomic_add(var_result, var_6, var_2, adj_result, adj_6, adj_2, adj_7);
        // adj: wp.atomic_add(result, 0, val)                                                 <L 9>
    }
    // adj: if val > 0.0:                                                                     <L 8>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_values, var_0, adj_values, adj_0, adj_1);
    // adj: val = values[tid]                                                                 <L 6>
    // adj: tid = wp.tid()                                                                    <L 5>
    // adj: def kernel_4(values: wp.array(dtype=float), result: wp.array(dtype=float)):       <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void kernel_4_06582103_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_kernel_4_06582103 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_4_06582103_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void kernel_4_06582103_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_kernel_4_06582103 *_wp_args,
    wp_args_kernel_4_06582103 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        kernel_4_06582103_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

