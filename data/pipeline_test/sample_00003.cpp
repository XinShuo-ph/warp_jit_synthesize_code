
#define WP_TILE_BLOCK_DIM 256
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

struct wp_args_math_0003_27fa5210 {
    wp::array_t<wp::float32> data;
    wp::float32 scale;
    wp::array_t<wp::float32> output;
};


void math_0003_27fa5210_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_math_0003_27fa5210 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::float32 var_scale = _wp_args->scale;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 2.0;
    wp::float32 var_5;
    //---------
    // forward
    // def math_0003(data: wp.array(dtype=float),                                             <L 4>
    // i = wp.tid()                                                                           <L 7>
    var_0 = builtin_tid1d();
    // val = data[i] * scale                                                                  <L 8>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::mul(var_3, var_scale);
    // output[i] = wp.pow(val, 2.0)                                                           <L 9>
    var_5 = wp::pow(var_2, var_4);
    wp::array_store(var_output, var_0, var_5);
}



void math_0003_27fa5210_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_math_0003_27fa5210 *_wp_args,
    wp_args_math_0003_27fa5210 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::float32 var_scale = _wp_args->scale;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    wp::array_t<wp::float32> adj_data = _wp_adj_args->data;
    wp::float32 adj_scale = _wp_adj_args->scale;
    wp::array_t<wp::float32> adj_output = _wp_adj_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 2.0;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def math_0003(data: wp.array(dtype=float),                                             <L 4>
    // i = wp.tid()                                                                           <L 7>
    var_0 = builtin_tid1d();
    // val = data[i] * scale                                                                  <L 8>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::mul(var_3, var_scale);
    // output[i] = wp.pow(val, 2.0)                                                           <L 9>
    var_5 = wp::pow(var_2, var_4);
    // wp::array_store(var_output, var_0, var_5);
    //---------
    // reverse
    wp::adj_array_store(var_output, var_0, var_5, adj_output, adj_0, adj_5);
    wp::adj_pow(var_2, var_4, var_5, adj_2, adj_4, adj_5);
    // adj: output[i] = wp.pow(val, 2.0)                                                      <L 9>
    wp::adj_mul(var_3, var_scale, adj_1, adj_scale, adj_2);
    wp::adj_address(var_data, var_0, adj_data, adj_0, adj_1);
    // adj: val = data[i] * scale                                                             <L 8>
    // adj: i = wp.tid()                                                                      <L 7>
    // adj: def math_0003(data: wp.array(dtype=float),                                        <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void math_0003_27fa5210_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_math_0003_27fa5210 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        math_0003_27fa5210_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void math_0003_27fa5210_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_math_0003_27fa5210 *_wp_args,
    wp_args_math_0003_27fa5210 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        math_0003_27fa5210_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

