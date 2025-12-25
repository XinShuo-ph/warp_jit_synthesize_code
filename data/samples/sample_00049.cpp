
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


// /tmp/warp_synthesis_f_wq04oe/temp_kernel_49.py:3
static wp::float32 helper_49_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def helper_49(x: float) -> float:                                                      <L 4>
    // return wp.sqrt(wp.abs(x))                                                              <L 5>
    var_0 = wp::abs(var_x);
    var_1 = wp::sqrt(var_0);
    return var_1;
}


// /tmp/warp_synthesis_f_wq04oe/temp_kernel_49.py:3
static void adj_helper_49_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    //---------
    // forward
    // def helper_49(x: float) -> float:                                                      <L 4>
    // return wp.sqrt(wp.abs(x))                                                              <L 5>
    var_0 = wp::abs(var_x);
    var_1 = wp::sqrt(var_0);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_1 += adj_ret;
    wp::adj_sqrt(var_0, var_1, adj_0, adj_1);
    wp::adj_abs(var_x, adj_x, adj_0);
    // adj: return wp.sqrt(wp.abs(x))                                                         <L 5>
    // adj: def helper_49(x: float) -> float:                                                 <L 4>
    return;
}

struct wp_args_function_0049_14a028fc {
    wp::array_t<wp::float32> data;
    wp::array_t<wp::float32> output;
};


void function_0049_14a028fc_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_function_0049_14a028fc *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def function_0049(data: wp.array(dtype=float),                                         <L 8>
    // i = wp.tid()                                                                           <L 10>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 11>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // output[i] = helper_49(val)                                                             <L 12>
    var_4 = helper_49_0(var_2);
    wp::array_store(var_output, var_0, var_4);
}



void function_0049_14a028fc_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_function_0049_14a028fc *_wp_args,
    wp_args_function_0049_14a028fc *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    wp::array_t<wp::float32> adj_data = _wp_adj_args->data;
    wp::array_t<wp::float32> adj_output = _wp_adj_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    //---------
    // forward
    // def function_0049(data: wp.array(dtype=float),                                         <L 8>
    // i = wp.tid()                                                                           <L 10>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 11>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // output[i] = helper_49(val)                                                             <L 12>
    var_4 = helper_49_0(var_2);
    // wp::array_store(var_output, var_0, var_4);
    //---------
    // reverse
    wp::adj_array_store(var_output, var_0, var_4, adj_output, adj_0, adj_4);
    adj_helper_49_0(var_2, adj_2, adj_4);
    // adj: output[i] = helper_49(val)                                                        <L 12>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_data, var_0, adj_data, adj_0, adj_1);
    // adj: val = data[i]                                                                     <L 11>
    // adj: i = wp.tid()                                                                      <L 10>
    // adj: def function_0049(data: wp.array(dtype=float),                                    <L 8>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void function_0049_14a028fc_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_function_0049_14a028fc *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        function_0049_14a028fc_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void function_0049_14a028fc_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_function_0049_14a028fc *_wp_args,
    wp_args_function_0049_14a028fc *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        function_0049_14a028fc_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

