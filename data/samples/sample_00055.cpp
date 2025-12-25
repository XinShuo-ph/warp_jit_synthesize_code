
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

struct wp_args_conditional_0055_08a67bfd {
    wp::array_t<wp::float32> data;
    wp::float32 threshold;
    wp::array_t<wp::float32> output;
};


void conditional_0055_08a67bfd_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_conditional_0055_08a67bfd *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::float32 var_threshold = _wp_args->threshold;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    bool var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.5;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = 0.5;
    wp::float32 var_11;
    //---------
    // forward
    // def conditional_0055(data: wp.array(dtype=float),                                      <L 4>
    // i = wp.tid()                                                                           <L 7>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 8>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val < threshold:                                                                    <L 10>
    var_4 = (var_2 < var_threshold);
    if (var_4) {
        // output[i] = val * 2.0                                                              <L 11>
        var_6 = wp::mul(var_2, var_5);
        wp::array_store(var_output, var_0, var_6);
    }
    if (!var_4) {
        // elif val < threshold * 0.5:                                                        <L 12>
        var_8 = wp::mul(var_threshold, var_7);
        var_9 = (var_2 < var_8);
        if (var_9) {
            // output[i] = threshold                                                          <L 13>
            wp::array_store(var_output, var_0, var_threshold);
        }
        if (!var_9) {
            // output[i] = val * 0.5                                                          <L 15>
            var_11 = wp::mul(var_2, var_10);
            wp::array_store(var_output, var_0, var_11);
        }
    }
}



void conditional_0055_08a67bfd_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_conditional_0055_08a67bfd *_wp_args,
    wp_args_conditional_0055_08a67bfd *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::float32 var_threshold = _wp_args->threshold;
    wp::array_t<wp::float32> var_output = _wp_args->output;
    wp::array_t<wp::float32> adj_data = _wp_adj_args->data;
    wp::float32 adj_threshold = _wp_adj_args->threshold;
    wp::array_t<wp::float32> adj_output = _wp_adj_args->output;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    bool var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.5;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = 0.5;
    wp::float32 var_11;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    bool adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    bool adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    //---------
    // forward
    // def conditional_0055(data: wp.array(dtype=float),                                      <L 4>
    // i = wp.tid()                                                                           <L 7>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 8>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val < threshold:                                                                    <L 10>
    var_4 = (var_2 < var_threshold);
    if (var_4) {
        // output[i] = val * 2.0                                                              <L 11>
        var_6 = wp::mul(var_2, var_5);
        // wp::array_store(var_output, var_0, var_6);
    }
    if (!var_4) {
        // elif val < threshold * 0.5:                                                        <L 12>
        var_8 = wp::mul(var_threshold, var_7);
        var_9 = (var_2 < var_8);
        if (var_9) {
            // output[i] = threshold                                                          <L 13>
            // wp::array_store(var_output, var_0, var_threshold);
        }
        if (!var_9) {
            // output[i] = val * 0.5                                                          <L 15>
            var_11 = wp::mul(var_2, var_10);
            // wp::array_store(var_output, var_0, var_11);
        }
    }
    //---------
    // reverse
    if (!var_4) {
        if (!var_9) {
            wp::adj_array_store(var_output, var_0, var_11, adj_output, adj_0, adj_11);
            wp::adj_mul(var_2, var_10, adj_2, adj_10, adj_11);
            // adj: output[i] = val * 0.5                                                     <L 15>
        }
        if (var_9) {
            wp::adj_array_store(var_output, var_0, var_threshold, adj_output, adj_0, adj_threshold);
            // adj: output[i] = threshold                                                     <L 13>
        }
        wp::adj_mul(var_threshold, var_7, adj_threshold, adj_7, adj_8);
        // adj: elif val < threshold * 0.5:                                                   <L 12>
    }
    if (var_4) {
        wp::adj_array_store(var_output, var_0, var_6, adj_output, adj_0, adj_6);
        wp::adj_mul(var_2, var_5, adj_2, adj_5, adj_6);
        // adj: output[i] = val * 2.0                                                         <L 11>
    }
    // adj: if val < threshold:                                                               <L 10>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_data, var_0, adj_data, adj_0, adj_1);
    // adj: val = data[i]                                                                     <L 8>
    // adj: i = wp.tid()                                                                      <L 7>
    // adj: def conditional_0055(data: wp.array(dtype=float),                                 <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void conditional_0055_08a67bfd_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_conditional_0055_08a67bfd *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        conditional_0055_08a67bfd_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void conditional_0055_08a67bfd_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_conditional_0055_08a67bfd *_wp_args,
    wp_args_conditional_0055_08a67bfd *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        conditional_0055_08a67bfd_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

