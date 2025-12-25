
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

struct wp_args_reduction_0074_33e10d7a {
    wp::array_t<wp::float32> data;
    wp::array_t<wp::float32> result;
    wp::int32 n;
};


void reduction_0074_33e10d7a_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_reduction_0074_33e10d7a *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::int32 var_n = _wp_args->n;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::range_t var_8;
    wp::int32 var_9;
    wp::float32* var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    //---------
    // forward
    // def reduction_0074(data: wp.array(dtype=float),                                        <L 4>
    // tid = wp.tid()                                                                         <L 7>
    var_0 = builtin_tid1d();
    // local_result = float(0.0) if tid < n else float(0.0)                                   <L 10>
    var_1 = (var_0 < var_n);
    if (var_1) {
        var_3 = wp::float(var_2);
    }
    if (!var_1) {
        var_5 = wp::float(var_4);
    }
    var_6 = wp::where(var_1, var_3, var_5);
    // for i in range(tid, n, 1):                                                             <L 12>
    var_8 = wp::range(var_0, var_n, var_7);
    start_for_0:;
        if (iter_cmp(var_8) == 0) goto end_for_0;
        var_9 = wp::iter_next(var_8);
        // val = data[i]                                                                      <L 13>
        var_10 = wp::address(var_data, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // local_result = local_result + val * val                                            <L 14>
        var_13 = wp::mul(var_11, var_11);
        var_14 = wp::add(var_6, var_13);
        // break  # Simplified                                                                <L 15>
        wp::assign(var_6, var_14);
        goto end_for_0;
        goto start_for_0;
    end_for_0:;
    // result[tid] = local_result                                                             <L 17>
    wp::array_store(var_result, var_0, var_6);
}



void reduction_0074_33e10d7a_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_reduction_0074_33e10d7a *_wp_args,
    wp_args_reduction_0074_33e10d7a *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_data = _wp_args->data;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::int32 var_n = _wp_args->n;
    wp::array_t<wp::float32> adj_data = _wp_adj_args->data;
    wp::array_t<wp::float32> adj_result = _wp_adj_args->result;
    wp::int32 adj_n = _wp_adj_args->n;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::range_t var_8;
    wp::int32 var_9;
    wp::float32* var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::range_t adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    //---------
    // forward
    // def reduction_0074(data: wp.array(dtype=float),                                        <L 4>
    // tid = wp.tid()                                                                         <L 7>
    var_0 = builtin_tid1d();
    // local_result = float(0.0) if tid < n else float(0.0)                                   <L 10>
    var_1 = (var_0 < var_n);
    if (var_1) {
        var_3 = wp::float(var_2);
    }
    if (!var_1) {
        var_5 = wp::float(var_4);
    }
    var_6 = wp::where(var_1, var_3, var_5);
    // for i in range(tid, n, 1):                                                             <L 12>
    var_8 = wp::range(var_0, var_n, var_7);
    // result[tid] = local_result                                                             <L 17>
    // wp::array_store(var_result, var_0, var_6);
    //---------
    // reverse
    wp::adj_array_store(var_result, var_0, var_6, adj_result, adj_0, adj_6);
    // adj: result[tid] = local_result                                                        <L 17>
    var_8 = wp::iter_reverse(var_8);
    start_for_0:;
        if (iter_cmp(var_8) == 0) goto end_for_0;
        var_9 = wp::iter_next(var_8);
    	adj_10 = {};
    	adj_11 = {};
    	adj_12 = {};
    	adj_13 = {};
    	adj_14 = {};
        // val = data[i]                                                                      <L 13>
        var_10 = wp::address(var_data, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // local_result = local_result + val * val                                            <L 14>
        var_13 = wp::mul(var_11, var_11);
        var_14 = wp::add(var_6, var_13);
        // break  # Simplified                                                                <L 15>
        wp::assign(var_6, var_14);
        goto end_for_0;
        wp::adj_assign(var_6, var_14, adj_6, adj_14);
        // adj: break  # Simplified                                                           <L 15>
        wp::adj_add(var_6, var_13, adj_6, adj_13, adj_14);
        wp::adj_mul(var_11, var_11, adj_11, adj_11, adj_13);
        // adj: local_result = local_result + val * val                                       <L 14>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_data, var_9, adj_data, adj_9, adj_10);
        // adj: val = data[i]                                                                 <L 13>
    	goto start_for_0;
    end_for_0:;
    wp::adj_range(var_0, var_n, var_7, adj_0, adj_n, adj_7, adj_8);
    // adj: for i in range(tid, n, 1):                                                        <L 12>
    wp::adj_where(var_1, var_3, var_5, adj_1, adj_3, adj_5, adj_6);
    if (!var_1) {
        wp::adj_float(var_4, adj_4, adj_5);
    }
    if (var_1) {
        wp::adj_float(var_2, adj_2, adj_3);
    }
    // adj: local_result = float(0.0) if tid < n else float(0.0)                              <L 10>
    // adj: tid = wp.tid()                                                                    <L 7>
    // adj: def reduction_0074(data: wp.array(dtype=float),                                   <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void reduction_0074_33e10d7a_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_reduction_0074_33e10d7a *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        reduction_0074_33e10d7a_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void reduction_0074_33e10d7a_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_reduction_0074_33e10d7a *_wp_args,
    wp_args_reduction_0074_33e10d7a *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        reduction_0074_33e10d7a_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

