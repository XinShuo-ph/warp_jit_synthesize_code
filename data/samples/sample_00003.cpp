
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

struct wp_args_loop_0003_6cd6e428 {
    wp::array_t<wp::float32> matrix;
    wp::array_t<wp::float32> vector;
    wp::array_t<wp::float32> result;
    wp::int32 n;
};


void loop_0003_6cd6e428_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_loop_0003_6cd6e428 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_matrix = _wp_args->matrix;
    wp::array_t<wp::float32> var_vector = _wp_args->vector;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::int32 var_n = _wp_args->n;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    wp::float32 var_2;
    wp::range_t var_3;
    wp::int32 var_4;
    wp::float32* var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def loop_0003(matrix: wp.array(dtype=float, ndim=2),                                   <L 4>
    // i = wp.tid()                                                                           <L 8>
    var_0 = builtin_tid1d();
    // sum_val = float(0.0)                                                                   <L 10>
    var_2 = wp::float(var_1);
    // for j in range(n):                                                                     <L 11>
    var_3 = wp::range(var_n);
    start_for_0:;
        if (iter_cmp(var_3) == 0) goto end_for_0;
        var_4 = wp::iter_next(var_3);
        // sum_val = sum_val + matrix[i, j] * vector[j]                                       <L 12>
        var_5 = wp::address(var_matrix, var_0, var_4);
        var_6 = wp::address(var_vector, var_4);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::mul(var_8, var_9);
        var_10 = wp::add(var_2, var_7);
        wp::assign(var_2, var_10);
        goto start_for_0;
    end_for_0:;
    // result[i] = sum_val                                                                    <L 14>
    wp::array_store(var_result, var_0, var_2);
}



void loop_0003_6cd6e428_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_loop_0003_6cd6e428 *_wp_args,
    wp_args_loop_0003_6cd6e428 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_matrix = _wp_args->matrix;
    wp::array_t<wp::float32> var_vector = _wp_args->vector;
    wp::array_t<wp::float32> var_result = _wp_args->result;
    wp::int32 var_n = _wp_args->n;
    wp::array_t<wp::float32> adj_matrix = _wp_adj_args->matrix;
    wp::array_t<wp::float32> adj_vector = _wp_adj_args->vector;
    wp::array_t<wp::float32> adj_result = _wp_adj_args->result;
    wp::int32 adj_n = _wp_adj_args->n;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    wp::float32 var_2;
    wp::range_t var_3;
    wp::int32 var_4;
    wp::float32* var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::range_t adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    //---------
    // forward
    // def loop_0003(matrix: wp.array(dtype=float, ndim=2),                                   <L 4>
    // i = wp.tid()                                                                           <L 8>
    var_0 = builtin_tid1d();
    // sum_val = float(0.0)                                                                   <L 10>
    var_2 = wp::float(var_1);
    // for j in range(n):                                                                     <L 11>
    var_3 = wp::range(var_n);
    // result[i] = sum_val                                                                    <L 14>
    // wp::array_store(var_result, var_0, var_2);
    //---------
    // reverse
    wp::adj_array_store(var_result, var_0, var_2, adj_result, adj_0, adj_2);
    // adj: result[i] = sum_val                                                               <L 14>
    var_3 = wp::iter_reverse(var_3);
    start_for_0:;
        if (iter_cmp(var_3) == 0) goto end_for_0;
        var_4 = wp::iter_next(var_3);
    	adj_5 = {};
    	adj_6 = {};
    	adj_7 = {};
    	adj_8 = {};
    	adj_9 = {};
    	adj_10 = {};
        // sum_val = sum_val + matrix[i, j] * vector[j]                                       <L 12>
        var_5 = wp::address(var_matrix, var_0, var_4);
        var_6 = wp::address(var_vector, var_4);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::mul(var_8, var_9);
        var_10 = wp::add(var_2, var_7);
        wp::assign(var_2, var_10);
        wp::adj_assign(var_2, var_10, adj_2, adj_10);
        wp::adj_add(var_2, var_7, adj_2, adj_7, adj_10);
        wp::adj_mul(var_8, var_9, adj_5, adj_6, adj_7);
        wp::adj_address(var_vector, var_4, adj_vector, adj_4, adj_6);
        wp::adj_address(var_matrix, var_0, var_4, adj_matrix, adj_0, adj_4, adj_5);
        // adj: sum_val = sum_val + matrix[i, j] * vector[j]                                  <L 12>
    	goto start_for_0;
    end_for_0:;
    wp::adj_range(var_n, adj_n, adj_3);
    // adj: for j in range(n):                                                                <L 11>
    wp::adj_float(var_1, adj_1, adj_2);
    // adj: sum_val = float(0.0)                                                              <L 10>
    // adj: i = wp.tid()                                                                      <L 8>
    // adj: def loop_0003(matrix: wp.array(dtype=float, ndim=2),                              <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void loop_0003_6cd6e428_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_loop_0003_6cd6e428 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        loop_0003_6cd6e428_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void loop_0003_6cd6e428_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_loop_0003_6cd6e428 *_wp_args,
    wp_args_loop_0003_6cd6e428 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        loop_0003_6cd6e428_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

