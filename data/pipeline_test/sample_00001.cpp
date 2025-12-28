
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

struct wp_args_arithmetic_0001_fc7af92c {
    wp::array_t<wp::float32> a;
    wp::array_t<wp::float32> b;
    wp::array_t<wp::float32> c;
};


void arithmetic_0001_fc7af92c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_arithmetic_0001_fc7af92c *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_a = _wp_args->a;
    wp::array_t<wp::float32> var_b = _wp_args->b;
    wp::array_t<wp::float32> var_c = _wp_args->c;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def arithmetic_0001(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):       <L 4>
    // i = wp.tid()                                                                           <L 5>
    var_0 = builtin_tid1d();
    // c[i] = ((a[i] * b[i]) + b[i])                                                          <L 6>
    var_1 = wp::address(var_a, var_0);
    var_2 = wp::address(var_b, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::mul(var_4, var_5);
    var_6 = wp::address(var_b, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::add(var_3, var_8);
    wp::array_store(var_c, var_0, var_7);
}



void arithmetic_0001_fc7af92c_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_arithmetic_0001_fc7af92c *_wp_args,
    wp_args_arithmetic_0001_fc7af92c *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_a = _wp_args->a;
    wp::array_t<wp::float32> var_b = _wp_args->b;
    wp::array_t<wp::float32> var_c = _wp_args->c;
    wp::array_t<wp::float32> adj_a = _wp_adj_args->a;
    wp::array_t<wp::float32> adj_b = _wp_adj_args->b;
    wp::array_t<wp::float32> adj_c = _wp_adj_args->c;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32* var_6;
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
    // def arithmetic_0001(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):       <L 4>
    // i = wp.tid()                                                                           <L 5>
    var_0 = builtin_tid1d();
    // c[i] = ((a[i] * b[i]) + b[i])                                                          <L 6>
    var_1 = wp::address(var_a, var_0);
    var_2 = wp::address(var_b, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::mul(var_4, var_5);
    var_6 = wp::address(var_b, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::add(var_3, var_8);
    // wp::array_store(var_c, var_0, var_7);
    //---------
    // reverse
    wp::adj_array_store(var_c, var_0, var_7, adj_c, adj_0, adj_7);
    wp::adj_add(var_3, var_8, adj_3, adj_6, adj_7);
    wp::adj_address(var_b, var_0, adj_b, adj_0, adj_6);
    wp::adj_mul(var_4, var_5, adj_1, adj_2, adj_3);
    wp::adj_address(var_b, var_0, adj_b, adj_0, adj_2);
    wp::adj_address(var_a, var_0, adj_a, adj_0, adj_1);
    // adj: c[i] = ((a[i] * b[i]) + b[i])                                                     <L 6>
    // adj: i = wp.tid()                                                                      <L 5>
    // adj: def arithmetic_0001(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):  <L 4>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void arithmetic_0001_fc7af92c_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_arithmetic_0001_fc7af92c *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        arithmetic_0001_fc7af92c_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void arithmetic_0001_fc7af92c_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_arithmetic_0001_fc7af92c *_wp_args,
    wp_args_arithmetic_0001_fc7af92c *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        arithmetic_0001_fc7af92c_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

