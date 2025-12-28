
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

struct wp_args_simple_add_9ad1d227 {
    wp::array_t<wp::float32> a;
    wp::array_t<wp::float32> b;
    wp::array_t<wp::float32> c;
};


void simple_add_9ad1d227_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_simple_add_9ad1d227 *_wp_args)
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



extern "C" {

// Python CPU entry points
WP_API void simple_add_9ad1d227_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_simple_add_9ad1d227 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        simple_add_9ad1d227_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C

