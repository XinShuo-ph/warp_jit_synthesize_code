
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


// /workspace/code/examples/example_03_functions.py:18
static wp::vec_t<3, wp::float32> apply_force_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_attractor,
    wp::float32 var_strength)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.01;
    bool var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    //---------
    // forward
    // def apply_force(pos: wp.vec3, attractor: wp.vec3, strength: float) -> wp.vec3:         <L 19>
    // diff = attractor - pos                                                                 <L 21>
    var_0 = wp::sub(var_attractor, var_pos);
    // dist_sq = wp.dot(diff, diff)                                                           <L 22>
    var_1 = wp::dot(var_0, var_0);
    // if dist_sq < 0.01:                                                                     <L 25>
    var_3 = (var_1 < var_2);
    if (var_3) {
        // return wp.vec3(0.0, 0.0, 0.0)                                                      <L 26>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
        return var_7;
    }
    // force_mag = strength / dist_sq                                                         <L 29>
    var_8 = wp::div(var_strength, var_1);
    // direction = wp.normalize(diff)                                                         <L 30>
    var_9 = wp::normalize(var_0);
    // return direction * force_mag                                                           <L 32>
    var_10 = wp::mul(var_9, var_8);
    return var_10;
}


// /workspace/code/examples/example_03_functions.py:18
static void adj_apply_force_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::vec_t<3, wp::float32> var_attractor,
    wp::float32 var_strength,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::vec_t<3, wp::float32> & adj_attractor,
    wp::float32 & adj_strength,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.01;
    bool var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    bool adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::float32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    //---------
    // forward
    // def apply_force(pos: wp.vec3, attractor: wp.vec3, strength: float) -> wp.vec3:         <L 19>
    // diff = attractor - pos                                                                 <L 21>
    var_0 = wp::sub(var_attractor, var_pos);
    // dist_sq = wp.dot(diff, diff)                                                           <L 22>
    var_1 = wp::dot(var_0, var_0);
    // if dist_sq < 0.01:                                                                     <L 25>
    var_3 = (var_1 < var_2);
    if (var_3) {
        // return wp.vec3(0.0, 0.0, 0.0)                                                      <L 26>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
        goto label0;
    }
    // force_mag = strength / dist_sq                                                         <L 29>
    var_8 = wp::div(var_strength, var_1);
    // direction = wp.normalize(diff)                                                         <L 30>
    var_9 = wp::normalize(var_0);
    // return direction * force_mag                                                           <L 32>
    var_10 = wp::mul(var_9, var_8);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_10 += adj_ret;
    wp::adj_mul(var_9, var_8, adj_9, adj_8, adj_10);
    // adj: return direction * force_mag                                                      <L 32>
    wp::adj_normalize(var_0, var_9, adj_0, adj_9);
    // adj: direction = wp.normalize(diff)                                                    <L 30>
    wp::adj_div(var_strength, var_1, var_8, adj_strength, adj_1, adj_8);
    // adj: force_mag = strength / dist_sq                                                    <L 29>
    if (var_3) {
        label0:;
        adj_7 += adj_ret;
        wp::adj_vec_t(var_4, var_5, var_6, adj_4, adj_5, adj_6, adj_7);
        // adj: return wp.vec3(0.0, 0.0, 0.0)                                                 <L 26>
    }
    // adj: if dist_sq < 0.01:                                                                <L 25>
    wp::adj_dot(var_0, var_0, adj_0, adj_0, adj_1);
    // adj: dist_sq = wp.dot(diff, diff)                                                      <L 22>
    wp::adj_sub(var_attractor, var_pos, adj_attractor, adj_pos, adj_0);
    // adj: diff = attractor - pos                                                            <L 21>
    // adj: def apply_force(pos: wp.vec3, attractor: wp.vec3, strength: float) -> wp.vec3:    <L 19>
    return;
}

struct wp_args_n_body_forces_1a00092c {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::vec_t<3, wp::float32>> attractors;
    wp::int32 n_attractors;
    wp::array_t<wp::vec_t<3, wp::float32>> forces;
};


void n_body_forces_1a00092c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_n_body_forces_1a00092c *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_attractors = _wp_args->attractors;
    wp::int32 var_n_attractors = _wp_args->n_attractors;
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces = _wp_args->forces;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::range_t var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 1.0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // forward
    // def n_body_forces(                                                                     <L 36>
    // i = wp.tid()                                                                           <L 42>
    var_0 = builtin_tid1d();
    // pos = positions[i]                                                                     <L 44>
    var_1 = wp::address(var_positions, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // total_force = wp.vec3(0.0, 0.0, 0.0)                                                   <L 45>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // for j in range(n_attractors):                                                          <L 48>
    var_8 = wp::range(var_n_attractors);
    start_for_0:;
        if (iter_cmp(var_8) == 0) goto end_for_0;
        var_9 = wp::iter_next(var_8);
        // attractor_pos = attractors[j]                                                      <L 49>
        var_10 = wp::address(var_attractors, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // force = apply_force(pos, attractor_pos, 1.0)                                       <L 50>
        var_14 = apply_force_0(var_2, var_11, var_13);
        // total_force = total_force + force                                                  <L 51>
        var_15 = wp::add(var_7, var_14);
        wp::assign(var_7, var_15);
        goto start_for_0;
    end_for_0:;
    // forces[i] = total_force                                                                <L 53>
    wp::array_store(var_forces, var_0, var_7);
}



void n_body_forces_1a00092c_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_n_body_forces_1a00092c *_wp_args,
    wp_args_n_body_forces_1a00092c *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_attractors = _wp_args->attractors;
    wp::int32 var_n_attractors = _wp_args->n_attractors;
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces = _wp_args->forces;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_attractors = _wp_adj_args->attractors;
    wp::int32 adj_n_attractors = _wp_adj_args->n_attractors;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces = _wp_adj_args->forces;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::range_t var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 1.0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::range_t adj_8 = {};
    wp::int32 adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::float32 adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    //---------
    // forward
    // def n_body_forces(                                                                     <L 36>
    // i = wp.tid()                                                                           <L 42>
    var_0 = builtin_tid1d();
    // pos = positions[i]                                                                     <L 44>
    var_1 = wp::address(var_positions, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // total_force = wp.vec3(0.0, 0.0, 0.0)                                                   <L 45>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // for j in range(n_attractors):                                                          <L 48>
    var_8 = wp::range(var_n_attractors);
    // forces[i] = total_force                                                                <L 53>
    // wp::array_store(var_forces, var_0, var_7);
    //---------
    // reverse
    wp::adj_array_store(var_forces, var_0, var_7, adj_forces, adj_0, adj_7);
    // adj: forces[i] = total_force                                                           <L 53>
    var_8 = wp::iter_reverse(var_8);
    start_for_0:;
        if (iter_cmp(var_8) == 0) goto end_for_0;
        var_9 = wp::iter_next(var_8);
    	adj_10 = {};
    	adj_11 = {};
    	adj_12 = {};
    	adj_13 = {};
    	adj_14 = {};
    	adj_15 = {};
        // attractor_pos = attractors[j]                                                      <L 49>
        var_10 = wp::address(var_attractors, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // force = apply_force(pos, attractor_pos, 1.0)                                       <L 50>
        var_14 = apply_force_0(var_2, var_11, var_13);
        // total_force = total_force + force                                                  <L 51>
        var_15 = wp::add(var_7, var_14);
        wp::assign(var_7, var_15);
        wp::adj_assign(var_7, var_15, adj_7, adj_15);
        wp::adj_add(var_7, var_14, adj_7, adj_14, adj_15);
        // adj: total_force = total_force + force                                             <L 51>
        adj_apply_force_0(var_2, var_11, var_13, adj_2, adj_11, adj_13, adj_14);
        // adj: force = apply_force(pos, attractor_pos, 1.0)                                  <L 50>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_attractors, var_9, adj_attractors, adj_9, adj_10);
        // adj: attractor_pos = attractors[j]                                                 <L 49>
    	goto start_for_0;
    end_for_0:;
    wp::adj_range(var_n_attractors, adj_n_attractors, adj_8);
    // adj: for j in range(n_attractors):                                                     <L 48>
    wp::adj_vec_t(var_4, var_5, var_6, adj_4, adj_5, adj_6, adj_7);
    // adj: total_force = wp.vec3(0.0, 0.0, 0.0)                                              <L 45>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_positions, var_0, adj_positions, adj_0, adj_1);
    // adj: pos = positions[i]                                                                <L 44>
    // adj: i = wp.tid()                                                                      <L 42>
    // adj: def n_body_forces(                                                                <L 36>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void n_body_forces_1a00092c_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_n_body_forces_1a00092c *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        n_body_forces_1a00092c_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void n_body_forces_1a00092c_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_n_body_forces_1a00092c *_wp_args,
    wp_args_n_body_forces_1a00092c *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        n_body_forces_1a00092c_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

