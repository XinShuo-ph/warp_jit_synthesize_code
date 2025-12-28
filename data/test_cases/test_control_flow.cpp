
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


// /workspace/code/extraction/test_extractor.py:85
static wp::float32 sigmoid_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def sigmoid(x: float) -> float:                                                        <L 86>
    // return 1.0 / (1.0 + wp.exp(-x))                                                        <L 88>
    var_2 = wp::neg(var_x);
    var_3 = wp::exp(var_2);
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::div(var_0, var_4);
    return var_5;
}


// /workspace/code/extraction/test_extractor.py:85
static void adj_sigmoid_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def sigmoid(x: float) -> float:                                                        <L 86>
    // return 1.0 / (1.0 + wp.exp(-x))                                                        <L 88>
    var_2 = wp::neg(var_x);
    var_3 = wp::exp(var_2);
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::div(var_0, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_div(var_0, var_4, var_5, adj_0, adj_4, adj_5);
    wp::adj_add(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_exp(var_2, var_3, adj_2, adj_3);
    wp::adj_neg(var_x, adj_x, adj_2);
    // adj: return 1.0 / (1.0 + wp.exp(-x))                                                   <L 88>
    // adj: def sigmoid(x: float) -> float:                                                   <L 86>
    return;
}

struct wp_args_test_functions_beb46a2c {
    wp::array_t<wp::float32> inputs;
    wp::array_t<wp::float32> weights;
    wp::float32 bias;
    wp::array_t<wp::float32> outputs;
};


void test_functions_beb46a2c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_functions_beb46a2c *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_inputs = _wp_args->inputs;
    wp::array_t<wp::float32> var_weights = _wp_args->weights;
    wp::float32 var_bias = _wp_args->bias;
    wp::array_t<wp::float32> var_outputs = _wp_args->outputs;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    //---------
    // forward
    // def test_functions(inputs: wp.array(dtype=float),                                      <L 92>
    // i = wp.tid()                                                                           <L 97>
    var_0 = builtin_tid1d();
    // z = inputs[i] * weights[i] + bias                                                      <L 100>
    var_1 = wp::address(var_inputs, var_0);
    var_2 = wp::address(var_weights, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::mul(var_4, var_5);
    var_6 = wp::add(var_3, var_bias);
    // outputs[i] = sigmoid(z)                                                                <L 103>
    var_7 = sigmoid_0(var_6);
    wp::array_store(var_outputs, var_0, var_7);
}



void test_functions_beb46a2c_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_functions_beb46a2c *_wp_args,
    wp_args_test_functions_beb46a2c *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_inputs = _wp_args->inputs;
    wp::array_t<wp::float32> var_weights = _wp_args->weights;
    wp::float32 var_bias = _wp_args->bias;
    wp::array_t<wp::float32> var_outputs = _wp_args->outputs;
    wp::array_t<wp::float32> adj_inputs = _wp_adj_args->inputs;
    wp::array_t<wp::float32> adj_weights = _wp_adj_args->weights;
    wp::float32 adj_bias = _wp_adj_args->bias;
    wp::array_t<wp::float32> adj_outputs = _wp_adj_args->outputs;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
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
    //---------
    // forward
    // def test_functions(inputs: wp.array(dtype=float),                                      <L 92>
    // i = wp.tid()                                                                           <L 97>
    var_0 = builtin_tid1d();
    // z = inputs[i] * weights[i] + bias                                                      <L 100>
    var_1 = wp::address(var_inputs, var_0);
    var_2 = wp::address(var_weights, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::mul(var_4, var_5);
    var_6 = wp::add(var_3, var_bias);
    // outputs[i] = sigmoid(z)                                                                <L 103>
    var_7 = sigmoid_0(var_6);
    // wp::array_store(var_outputs, var_0, var_7);
    //---------
    // reverse
    wp::adj_array_store(var_outputs, var_0, var_7, adj_outputs, adj_0, adj_7);
    adj_sigmoid_0(var_6, adj_6, adj_7);
    // adj: outputs[i] = sigmoid(z)                                                           <L 103>
    wp::adj_add(var_3, var_bias, adj_3, adj_bias, adj_6);
    wp::adj_mul(var_4, var_5, adj_1, adj_2, adj_3);
    wp::adj_address(var_weights, var_0, adj_weights, adj_0, adj_2);
    wp::adj_address(var_inputs, var_0, adj_inputs, adj_0, adj_1);
    // adj: z = inputs[i] * weights[i] + bias                                                 <L 100>
    // adj: i = wp.tid()                                                                      <L 97>
    // adj: def test_functions(inputs: wp.array(dtype=float),                                 <L 92>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void test_functions_beb46a2c_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_test_functions_beb46a2c *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_functions_beb46a2c_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void test_functions_beb46a2c_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_test_functions_beb46a2c *_wp_args,
    wp_args_test_functions_beb46a2c *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_functions_beb46a2c_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_test_control_flow_973bf48f {
    wp::array_t<wp::float32> data;
    wp::float32 threshold;
    wp::array_t<wp::float32> output;
};


void test_control_flow_973bf48f_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_control_flow_973bf48f *_wp_args)
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
    const wp::float32 var_4 = 0.0;
    bool var_5;
    wp::float32 var_6;
    bool var_7;
    const wp::float32 var_8 = 2.0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.5;
    wp::float32 var_12;
    wp::float32 var_13;
    //---------
    // forward
    // def test_control_flow(data: wp.array(dtype=float),                                     <L 52>
    // i = wp.tid()                                                                           <L 56>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 58>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val < 0.0:                                                                          <L 60>
    var_5 = (var_2 < var_4);
    if (var_5) {
        // output[i] = -val                                                                   <L 61>
        var_6 = wp::neg(var_2);
        wp::array_store(var_output, var_0, var_6);
    }
    if (!var_5) {
        // elif val < threshold:                                                              <L 62>
        var_7 = (var_2 < var_threshold);
        if (var_7) {
            // output[i] = val * 2.0                                                          <L 63>
            var_9 = wp::mul(var_2, var_8);
            wp::array_store(var_output, var_0, var_9);
        }
        if (!var_7) {
            // output[i] = threshold + (val - threshold) * 0.5                                <L 65>
            var_10 = wp::sub(var_2, var_threshold);
            var_12 = wp::mul(var_10, var_11);
            var_13 = wp::add(var_threshold, var_12);
            wp::array_store(var_output, var_0, var_13);
        }
    }
}



void test_control_flow_973bf48f_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_control_flow_973bf48f *_wp_args,
    wp_args_test_control_flow_973bf48f *_wp_adj_args)
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
    const wp::float32 var_4 = 0.0;
    bool var_5;
    wp::float32 var_6;
    bool var_7;
    const wp::float32 var_8 = 2.0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 0.5;
    wp::float32 var_12;
    wp::float32 var_13;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    bool adj_5 = {};
    wp::float32 adj_6 = {};
    bool adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    //---------
    // forward
    // def test_control_flow(data: wp.array(dtype=float),                                     <L 52>
    // i = wp.tid()                                                                           <L 56>
    var_0 = builtin_tid1d();
    // val = data[i]                                                                          <L 58>
    var_1 = wp::address(var_data, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if val < 0.0:                                                                          <L 60>
    var_5 = (var_2 < var_4);
    if (var_5) {
        // output[i] = -val                                                                   <L 61>
        var_6 = wp::neg(var_2);
        // wp::array_store(var_output, var_0, var_6);
    }
    if (!var_5) {
        // elif val < threshold:                                                              <L 62>
        var_7 = (var_2 < var_threshold);
        if (var_7) {
            // output[i] = val * 2.0                                                          <L 63>
            var_9 = wp::mul(var_2, var_8);
            // wp::array_store(var_output, var_0, var_9);
        }
        if (!var_7) {
            // output[i] = threshold + (val - threshold) * 0.5                                <L 65>
            var_10 = wp::sub(var_2, var_threshold);
            var_12 = wp::mul(var_10, var_11);
            var_13 = wp::add(var_threshold, var_12);
            // wp::array_store(var_output, var_0, var_13);
        }
    }
    //---------
    // reverse
    if (!var_5) {
        if (!var_7) {
            wp::adj_array_store(var_output, var_0, var_13, adj_output, adj_0, adj_13);
            wp::adj_add(var_threshold, var_12, adj_threshold, adj_12, adj_13);
            wp::adj_mul(var_10, var_11, adj_10, adj_11, adj_12);
            wp::adj_sub(var_2, var_threshold, adj_2, adj_threshold, adj_10);
            // adj: output[i] = threshold + (val - threshold) * 0.5                           <L 65>
        }
        if (var_7) {
            wp::adj_array_store(var_output, var_0, var_9, adj_output, adj_0, adj_9);
            wp::adj_mul(var_2, var_8, adj_2, adj_8, adj_9);
            // adj: output[i] = val * 2.0                                                     <L 63>
        }
        // adj: elif val < threshold:                                                         <L 62>
    }
    if (var_5) {
        wp::adj_array_store(var_output, var_0, var_6, adj_output, adj_0, adj_6);
        wp::adj_neg(var_2, adj_2, adj_6);
        // adj: output[i] = -val                                                              <L 61>
    }
    // adj: if val < 0.0:                                                                     <L 60>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_data, var_0, adj_data, adj_0, adj_1);
    // adj: val = data[i]                                                                     <L 58>
    // adj: i = wp.tid()                                                                      <L 56>
    // adj: def test_control_flow(data: wp.array(dtype=float),                                <L 52>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void test_control_flow_973bf48f_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_test_control_flow_973bf48f *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_control_flow_973bf48f_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void test_control_flow_973bf48f_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_test_control_flow_973bf48f *_wp_args,
    wp_args_test_control_flow_973bf48f *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_control_flow_973bf48f_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_test_arithmetic_2cbdc862 {
    wp::array_t<wp::float32> a;
    wp::array_t<wp::float32> b;
    wp::array_t<wp::float32> c;
};


void test_arithmetic_2cbdc862_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_arithmetic_2cbdc862 *_wp_args)
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
    const wp::float32 var_2 = 2.0;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    //---------
    // forward
    // def test_arithmetic(a: wp.array(dtype=float),                                          <L 19>
    // i = wp.tid()                                                                           <L 23>
    var_0 = builtin_tid1d();
    // c[i] = a[i] * 2.0 + b[i] - 1.0                                                         <L 24>
    var_1 = wp::address(var_a, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::mul(var_4, var_2);
    var_5 = wp::address(var_b, var_0);
    var_7 = wp::load(var_5);
    var_6 = wp::add(var_3, var_7);
    var_9 = wp::sub(var_6, var_8);
    wp::array_store(var_c, var_0, var_9);
}



void test_arithmetic_2cbdc862_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_arithmetic_2cbdc862 *_wp_args,
    wp_args_test_arithmetic_2cbdc862 *_wp_adj_args)
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
    const wp::float32 var_2 = 2.0;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32* var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
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
    wp::float32 adj_9 = {};
    //---------
    // forward
    // def test_arithmetic(a: wp.array(dtype=float),                                          <L 19>
    // i = wp.tid()                                                                           <L 23>
    var_0 = builtin_tid1d();
    // c[i] = a[i] * 2.0 + b[i] - 1.0                                                         <L 24>
    var_1 = wp::address(var_a, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::mul(var_4, var_2);
    var_5 = wp::address(var_b, var_0);
    var_7 = wp::load(var_5);
    var_6 = wp::add(var_3, var_7);
    var_9 = wp::sub(var_6, var_8);
    // wp::array_store(var_c, var_0, var_9);
    //---------
    // reverse
    wp::adj_array_store(var_c, var_0, var_9, adj_c, adj_0, adj_9);
    wp::adj_sub(var_6, var_8, adj_6, adj_8, adj_9);
    wp::adj_add(var_3, var_7, adj_3, adj_5, adj_6);
    wp::adj_address(var_b, var_0, adj_b, adj_0, adj_5);
    wp::adj_mul(var_4, var_2, adj_1, adj_2, adj_3);
    wp::adj_address(var_a, var_0, adj_a, adj_0, adj_1);
    // adj: c[i] = a[i] * 2.0 + b[i] - 1.0                                                    <L 24>
    // adj: i = wp.tid()                                                                      <L 23>
    // adj: def test_arithmetic(a: wp.array(dtype=float),                                     <L 19>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void test_arithmetic_2cbdc862_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_test_arithmetic_2cbdc862 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_arithmetic_2cbdc862_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void test_arithmetic_2cbdc862_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_test_arithmetic_2cbdc862 *_wp_args,
    wp_args_test_arithmetic_2cbdc862 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_arithmetic_2cbdc862_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_test_loops_f8db4183 {
    wp::array_t<wp::float32> matrix;
    wp::array_t<wp::float32> vector;
    wp::array_t<wp::float32> result;
    wp::int32 n;
};


void test_loops_f8db4183_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_loops_f8db4183 *_wp_args)
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
    // def test_loops(matrix: wp.array(dtype=float, ndim=2),                                  <L 70>
    // i = wp.tid()                                                                           <L 75>
    var_0 = builtin_tid1d();
    // sum_val = float(0.0)  # Must use float() for mutable loop variable                     <L 77>
    var_2 = wp::float(var_1);
    // for j in range(n):                                                                     <L 78>
    var_3 = wp::range(var_n);
    start_for_0:;
        if (iter_cmp(var_3) == 0) goto end_for_0;
        var_4 = wp::iter_next(var_3);
        // sum_val = sum_val + matrix[i, j] * vector[j]                                       <L 79>
        var_5 = wp::address(var_matrix, var_0, var_4);
        var_6 = wp::address(var_vector, var_4);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::mul(var_8, var_9);
        var_10 = wp::add(var_2, var_7);
        wp::assign(var_2, var_10);
        goto start_for_0;
    end_for_0:;
    // result[i] = sum_val                                                                    <L 81>
    wp::array_store(var_result, var_0, var_2);
}



void test_loops_f8db4183_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_loops_f8db4183 *_wp_args,
    wp_args_test_loops_f8db4183 *_wp_adj_args)
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
    // def test_loops(matrix: wp.array(dtype=float, ndim=2),                                  <L 70>
    // i = wp.tid()                                                                           <L 75>
    var_0 = builtin_tid1d();
    // sum_val = float(0.0)  # Must use float() for mutable loop variable                     <L 77>
    var_2 = wp::float(var_1);
    // for j in range(n):                                                                     <L 78>
    var_3 = wp::range(var_n);
    // result[i] = sum_val                                                                    <L 81>
    // wp::array_store(var_result, var_0, var_2);
    //---------
    // reverse
    wp::adj_array_store(var_result, var_0, var_2, adj_result, adj_0, adj_2);
    // adj: result[i] = sum_val                                                               <L 81>
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
        // sum_val = sum_val + matrix[i, j] * vector[j]                                       <L 79>
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
        // adj: sum_val = sum_val + matrix[i, j] * vector[j]                                  <L 79>
    	goto start_for_0;
    end_for_0:;
    wp::adj_range(var_n, adj_n, adj_3);
    // adj: for j in range(n):                                                                <L 78>
    wp::adj_float(var_1, adj_1, adj_2);
    // adj: sum_val = float(0.0)  # Must use float() for mutable loop variable                <L 77>
    // adj: i = wp.tid()                                                                      <L 75>
    // adj: def test_loops(matrix: wp.array(dtype=float, ndim=2),                             <L 70>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void test_loops_f8db4183_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_test_loops_f8db4183 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_loops_f8db4183_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void test_loops_f8db4183_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_test_loops_f8db4183 *_wp_args,
    wp_args_test_loops_f8db4183 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_loops_f8db4183_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_test_vectors_183acf61 {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::vec_t<3, wp::float32>> velocities;
    wp::array_t<wp::vec_t<3, wp::float32>> forces;
    wp::float32 dt;
};


void test_vectors_183acf61_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_vectors_183acf61 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces = _wp_args->forces;
    wp::float32 var_dt = _wp_args->dt;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 10.0;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 10.0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // forward
    // def test_vectors(positions: wp.array(dtype=wp.vec3),                                   <L 29>
    // i = wp.tid()                                                                           <L 34>
    var_0 = builtin_tid1d();
    // vel = velocities[i]                                                                    <L 36>
    var_1 = wp::address(var_velocities, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // force = forces[i]                                                                      <L 37>
    var_4 = wp::address(var_forces, var_0);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // new_vel = vel + force * dt                                                             <L 40>
    var_7 = wp::mul(var_5, var_dt);
    var_8 = wp::add(var_2, var_7);
    // speed = wp.length(new_vel)                                                             <L 43>
    var_9 = wp::length(var_8);
    // if speed > 10.0:                                                                       <L 44>
    var_11 = (var_9 > var_10);
    if (var_11) {
        // new_vel = wp.normalize(new_vel) * 10.0                                             <L 45>
        var_12 = wp::normalize(var_8);
        var_14 = wp::mul(var_12, var_13);
    }
    var_15 = wp::where(var_11, var_14, var_8);
    // velocities[i] = new_vel                                                                <L 47>
    wp::array_store(var_velocities, var_0, var_15);
}



void test_vectors_183acf61_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_test_vectors_183acf61 *_wp_args,
    wp_args_test_vectors_183acf61 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces = _wp_args->forces;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_velocities = _wp_adj_args->velocities;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces = _wp_adj_args->forces;
    wp::float32 adj_dt = _wp_adj_args->dt;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 10.0;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 10.0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    bool adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::float32 adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    //---------
    // forward
    // def test_vectors(positions: wp.array(dtype=wp.vec3),                                   <L 29>
    // i = wp.tid()                                                                           <L 34>
    var_0 = builtin_tid1d();
    // vel = velocities[i]                                                                    <L 36>
    var_1 = wp::address(var_velocities, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // force = forces[i]                                                                      <L 37>
    var_4 = wp::address(var_forces, var_0);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // new_vel = vel + force * dt                                                             <L 40>
    var_7 = wp::mul(var_5, var_dt);
    var_8 = wp::add(var_2, var_7);
    // speed = wp.length(new_vel)                                                             <L 43>
    var_9 = wp::length(var_8);
    // if speed > 10.0:                                                                       <L 44>
    var_11 = (var_9 > var_10);
    if (var_11) {
        // new_vel = wp.normalize(new_vel) * 10.0                                             <L 45>
        var_12 = wp::normalize(var_8);
        var_14 = wp::mul(var_12, var_13);
    }
    var_15 = wp::where(var_11, var_14, var_8);
    // velocities[i] = new_vel                                                                <L 47>
    // wp::array_store(var_velocities, var_0, var_15);
    //---------
    // reverse
    wp::adj_array_store(var_velocities, var_0, var_15, adj_velocities, adj_0, adj_15);
    // adj: velocities[i] = new_vel                                                           <L 47>
    wp::adj_where(var_11, var_14, var_8, adj_11, adj_14, adj_8, adj_15);
    if (var_11) {
        wp::adj_mul(var_12, var_13, adj_12, adj_13, adj_14);
        wp::adj_normalize(var_8, var_12, adj_8, adj_12);
        // adj: new_vel = wp.normalize(new_vel) * 10.0                                        <L 45>
    }
    // adj: if speed > 10.0:                                                                  <L 44>
    wp::adj_length(var_8, var_9, adj_8, adj_9);
    // adj: speed = wp.length(new_vel)                                                        <L 43>
    wp::adj_add(var_2, var_7, adj_2, adj_7, adj_8);
    wp::adj_mul(var_5, var_dt, adj_5, adj_dt, adj_7);
    // adj: new_vel = vel + force * dt                                                        <L 40>
    wp::adj_copy(var_6, adj_4, adj_5);
    wp::adj_address(var_forces, var_0, adj_forces, adj_0, adj_4);
    // adj: force = forces[i]                                                                 <L 37>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_velocities, var_0, adj_velocities, adj_0, adj_1);
    // adj: vel = velocities[i]                                                               <L 36>
    // adj: i = wp.tid()                                                                      <L 34>
    // adj: def test_vectors(positions: wp.array(dtype=wp.vec3),                              <L 29>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void test_vectors_183acf61_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_test_vectors_183acf61 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_vectors_183acf61_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void test_vectors_183acf61_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_test_vectors_183acf61 *_wp_args,
    wp_args_test_vectors_183acf61 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        test_vectors_183acf61_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

