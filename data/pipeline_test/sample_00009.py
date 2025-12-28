import warp as wp

@wp.func
def helper_9(x: float) -> float:
    return x * x + 1.0

@wp.kernel
def function_0009(data: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    output[i] = helper_9(val)
