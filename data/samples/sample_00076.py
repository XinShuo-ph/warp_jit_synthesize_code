import warp as wp

@wp.func
def helper_76(x: float) -> float:
    return x * x + 1.0

@wp.kernel
def function_0076(data: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    output[i] = helper_76(val)
