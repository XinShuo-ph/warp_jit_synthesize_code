import warp as wp

@wp.func
def helper_4(x: float) -> float:
    return x * x + 1.0

@wp.kernel
def function_0004(data: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    output[i] = helper_4(val)
