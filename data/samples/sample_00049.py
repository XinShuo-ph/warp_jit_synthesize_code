import warp as wp

@wp.func
def helper_49(x: float) -> float:
    return wp.sqrt(wp.abs(x))

@wp.kernel
def function_0049(data: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    output[i] = helper_49(val)
