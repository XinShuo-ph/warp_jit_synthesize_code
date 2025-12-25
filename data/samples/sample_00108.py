import warp as wp

@wp.kernel
def math_0108(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.pow(val, 2.0)
