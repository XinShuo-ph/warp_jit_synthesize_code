import warp as wp

@wp.kernel
def math_0065(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.sin(val * 3.14159)
