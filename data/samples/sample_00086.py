import warp as wp

@wp.kernel
def math_0086(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.sqrt(wp.abs(val))
