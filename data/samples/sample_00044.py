import warp as wp

@wp.kernel
def math_0044(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.exp(val * 0.1)
