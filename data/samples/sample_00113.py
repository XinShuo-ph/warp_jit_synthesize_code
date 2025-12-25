import warp as wp

@wp.kernel
def math_0113(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.log(wp.abs(val) + 1.0)
