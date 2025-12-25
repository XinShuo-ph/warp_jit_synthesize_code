import warp as wp

@wp.kernel
def conditional_0012(data: wp.array(dtype=float),
           threshold: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    
    if val < 0.0:
        output[i] = -val
    elif val < threshold * 0.5:
        output[i] = val * 2.0
    else:
        output[i] = val
