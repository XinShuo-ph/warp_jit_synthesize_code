import warp as wp

@wp.kernel
def conditional_0004(data: wp.array(dtype=float),
           threshold: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    
    if val > 1.0:
        output[i] = 1.0
    elif val < threshold * 0.5:
        output[i] = val
    else:
        output[i] = 0.0
