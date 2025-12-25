import warp as wp

@wp.kernel
def conditional_0019(data: wp.array(dtype=float),
           threshold: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    
    if val < threshold:
        output[i] = val * 2.0
    elif val < threshold * 0.5:
        output[i] = threshold
    else:
        output[i] = val * 0.5
