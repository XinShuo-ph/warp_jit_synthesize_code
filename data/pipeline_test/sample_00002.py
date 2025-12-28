import warp as wp

@wp.kernel
def reduction_0002(data: wp.array(dtype=float),
           result: wp.array(dtype=float),
           n: int):
    tid = wp.tid()
    
    # Local reduction (simplified for demonstration)
    local_result = float(0.0) if tid < n else float(0.0)
    
    for i in range(tid, n, 1):
        val = data[i]
        local_result = local_result + val * val
        break  # Simplified
    
    result[tid] = local_result
