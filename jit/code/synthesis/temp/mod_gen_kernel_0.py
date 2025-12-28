import warp as wp

@wp.kernel
def gen_kernel_0(data: wp.array(dtype=float), arg_0: int, arg_1: float, arg_2: float):
    tid = wp.tid()
    t0 = data[tid]
    data[tid] = t0
    data[tid] = t0
    t3 = float(tid)
    t4 = wp.max(tid, tid)
    t5 = t0 + t0
    t6 = t4 + tid
    t7 = t5 + t0
