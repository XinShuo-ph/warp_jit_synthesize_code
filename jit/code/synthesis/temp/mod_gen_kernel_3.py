import warp as wp

@wp.kernel
def gen_kernel_3(data: wp.array(dtype=float), arg_0: float, arg_1: wp.array(dtype=float), arg_2: wp.array(dtype=float), arg_3: int):
    tid = wp.tid()
    t1 = float(-7.09)
    t2 = float(-4.20)
    arg_2[tid] = t2
    data[tid] = t1
    t5 = arg_1[tid]
