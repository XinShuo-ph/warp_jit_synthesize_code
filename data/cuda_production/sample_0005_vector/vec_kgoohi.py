@wp.kernel
def vec_kgoohi(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = wp.cross(a[tid], b[tid])
