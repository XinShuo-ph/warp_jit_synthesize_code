import warp as wp

@wp.kernel
def vector_0005(pos: wp.array(dtype=wp.vec3), 
           vel: wp.array(dtype=wp.vec3),
           result: wp.array(dtype=wp.vec3),
           dt: float):
    i = wp.tid()
    result[i] = pos[i] + vel[i] * dt
