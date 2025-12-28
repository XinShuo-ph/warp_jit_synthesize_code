import warp as wp
@wp.kernel
def test_vectors(positions: wp.array(dtype=wp.vec3),
                 velocities: wp.array(dtype=wp.vec3),
                 forces: wp.array(dtype=wp.vec3),
                 dt: float):
    """Vector math and built-in functions."""
    i = wp.tid()

    vel = velocities[i]
    force = forces[i]

    # Update velocity
    new_vel = vel + force * dt

    # Clamp velocity magnitude
    speed = wp.length(new_vel)
    if speed > 10.0:
        new_vel = wp.normalize(new_vel) * 10.0

    velocities[i] = new_vel
