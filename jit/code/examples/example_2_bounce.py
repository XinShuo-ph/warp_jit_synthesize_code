import warp as wp
import numpy as np

wp.init()

@wp.kernel
def simulate(x: wp.array(dtype=wp.vec3),
             v: wp.array(dtype=wp.vec3),
             dt: float):
    tid = wp.tid()
    
    # Gravity
    g = wp.vec3(0.0, -9.8, 0.0)
    v[tid] = v[tid] + g * dt
    
    # Integration
    x[tid] = x[tid] + v[tid] * dt
    
    # Ground collision (y = 0)
    p = x[tid]
    if p[1] < 0.0:
        p[1] = 0.0
        x[tid] = p
        vel = v[tid]
        v[tid] = wp.vec3(vel[0], -vel[1] * 0.5, vel[2]) # Bounce with damping

def run():
    num_particles = 5
    steps = 100
    dt = 0.016
    
    x_np = np.zeros((num_particles, 3), dtype=np.float32)
    x_np[:, 1] = 10.0 # Start at height 10
    v_np = np.zeros((num_particles, 3), dtype=np.float32)
    
    x = wp.from_numpy(x_np, dtype=wp.vec3)
    v = wp.from_numpy(v_np, dtype=wp.vec3)
    
    print("Simulating...")
    for i in range(steps):
        wp.launch(kernel=simulate, dim=num_particles, inputs=[x, v, dt])
    
    final_x = x.numpy()
    print(f"Final positions (y should be close to ground or bounced):\n{final_x}")
    
    # Basic check: particles shouldn't be excessively below ground
    assert np.all(final_x[:, 1] >= 0.0)
    print("Test passed!")

if __name__ == "__main__":
    run()
