import warp as wp
import numpy as np

wp.init()

@wp.struct
class Particle:
    pos: wp.vec3
    mass: float

@wp.kernel
def center_of_mass(particles: wp.array(dtype=Particle),
                   com: wp.array(dtype=wp.vec3)):
    # Simple reduction (not efficient for GPU but fine for example)
    # Note: Warp kernels are typically element-wise, but we can do a loop inside if small or atomic add
    
    # We will compute partial mass weighted position for each thread and atomic add to output
    tid = wp.tid()
    p = particles[tid]
    
    weighted_pos = p.pos * p.mass
    wp.atomic_add(com, 0, weighted_pos)

def run():
    n = 10
    
    # Create struct array
    p_data = wp.zeros(n, dtype=Particle)
    
    # Initialize data on host then copy (simulated by launching init kernel or manipulating numpy views if supported)
    # Warp struct arrays are tricky to init from numpy directly if not SoA/AoS managed properly
    # Easier to use a kernel to init
    
    @wp.kernel
    def init_particles(particles: wp.array(dtype=Particle)):
        tid = wp.tid()
        particles[tid].pos = wp.vec3(float(tid), 0.0, 0.0)
        particles[tid].mass = 1.0

    wp.launch(init_particles, dim=n, inputs=[p_data])
    
    com = wp.zeros(1, dtype=wp.vec3)
    
    wp.launch(center_of_mass, dim=n, inputs=[p_data, com])
    
    com_res = com.numpy()[0]
    print(f"Total weighted sum: {com_res}")
    
    expected_x = sum([float(i) * 1.0 for i in range(n)])
    print(f"Expected x: {expected_x}")
    
    assert np.isclose(com_res[0], expected_x)
    print("Test passed!")

if __name__ == "__main__":
    run()
