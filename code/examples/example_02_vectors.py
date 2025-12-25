"""
Example 2: Vector Operations
Demonstrate vector operations and built-in functions.
"""

import warp as wp
import numpy as np

wp.init()

@wp.kernel
def vector_operations(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    new_positions: wp.array(dtype=wp.vec3)):
    """Update positions using velocity and timestep."""
    i = wp.tid()
    
    # Vector addition
    pos = positions[i]
    vel = velocities[i]
    
    # Integrate position
    new_pos = pos + vel * dt
    
    # Compute distance from origin
    dist = wp.length(new_pos)
    
    # Normalize if too far from origin (keep within radius 10)
    if dist > 10.0:
        new_pos = wp.normalize(new_pos) * 10.0
    
    new_positions[i] = new_pos


def main():
    n = 5
    
    # Create random positions and velocities
    np.random.seed(42)
    pos_data = np.random.randn(n, 3).astype(np.float32)
    vel_data = np.random.randn(n, 3).astype(np.float32) * 0.1
    
    positions = wp.array(pos_data, dtype=wp.vec3)
    velocities = wp.array(vel_data, dtype=wp.vec3)
    new_positions = wp.zeros(n, dtype=wp.vec3)
    
    print("Initial positions:")
    print(positions.numpy())
    print("\nVelocities:")
    print(velocities.numpy())
    
    # Launch kernel
    wp.launch(kernel=vector_operations, dim=n, 
             inputs=[positions, velocities, 0.1, new_positions])
    
    wp.synchronize()
    
    print("\nNew positions:")
    print(new_positions.numpy())
    print("✓ Test completed!")
    
    return True


if __name__ == "__main__":
    main()
