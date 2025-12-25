"""
Example 3: Functions and Control Flow
Demonstrate wp.func decorators and control flow structures.
"""

import warp as wp
import numpy as np

wp.init()

@wp.func
def compute_distance_squared(p1: wp.vec3, p2: wp.vec3) -> float:
    """Compute squared distance between two points."""
    diff = p1 - p2
    return wp.dot(diff, diff)


@wp.func
def apply_force(pos: wp.vec3, attractor: wp.vec3, strength: float) -> wp.vec3:
    """Compute force from an attractor point."""
    diff = attractor - pos
    dist_sq = wp.dot(diff, diff)
    
    # Avoid division by zero
    if dist_sq < 0.01:
        return wp.vec3(0.0, 0.0, 0.0)
    
    # Force proportional to 1/r^2
    force_mag = strength / dist_sq
    direction = wp.normalize(diff)
    
    return direction * force_mag


@wp.kernel
def n_body_forces(
    positions: wp.array(dtype=wp.vec3),
    attractors: wp.array(dtype=wp.vec3),
    n_attractors: int,
    forces: wp.array(dtype=wp.vec3)):
    """Compute forces on particles from multiple attractors."""
    i = wp.tid()
    
    pos = positions[i]
    total_force = wp.vec3(0.0, 0.0, 0.0)
    
    # Sum forces from all attractors
    for j in range(n_attractors):
        attractor_pos = attractors[j]
        force = apply_force(pos, attractor_pos, 1.0)
        total_force = total_force + force
    
    forces[i] = total_force


def main():
    n_particles = 8
    n_attractors = 3
    
    # Create particles and attractors
    np.random.seed(123)
    particles = np.random.randn(n_particles, 3).astype(np.float32) * 2.0
    attractors = np.random.randn(n_attractors, 3).astype(np.float32) * 5.0
    
    positions = wp.array(particles, dtype=wp.vec3)
    attractor_pos = wp.array(attractors, dtype=wp.vec3)
    forces = wp.zeros(n_particles, dtype=wp.vec3)
    
    print(f"Particles: {n_particles}")
    print(f"Attractors: {n_attractors}")
    print(f"\nParticle positions:\n{positions.numpy()}")
    print(f"\nAttractor positions:\n{attractor_pos.numpy()}")
    
    # Launch kernel
    wp.launch(kernel=n_body_forces, dim=n_particles,
             inputs=[positions, attractor_pos, n_attractors, forces])
    
    wp.synchronize()
    
    print(f"\nComputed forces:\n{forces.numpy()}")
    print("✓ Test completed!")
    
    return True


if __name__ == "__main__":
    main()
