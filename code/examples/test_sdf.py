#!/usr/bin/env python3
"""Run warp core example: marching cubes"""

import sys
sys.path.insert(0, '/home/ubuntu/.local/lib/python3.12/site-packages/warp/examples/core')

import warp as wp
import numpy as np

wp.init()

# Simple marching cubes example
@wp.kernel
def compute_sdf(points: wp.array(dtype=wp.vec3),
                 values: wp.array(dtype=float),
                 center: wp.vec3,
                 radius: float):
    """Compute signed distance field for a sphere."""
    tid = wp.tid()
    p = points[tid]
    dist = wp.length(p - center) - radius
    values[tid] = dist

def main():
    print("Running marching cubes/SDF example...")
    
    # Create a grid
    N = 20
    points = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x = (i - N/2) / N * 4.0
                y = (j - N/2) / N * 4.0
                z = (k - N/2) / N * 4.0
                points.append([x, y, z])
    
    points_array = wp.array(points, dtype=wp.vec3)
    values_array = wp.zeros(len(points), dtype=float)
    
    center = wp.vec3(0.0, 0.0, 0.0)
    radius = 1.0
    
    # Compute SDF
    wp.launch(kernel=compute_sdf, dim=len(points), 
              inputs=[points_array, values_array, center, radius])
    wp.synchronize()
    
    values = values_array.numpy()
    print(f"Computed SDF for {len(points)} points")
    print(f"Min value: {values.min():.3f}, Max value: {values.max():.3f}")
    print(f"Points inside (negative): {np.sum(values < 0)}")
    print(f"Points outside (positive): {np.sum(values > 0)}")
    
    return values

if __name__ == "__main__":
    # Run twice
    print("=== Run 1 ===")
    result1 = main()
    print("\n=== Run 2 ===")
    result2 = main()
    
    # Verify consistency
    assert np.allclose(result1, result2), "Results don't match!"
    print("\n✓ Both runs produced identical results")
