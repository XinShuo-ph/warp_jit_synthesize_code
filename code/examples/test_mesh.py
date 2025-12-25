#!/usr/bin/env python3
"""Test mesh operations with warp"""

import warp as wp
import numpy as np

wp.init()

@wp.kernel
def compute_face_normals(vertices: wp.array(dtype=wp.vec3),
                          indices: wp.array(dtype=int),
                          normals: wp.array(dtype=wp.vec3)):
    """Compute face normals for triangular mesh."""
    face_id = wp.tid()
    
    # Get triangle vertices
    i0 = indices[face_id * 3 + 0]
    i1 = indices[face_id * 3 + 1]
    i2 = indices[face_id * 3 + 2]
    
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    
    # Compute edges
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Compute normal (cross product)
    normal = wp.cross(e1, e2)
    normal = wp.normalize(normal)
    
    normals[face_id] = normal

def main():
    print("Running mesh normals example...")
    
    # Create a simple tetrahedron
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=np.float32)
    
    # Define faces (4 triangular faces)
    indices = np.array([
        0, 1, 2,  # bottom
        0, 1, 3,  # front
        1, 2, 3,  # right
        2, 0, 3   # left
    ], dtype=np.int32)
    
    num_faces = len(indices) // 3
    
    vertices_wp = wp.array(vertices, dtype=wp.vec3)
    indices_wp = wp.array(indices, dtype=int)
    normals_wp = wp.zeros(num_faces, dtype=wp.vec3)
    
    # Compute normals
    wp.launch(kernel=compute_face_normals, dim=num_faces,
              inputs=[vertices_wp, indices_wp, normals_wp])
    wp.synchronize()
    
    normals = normals_wp.numpy()
    print(f"Computed normals for {num_faces} faces:")
    for i, normal in enumerate(normals):
        print(f"  Face {i}: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    
    # Verify normals are unit length
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0), "Normals are not unit length!"
    
    return normals

if __name__ == "__main__":
    print("=== Run 1 ===")
    result1 = main()
    print("\n=== Run 2 ===")
    result2 = main()
    
    assert np.allclose(result1, result2), "Results don't match!"
    print("\n✓ Both runs produced identical results")
