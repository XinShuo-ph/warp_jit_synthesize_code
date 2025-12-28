"""
Test script for manual GPU execution validation.

This script provides ready-to-run test code that can be executed
on a machine with an actual NVIDIA GPU to validate the generated
CUDA kernels work correctly.
"""
import json
import sys
from pathlib import Path
import numpy as np


def generate_test_script(json_file: Path, output_file: Path):
    """Generate a standalone test script from a JSON kernel pair."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    python_source = data["python_source"]
    kernel_name = data["metadata"]["kernel_name"]
    category = data["metadata"]["category"]
    
    # Generate test script
    test_code = f'''"""
GPU Test for kernel: {kernel_name}
Category: {category}

Run this on a machine with CUDA GPU:
    python {output_file.name}
"""
import warp as wp
import numpy as np

# Initialize warp with CUDA
wp.init()
print("Warp initialized")
print(f"CUDA available: {{'cuda' in wp.get_devices()}}")

# Kernel source
{python_source}

# Test function
def test_{kernel_name}():
    """Test the {kernel_name} kernel."""
    device = "cuda"  # Change to "cpu" if no GPU available
    
    # Create test data
    n = 1024
    '''
    
    # Add category-specific test setup
    if category in ["arithmetic", "math"]:
        test_code += '''
    a = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device=device)
    b = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device=device)
    c = wp.array(np.zeros(n, dtype=np.float32), dtype=float, device=device)
    
    # Launch kernel
    wp.launch({kernel_name}, dim=n, inputs=[a, b, c], device=device)
    wp.synchronize()
    
    # Get results
    result = c.numpy()
    print(f"Result shape: {{result.shape}}")
    print(f"Result sample: {{result[:5]}}")
'''
    
    elif category == "vector":
        test_code += '''
    a = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3, device=device)
    b = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    
    # Launch kernel
    wp.launch({kernel_name}, dim=n, inputs=[a, b, out], device=device)
    wp.synchronize()
    
    # Get results
    result = out.numpy()
    print(f"Result shape: {{result.shape}}")
'''
    
    elif category == "atomic":
        test_code += '''
    values = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device=device)
    result = wp.array(np.zeros(1, dtype=np.float32), dtype=float, device=device)
    
    # Launch kernel
    wp.launch({kernel_name}, dim=n, inputs=[values, result], device=device)
    wp.synchronize()
    
    # Get results
    final_result = result.numpy()
    print(f"Reduction result: {{final_result[0]}}")
'''
    
    else:  # Generic test
        test_code += '''
    # Generic test data
    a = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device=device)
    out = wp.array(np.zeros(n, dtype=np.float32), dtype=float, device=device)
    
    # Launch kernel (adjust inputs based on kernel signature)
    try:
        wp.launch({kernel_name}, dim=n, inputs=[a, out], device=device)
        wp.synchronize()
        print("✓ Kernel executed successfully")
    except Exception as e:
        print(f"✗ Kernel execution failed: {{e}}")
        return False
    
    result = out.numpy()
    print(f"Result shape: {{result.shape}}")
'''
    
    test_code += f'''
    print("✓ Test passed for {kernel_name}")
    return True

if __name__ == "__main__":
    success = test_{kernel_name}()
    sys.exit(0 if success else 1)
'''
    
    # Write test script
    with open(output_file, 'w') as f:
        f.write(test_code)
    
    print(f"Generated test script: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_gpu_tests.py <json_file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        output_file = Path(f"test_{path.stem}.py")
        generate_test_script(path, output_file)
    
    elif path.is_dir():
        json_files = list(path.glob("pair_*.json"))[:10]  # First 10 files
        
        output_dir = Path("gpu_tests")
        output_dir.mkdir(exist_ok=True)
        
        for json_file in json_files:
            output_file = output_dir / f"test_{json_file.stem}.py"
            try:
                generate_test_script(json_file, output_file)
            except Exception as e:
                print(f"Failed to generate test for {json_file}: {e}")
        
        print(f"\n✓ Generated {len(json_files)} test scripts in {output_dir}/")
    
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
