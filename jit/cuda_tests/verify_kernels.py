import os
import sys
import json
import argparse
import numpy as np
import warp as wp
import importlib.util
from pathlib import Path

# Add synthesis directory to path to reuse compile_kernel_from_source
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))
# Add extraction directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "extraction"))

# We might not be able to import pipeline if dependencies are missing, so let's copy the compile function
# or try to import it.
try:
    from pipeline import compile_kernel_from_source
except ImportError:
    print("Could not import pipeline.compile_kernel_from_source. Defining it locally.")
    import hashlib
    import tempfile

    def kernel_source_hash(source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:8]

    def compile_kernel_from_source(source: str, kernel_name: str):
        module_source = f'''import warp as wp

{source}
'''
        source_hash = kernel_source_hash(source)
        temp_dir = Path(tempfile.gettempdir()) / "warp_verification"
        temp_dir.mkdir(exist_ok=True)
        
        module_name = f"verify_{kernel_name}_{source_hash}"
        temp_file = temp_dir / f"{module_name}.py"
        
        with open(temp_file, 'w') as f:
            f.write(module_source)
        
        spec = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise RuntimeError(f"Failed to load module: {e}")
        
        kernel = getattr(module, kernel_name, None)
        if kernel is None:
            del sys.modules[module_name]
            raise RuntimeError(f"Kernel {kernel_name} not found in module")
        
        return kernel

def create_dummy_data(arg_types, n, device):
    """Create dummy data for kernel arguments based on types."""
    args = []
    
    # Simple parsing of arg_types string
    # Expected format: "wp.array(dtype=float)", "float", "wp.vec3", etc.
    
    for name, type_str in arg_types.items():
        if "wp.array" in type_str:
            # Parse dtype
            if "dtype=float" in type_str:
                dtype = float
                np_dtype = np.float32
            elif "dtype=wp.vec2" in type_str:
                dtype = wp.vec2
                np_dtype = np.float32 # specialized handling needed for numpy
            elif "dtype=wp.vec3" in type_str:
                dtype = wp.vec3
                np_dtype = np.float32
            elif "dtype=wp.vec4" in type_str:
                dtype = wp.vec4
                np_dtype = np.float32
            elif "dtype=wp.mat22" in type_str:
                dtype = wp.mat22
                np_dtype = np.float32
            elif "dtype=wp.mat33" in type_str:
                dtype = wp.mat33
                np_dtype = np.float32
            elif "dtype=wp.mat44" in type_str:
                dtype = wp.mat44
                np_dtype = np.float32
            else:
                dtype = float # Default
                np_dtype = np.float32
            
            # Create array
            if dtype in [float, int]:
                arr_data = np.random.randn(n).astype(np_dtype)
                args.append(wp.array(arr_data, dtype=dtype, device=device))
            else:
                # Vector/Matrix types
                # Need to check shape
                if dtype == wp.vec2:
                    shape = (n, 2)
                elif dtype == wp.vec3:
                    shape = (n, 3)
                elif dtype == wp.vec4:
                    shape = (n, 4)
                elif dtype == wp.mat22:
                    shape = (n, 2, 2)
                elif dtype == wp.mat33:
                    shape = (n, 3, 3)
                elif dtype == wp.mat44:
                    shape = (n, 4, 4)
                
                arr_data = np.random.randn(*shape).astype(np_dtype)
                args.append(wp.array(arr_data, dtype=dtype, device=device))

        elif type_str == "float":
            args.append(1.5)
        elif type_str == "int":
            args.append(5)
        else:
            # Fallback
            args.append(1.0)
            
    return args

def verify_file(filepath: Path, device: str = "cuda"):
    print(f"Verifying {filepath.name}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    source = data["python_source"]
    metadata = data["metadata"]
    kernel_name = metadata["kernel_name"]
    arg_types = metadata["arg_types"]
    
    try:
        kernel = compile_kernel_from_source(source, kernel_name)
    except Exception as e:
        print(f"  ❌ Compilation failed: {e}")
        return False
        
    n = 1024
    try:
        args = create_dummy_data(arg_types, n, device)
    except Exception as e:
        print(f"  ❌ Data creation failed: {e}")
        return False
    
    try:
        wp.launch(kernel, dim=n, inputs=args, device=device)
        wp.synchronize()
        print("  ✅ Execution successful")
        return True
    except Exception as e:
        print(f"  ❌ Execution failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing JSON pairs")
    parser.add_argument("--device", default="cuda", help="Device to verify on")
    args = parser.parse_args()
    
    wp.init()
    
    data_dir = Path(args.data_dir)
    files = sorted(list(data_dir.glob("*.json")))
    
    if not files:
        print(f"No JSON files found in {data_dir}")
        return
    
    passed = 0
    failed = 0
    
    for f in files:
        if verify_file(f, args.device):
            passed += 1
        else:
            failed += 1
            
    print(f"\nSummary: {passed} passed, {failed} failed")

if __name__ == "__main__":
    main()
