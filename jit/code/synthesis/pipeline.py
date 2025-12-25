import sys
import os
import warp as wp
import uuid
import json
import importlib.util

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from jit.code.synthesis.generator import KernelGenerator
from jit.code.extraction.ir_extractor import extract_ir

def generate_pair(output_dir="jit/data/samples", temp_dir="jit/code/synthesis/temp"):
    gen = KernelGenerator()
    name = f"kernel_{uuid.uuid4().hex[:8]}"
    source = gen.generate_kernel(name=name)
    
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"{name}.py")
    
    with open(temp_file, 'w') as f:
        f.write(source)
        
    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(name, temp_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        
        # Get the kernel object
        kernel_func = getattr(module, name)
        
        # Extract IR
        data = extract_ir(kernel_func)
        
        # Add metadata
        data['id'] = name
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{name}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filename
        
    except Exception as e:
        print(f"Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup temp file? Maybe keep for debugging or if we want the source file reference to be valid?
        # If we delete it, inspect.getsource might fail if called later? 
        # But extract_ir calls it. So afterwards it should be fine.
        # But for now I'll keep it or delete it.
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    wp.init()
    filename = generate_pair()
    if filename:
        print(f"Generated sample: {filename}")
    else:
        print("Failed to generate sample")
