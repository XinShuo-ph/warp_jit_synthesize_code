import os
import sys
import importlib.util
import json
import hashlib
import warp as wp

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from jit.code.synthesis.generator import KernelGenerator
from jit.code.extraction.ir_extractor import get_kernel_ir

class SynthesisPipeline:
    def __init__(self, output_dir="jit/data/samples", temp_dir="jit/code/synthesis/temp_modules"):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.generator = KernelGenerator()
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ensure temp dir is an init package
        with open(os.path.join(self.temp_dir, "__init__.py"), "w") as f:
            f.write("")

    def generate_pair(self, idx):
        """
        Generates a kernel, compiles it, extracts IR, and saves the pair.
        """
        kernel_name = f"kernel_{idx}"
        
        # 1. Generate Code
        code = self.generator.generate_kernel_code(kernel_name)
        
        # 2. Save to temp file
        module_name = f"mod_{idx}_{hashlib.md5(code.encode()).hexdigest()[:8]}"
        file_path = os.path.join(self.temp_dir, f"{module_name}.py")
        
        with open(file_path, "w") as f:
            f.write("import warp as wp\n\n")
            f.write(code)
            
        # 3. Load module
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            kernel = getattr(module, kernel_name)
            
            # 4. Extract IR
            # Extract CUDA IR if possible, else CPU (or both)
            # For training data, we probably want CUDA IR as it's more structured/complex
            # But in this CPU env, we might only get CPU IR or partial CUDA IR
            # Let's try CUDA first
            ir = get_kernel_ir(kernel, device="cuda")
            
            if "Error" in ir:
                # Fallback or log error
                print(f"Warning for {kernel_name}: {ir}")
                # Try CPU
                ir_cpu = get_kernel_ir(kernel, device="cpu")
                if "Error" not in ir_cpu:
                    ir = ir_cpu
            
            return {
                "id": idx,
                "kernel_name": kernel_name,
                "python_code": code,
                "ir_code": ir
            }
            
        except Exception as e:
            print(f"Error processing {kernel_name}: {e}")
            return None
        finally:
            # Cleanup temp file? Maybe keep for debugging for now
            pass

    def run_batch(self, count=10):
        results = []
        for i in range(count):
            res = self.generate_pair(i)
            if res:
                results.append(res)
                
                # Save individual sample
                sample_path = os.path.join(self.output_dir, f"sample_{i}.json")
                with open(sample_path, "w") as f:
                    json.dump(res, f, indent=2)
                    
        print(f"Generated {len(results)}/{count} samples.")

if __name__ == "__main__":
    wp.init()
    pipeline = SynthesisPipeline()
    pipeline.run_batch(count=100)
