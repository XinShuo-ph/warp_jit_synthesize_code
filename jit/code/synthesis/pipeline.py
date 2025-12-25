import warp as wp
import sys
import os
import json
import traceback
import importlib.util

# Add workspace to path
sys.path.append(os.getcwd())

from jit.code.synthesis.generator import KernelGenerator
from jit.code.extraction.ir_extractor import get_kernel_ir

class SynthesisPipeline:
    def __init__(self, output_dir="jit/data/samples", temp_dir="jit/code/synthesis/temp"):
        self.generator = KernelGenerator()
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ensure temp dir is importable
        abs_temp = os.path.abspath(temp_dir)
        if abs_temp not in sys.path:
            sys.path.append(abs_temp)
        
    def synthesize_pair(self, idx):
        kernel_name = f"gen_kernel_{idx}"
        module_name = f"mod_{kernel_name}"
        file_path = os.path.join(self.temp_dir, f"{module_name}.py")
        
        try:
            # 1. Generate Source
            # We need to include imports in the source file
            body_source = self.generator.generate_kernel(name=kernel_name)
            full_source = f"import warp as wp\n{body_source}"
            
            with open(file_path, "w") as f:
                f.write(full_source)
            
            # 2. Import Module
            # Invalidate caches to ensure we load the new file if we reused names (though we use unique names)
            importlib.invalidate_caches()
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not create spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            kernel = getattr(module, kernel_name)
            
            # 3. Extract IR
            ir_map = get_kernel_ir(kernel, device="cpu")
            
            # 4. Construct Sample
            sample = {
                "id": idx,
                "name": kernel_name,
                "python_source": full_source,
                "cpp_source_forward": ir_map["forward"],
                "cpp_source_backward": ir_map.get("backward", ""),
                "device": "cpu"
            }
            
            return sample
            
        except Exception as e:
            print(f"Error processing {kernel_name}: {e}")
            # traceback.print_exc()
            return None
        finally:
            # Cleanup source file if needed
            # os.remove(file_path)
            pass

    def run_batch(self, count=10, filename="dataset.jsonl"):
        results = []
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            for i in range(count):
                sample = self.synthesize_pair(i)
                if sample:
                    results.append(sample)
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                
                if (i+1) % 10 == 0:
                    print(f"Generated {i+1}/{count} samples...")
                    
        print(f"Saved {len(results)} samples to {filepath}")
        return len(results)

if __name__ == "__main__":
    wp.init()
    pipeline = SynthesisPipeline()
    pipeline.run_batch(count=5)
