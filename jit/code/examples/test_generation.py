import warp as wp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from jit.code.synthesis.generator import KernelGenerator

def test_generation():
    wp.init()
    gen = KernelGenerator(seed=42)
    
    for i in range(5):
        name = f"gen_kernel_{i}"
        code = gen.generate_kernel_code(name)
        print(f"--- Generated {name} ---")
        print(code)
        
        # dynamic execution
        local_scope = {}
        try:
            exec(code, globals(), local_scope)
            kernel = local_scope[name]
            print(f"Successfully compiled {name}: {kernel}")
            
            # Verify it is a warp kernel
            if isinstance(kernel, wp.Kernel):
                print("Confirmed object is a wp.Kernel")
            else:
                print("Error: Object is not a wp.Kernel")
                
        except Exception as e:
            print(f"Failed to compile {name}: {e}")
            raise e
        print("\n")

if __name__ == "__main__":
    test_generation()
