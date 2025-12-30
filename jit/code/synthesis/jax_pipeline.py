"""JAX Synthesis Pipeline - Generates Python→HLO/LLVM/PTX pairs."""
import json
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import jax

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from jax_generator import generate_kernel_batch, GENERATORS
from jax_ir_extractor import extract_ir, ExtractedIR

@dataclass
class TrainingPair:
    """A Python→IR training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_text: str
    llvm_ir: Optional[str]
    ptx_code: Optional[str]
    generator_type: str

def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n\n")
        f.write(source)
        temp_path = f.name
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int):
    """Extract IR from kernel source code."""
    module_name = f"synth_jax_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the function in the module
        func = getattr(module, kernel_name, None)
        
        if func is None:
            # Fallback: look for any jitted function
            for name in dir(module):
                obj = getattr(module, name)
                # primitive check for jitted function
                if hasattr(obj, 'lower'): 
                    func = obj
                    kernel_name = name
                    break
        
        if func is None:
             # Try to find the function by name even if not jitted (though generator adds @jax.jit)
            match = [n for n in dir(module) if not n.startswith('_') and n != 'jax' and n != 'jnp']
            if match:
                # Filter for functions
                import inspect
                funcs = [getattr(module, n) for n in match if inspect.isfunction(getattr(module, n)) or hasattr(getattr(module, n), 'lower')]
                if funcs:
                    func = funcs[0]
                    kernel_name = func.__name__

        if func is None:
            raise ValueError(f"No kernel found in generated source")
        
        # Extract IR
        ir = extract_ir(func, kernel_source)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e

def generate_training_pairs(
    count: int,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> List[TrainingPair]:
    """Generate training pairs."""
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        import random
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source = generator(seed)
        
        import re
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{i}"
        
        try:
            ir = extract_ir_from_source(kernel_source, kernel_name, i)
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_text=ir.hlo_text,
                llvm_ir=ir.llvm_ir,
                ptx_code=ir.ptx_code,
                generator_type=generator.__name__
            )
            pairs.append(pair)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} pairs ({failed} failed)")
                
        except Exception as e:
            failed += 1
            print(f"Failed to generate pair {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal: {len(pairs)} pairs generated, {failed} failed")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "jax_training_pairs.json"
        
        with open(output_path, 'w') as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        
        print(f"Saved to {output_path}")
    
    return pairs

def generate_batch_to_jsonl(
    count: int,
    output_path: str,
    base_seed: int = 42
):
    """Generate pairs and save as JSONL."""
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(GENERATORS)
            kernel_source = generator(seed)
            
            import re
            match = re.search(r'def (\w+)\(', kernel_source)
            kernel_name = match.group(1) if match else f"kernel_{i}"
            
            try:
                ir = extract_ir_from_source(kernel_source, kernel_name, i)
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "type": generator.__name__,
                    "hlo": ir.hlo_text,
                    "llvm": ir.llvm_ir,
                    "ptx": ir.ptx_code
                }
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                print(f"Failed pair {i}: {e}")
                continue
    
    print(f"\nTotal: {generated} pairs saved to {output_path}, {failed} failed")
    return generated

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→JAX IR training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--device", type=str, default="both", help="Ignored in JAX implementation (uses available backend)")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed)
    elif args.output:
        generate_training_pairs(args.count, args.seed, args.output)
    else:
        # Demo mode
        pairs = generate_training_pairs(args.count, args.seed)
        
        print("\n=== Sample Pairs ===")
        for pair in pairs[:3]:
            print(f"\n--- {pair.kernel_name} ({pair.generator_type}) ---")
            print("Python:")
            print(pair.python_source)
            print(f"\nHLO ({len(pair.hlo_text)} chars):")
            print(pair.hlo_text[:500] + "...")
            if pair.llvm_ir:
                print(f"\nLLVM/ASM ({len(pair.llvm_ir)} chars):")
                print(pair.llvm_ir[:500] + "...")
            if pair.ptx_code:
                print(f"\nPTX ({len(pair.ptx_code)} chars):")
                print(pair.ptx_code[:500] + "...")
