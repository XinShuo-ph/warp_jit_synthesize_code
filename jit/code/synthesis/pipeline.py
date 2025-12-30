"""Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
import json
import os
import sys
import tempfile
import importlib.util
import re
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import jax
import jax.numpy as jnp
from generator import generate_kernel_batch, GENERATORS


@dataclass
class TrainingPair:
    """A Python→HLO training pair."""
    id: int
    kernel_name: str
    python_source: str
    hlo_code: str
    optimized_hlo: Optional[str]
    mhlo_code: Optional[str]
    generator_type: str
    

def create_example_inputs(func_source: str, seed: int = 42):
    """Create example inputs for a JAX function based on its signature."""
    # Parse function signature to determine input types
    import re
    
    # Simple heuristic: count parameters
    match = re.search(r'def \w+\((.*?)\):', func_source)
    if not match:
        raise ValueError("Could not parse function signature")
    
    params = [p.strip() for p in match.group(1).split(',') if p.strip()]
    
    rng = np.random.RandomState(seed)
    
    # Create example inputs based on parameter count
    inputs = []
    for i, param in enumerate(params):
        # Check if parameter looks like a scalar (alpha, scale, n, etc.)
        if param in ['alpha', 'scale', 'n']:
            if param == 'n':
                inputs.append(5)  # Loop count
            else:
                inputs.append(rng.uniform(0.5, 2.0))  # Scalar
        else:
            # Create array input
            size = 8
            inputs.append(jnp.array(rng.randn(size).astype(np.float32)))
    
    return tuple(inputs)


def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n\n")
        f.write(source)
        temp_path = f.name
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, temp_path
    except Exception as e:
        os.unlink(temp_path)
        raise e


def extract_ir_from_source(kernel_source: str, kernel_name: str, module_id: int):
    """Extract HLO and MHLO IR from kernel source code."""
    from ir_extractor import extract_ir_with_grad
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        # Find the function in the module
        func = None
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_') and name == kernel_name:
                func = obj
                break
        
        if func is None:
            raise ValueError(f"No function '{kernel_name}' found in generated source")
        
        # Create example inputs
        example_inputs = create_example_inputs(kernel_source, seed=module_id)
        
        # Extract IR
        ir = extract_ir_with_grad(func, example_inputs, kernel_name)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.hlo_text, ir.optimized_hlo, ir.mhlo_text
        
    except Exception as e:
        # Cleanup on error
        if 'temp_path' in locals():
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
    """Generate training pairs with HLO code."""
    
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate kernel
        import random
        random.seed(seed)
        generator = random.choice(GENERATORS)
        kernel_source = generator(seed)
        
        # Extract kernel name from source
        match = re.search(r'def (\w+)\(', kernel_source)
        kernel_name = match.group(1) if match else f"kernel_{i}"
        
        try:
            hlo_code, optimized_hlo, mhlo_code = extract_ir_from_source(
                kernel_source, kernel_name, i
            )
            
            pair = TrainingPair(
                id=i,
                kernel_name=kernel_name,
                python_source=kernel_source.strip(),
                hlo_code=hlo_code,
                optimized_hlo=optimized_hlo,
                mhlo_code=mhlo_code,
                generator_type=generator.__name__
            )
            pairs.append(pair)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{count} pairs ({failed} failed)")
                
        except Exception as e:
            failed += 1
            print(f"Failed to generate pair {i}: {e}")
            continue
    
    print(f"\nTotal: {len(pairs)} pairs generated, {failed} failed")
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / "training_pairs.json"
        
        with open(output_path, 'w') as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        
        print(f"Saved to {output_path}")
    
    return pairs


def generate_batch_to_jsonl(
    count: int,
    output_path: str,
    base_seed: int = 42,
    include_mhlo: bool = True
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        include_mhlo: Whether to include MHLO representation
    """
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(GENERATORS)
            kernel_source = generator(seed)
            
            match = re.search(r'def (\w+)\(', kernel_source)
            kernel_name = match.group(1) if match else f"kernel_{i}"
            
            try:
                hlo_code, optimized_hlo, mhlo_code = extract_ir_from_source(
                    kernel_source, kernel_name, i
                )
                
                pair = {
                    "id": i,
                    "kernel_name": kernel_name,
                    "python": kernel_source.strip(),
                    "hlo": hlo_code,
                    "type": generator.__name__
                }
                
                if optimized_hlo:
                    pair["optimized_hlo"] = optimized_hlo
                
                if include_mhlo and mhlo_code:
                    pair["mhlo"] = mhlo_code
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                if failed % 10 == 0:
                    print(f"Failed {failed} pairs... (last error: {str(e)[:50]})")
                continue
    
    print(f"\nTotal: {generated} pairs saved to {output_path}, {failed} failed")
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→HLO training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--include-mhlo", action="store_true", default=True,
                        help="Include MHLO representation")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.include_mhlo)
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
            print(f"\nHLO ({len(pair.hlo_code)} chars):")
            print(pair.hlo_code[:500] + "...")
            if pair.optimized_hlo:
                print(f"\nOptimized HLO ({len(pair.optimized_hlo)} chars):")
                print(pair.optimized_hlo[:500] + "...")
