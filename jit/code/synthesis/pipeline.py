"""Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
import json
import os
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import List, Optional, Any
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
    function_name: str
    python_source: str
    hlo_code: str
    optimized_hlo: Optional[str]
    generator_type: str


def create_sample_inputs(func_source: str) -> tuple:
    """Create sample inputs based on function signature analysis."""
    # Simple heuristic: count number of parameters
    import re
    
    # Find function definition
    match = re.search(r'def \w+\((.*?)\):', func_source)
    if not match:
        return (jnp.array([1.0, 2.0, 3.0]),)
    
    params_str = match.group(1)
    # Remove type hints and whitespace
    params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]
    
    # Create sample inputs based on number of parameters
    n_params = len(params)
    
    if n_params == 0:
        return ()
    elif n_params == 1:
        # Single array parameter
        return (jnp.array([1.0, 2.0, 3.0, 4.0]),)
    elif n_params == 2:
        # Check if first param looks like a scalar
        if 'alpha' in params[0] or 'scale' in params[0]:
            return (2.5, jnp.array([1.0, 2.0, 3.0]))
        else:
            return (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
    elif n_params == 3:
        # Scalar + two arrays or three arrays
        if 'alpha' in params[0] or 'scale' in params[0]:
            return (2.5, jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        else:
            return (jnp.array([1.0, 2.0, 3.0]), 
                    jnp.array([4.0, 5.0, 6.0]),
                    jnp.array([7.0, 8.0, 9.0]))
    else:
        # Default: multiple arrays
        return tuple(jnp.array([float(i), float(i+1), float(i+2)]) 
                    for i in range(n_params))


def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the function
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import jax\n")
        f.write("import jax.numpy as jnp\n")
        f.write("import jax.lax\n\n")
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


def extract_ir_from_source(func_source: str, func_name: str, module_id: int, backend: str = "cpu"):
    """Extract HLO IR from function source code."""
    from ir_extractor import extract_ir
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(func_source, module_name)
        
        # Find the function in the module
        func = None
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_') and name != 'jax' and name != 'jnp':
                func = obj
                break
        
        if func is None:
            raise ValueError(f"No function found in generated source")
        
        # Create sample inputs
        sample_inputs = create_sample_inputs(func_source)
        
        # Extract IR with the sample inputs
        ir = extract_ir(func, *sample_inputs, enable_backward=True)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.hlo_text, ir.optimized_hlo
        
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
    output_dir: Optional[str] = None,
    backend: str = "cpu"
) -> List[TrainingPair]:
    """Generate training pairs with HLO code."""
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate function
        import random
        random.seed(seed)
        generator = random.choice(GENERATORS)
        func_source = generator(seed)
        
        # Extract function name from source
        import re
        match = re.search(r'def (\w+)\(', func_source)
        func_name = match.group(1) if match else f"function_{i}"
        
        try:
            hlo_code, optimized_hlo = extract_ir_from_source(func_source, func_name, i, backend)
            
            pair = TrainingPair(
                id=i,
                function_name=func_name,
                python_source=func_source.strip(),
                hlo_code=hlo_code,
                optimized_hlo=optimized_hlo,
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
    backend: str = "cpu",
    include_optimized: bool = True
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        backend: "cpu" or "gpu"
        include_optimized: Include optimized HLO in output
    """
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(GENERATORS)
            func_source = generator(seed)
            
            import re
            match = re.search(r'def (\w+)\(', func_source)
            func_name = match.group(1) if match else f"function_{i}"
            
            try:
                hlo_code, optimized_hlo = extract_ir_from_source(func_source, func_name, i, backend)
                
                pair = {
                    "id": i,
                    "function_name": func_name,
                    "python": func_source.strip(),
                    "hlo": hlo_code,
                    "type": generator.__name__,
                    "backend": backend
                }
                
                if include_optimized and optimized_hlo:
                    pair["optimized_hlo"] = optimized_hlo
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only print first few failures
                    print(f"Failed to generate pair {i}: {e}")
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
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Target backend for compilation")
    parser.add_argument("--include-optimized", action="store_true", 
                        help="Include optimized HLO in output")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.backend, args.include_optimized)
    elif args.output:
        generate_training_pairs(args.count, args.seed, args.output, args.backend)
    else:
        # Demo mode
        pairs = generate_training_pairs(args.count, args.seed, backend=args.backend)
        
        print("\n=== Sample Pairs ===")
        for pair in pairs[:3]:
            print(f"\n--- {pair.function_name} ({pair.generator_type}) ---")
            print("Python:")
            print(pair.python_source)
            print(f"\nHLO ({len(pair.hlo_code)} chars):")
            print(pair.hlo_code[:500] + "...")
            if pair.optimized_hlo:
                print(f"\nOptimized HLO ({len(pair.optimized_hlo)} chars):")
                print(pair.optimized_hlo[:500] + "...")
