"""Synthesis Pipeline - Generates Python→HLO pairs for LLM training."""
import json
import os
import sys
import tempfile
import importlib.util
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
    function_name: str
    python_source: str
    jaxpr_code: str
    hlo_code: Optional[str]
    generator_type: str
    

def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    # Create a temp file to store the function
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


def create_example_inputs(function_source: str):
    """Create example inputs based on function signature."""
    import re
    
    # Parse function signature to count arguments
    match = re.search(r'def \w+\((.*?)\):', function_source)
    if not match:
        return []
    
    args_str = match.group(1)
    if not args_str.strip():
        return []
    
    # Count arguments
    args = [a.strip() for a in args_str.split(',') if a.strip()]
    num_args = len(args)
    
    # Create appropriate inputs
    inputs = []
    for i in range(num_args):
        arg_name = args[i]
        # Check if it's a scalar or array based on naming convention
        if 'alpha' in arg_name or 'scale' in arg_name or arg_name in ['n']:
            # Scalar input
            inputs.append(2.0)
        else:
            # Array input
            inputs.append(jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    return inputs


def extract_ir_from_source(function_source: str, function_name: str, module_id: int):
    """Extract JAXPR and HLO IR from function source code."""
    from ir_extractor import extract_ir
    
    module_name = f"synth_module_{module_id}"
    
    try:
        module, temp_path = create_module_from_source(function_source, module_name)
        
        # Find the function in the module
        func = None
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_'):
                func = obj
                break
        
        if func is None:
            raise ValueError(f"No function found in generated source")
        
        # Create example inputs
        inputs = create_example_inputs(function_source)
        
        # Extract IR
        ir = extract_ir(func, *inputs)
        
        # Cleanup
        os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return ir.jaxpr_code, ir.hlo_code
        
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
    """Generate training pairs with both JAXPR and HLO code."""
    
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate function
        import random
        random.seed(seed)
        generator = random.choice(GENERATORS)
        function_source = generator(seed)
        
        # Extract function name from source
        import re
        match = re.search(r'def (\w+)\(', function_source)
        function_name = match.group(1) if match else f"function_{i}"
        
        try:
            jaxpr_code, hlo_code = extract_ir_from_source(function_source, function_name, i)
            
            pair = TrainingPair(
                id=i,
                function_name=function_name,
                python_source=function_source.strip(),
                jaxpr_code=jaxpr_code,
                hlo_code=hlo_code,
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
    device: str = "both"
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        device: "cpu", "gpu", or "both" (JAX handles device automatically)
    """
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            import random
            random.seed(seed)
            generator = random.choice(GENERATORS)
            function_source = generator(seed)
            
            import re
            match = re.search(r'def (\w+)\(', function_source)
            function_name = match.group(1) if match else f"function_{i}"
            
            try:
                jaxpr_code, hlo_code = extract_ir_from_source(function_source, function_name, i)
                
                pair = {
                    "id": i,
                    "function_name": function_name,
                    "python": function_source.strip(),
                    "type": generator.__name__
                }
                
                # For JAX, we include both JAXPR and HLO
                if device == "cpu" or device == "both":
                    pair["jaxpr"] = jaxpr_code
                    if hlo_code:
                        pair["hlo"] = hlo_code
                
                # Note: JAX automatically compiles for available hardware
                # GPU/CUDA code is handled transparently by XLA
                
                f.write(json.dumps(pair) + '\n')
                generated += 1
                
                if generated % 10 == 0:
                    print(f"Generated {generated} pairs...")
                    
            except Exception as e:
                failed += 1
                print(f"Failed {i}: {e}")
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
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "gpu", "both"],
                        help="Target device(s) for code generation")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.device)
    elif args.output:
        generate_training_pairs(args.count, args.seed, args.output)
    else:
        # Demo mode
        pairs = generate_training_pairs(args.count, args.seed)
        
        print("\n=== Sample Pairs ===")
        for pair in pairs[:3]:
            print(f"\n--- {pair.function_name} ({pair.generator_type}) ---")
            print("Python:")
            print(pair.python_source)
            print(f"\nJAXPR ({len(pair.jaxpr_code)} chars):")
            print(pair.jaxpr_code[:500] + "..." if len(pair.jaxpr_code) > 500 else pair.jaxpr_code)
            if pair.hlo_code:
                print(f"\nHLO ({len(pair.hlo_code)} chars):")
                print(pair.hlo_code[:500] + "..." if len(pair.hlo_code) > 500 else pair.hlo_code)
