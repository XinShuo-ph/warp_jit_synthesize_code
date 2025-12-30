"""Synthesis Pipeline - Generates Python→HLO/jaxpr pairs for LLM training."""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import jax
import jax.numpy as jnp
from generator import generate_function_batch, generate_random_function, GENERATORS


@dataclass
class TrainingPair:
    """A Python→IR training pair."""
    id: int
    function_name: str
    python_source: str
    jaxpr: str
    hlo: str
    stablehlo: Optional[str]
    generator_type: str


def extract_ir_from_spec(func_spec, enable_backward: bool = True):
    """Extract IR from a FunctionSpec."""
    from ir_extractor import extract_ir
    
    ir = extract_ir(
        func_spec.func,
        func_spec.sample_inputs,
        enable_backward=enable_backward
    )
    
    return ir.jaxpr, ir.hlo, ir.stablehlo


def generate_training_pairs(
    count: int,
    base_seed: int = 42,
    output_dir: Optional[str] = None
) -> List[TrainingPair]:
    """Generate training pairs with jaxpr and HLO code."""
    import random
    
    pairs = []
    failed = 0
    
    for i in range(count):
        seed = base_seed + i
        
        # Pick a generator and generate function
        random.seed(seed)
        generator = random.choice(GENERATORS)
        
        try:
            func_spec = generator(seed)
            jaxpr, hlo, stablehlo = extract_ir_from_spec(func_spec)
            
            pair = TrainingPair(
                id=i,
                function_name=func_spec.name,
                python_source=func_spec.source.strip(),
                jaxpr=jaxpr,
                hlo=hlo,
                stablehlo=stablehlo,
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
    ir_type: str = "both"
):
    """Generate pairs and save as JSONL.
    
    Args:
        count: Number of pairs to generate
        output_path: Output file path
        base_seed: Random seed base
        ir_type: "jaxpr", "hlo", "stablehlo", or "both" (jaxpr + hlo)
    """
    import random
    
    with open(output_path, 'w') as f:
        generated = 0
        failed = 0
        
        for i in range(count):
            seed = base_seed + i
            
            random.seed(seed)
            generator = random.choice(GENERATORS)
            
            try:
                func_spec = generator(seed)
                jaxpr, hlo, stablehlo = extract_ir_from_spec(func_spec)
                
                pair = {
                    "id": i,
                    "function_name": func_spec.name,
                    "python": func_spec.source.strip(),
                    "type": generator.__name__
                }
                
                if ir_type == "jaxpr":
                    pair["jaxpr"] = jaxpr
                elif ir_type == "hlo":
                    pair["hlo"] = hlo
                elif ir_type == "stablehlo":
                    if stablehlo:
                        pair["stablehlo"] = stablehlo
                    else:
                        pair["hlo"] = hlo  # Fallback
                else:  # both
                    pair["jaxpr"] = jaxpr
                    pair["hlo"] = hlo
                    if stablehlo:
                        pair["stablehlo"] = stablehlo
                
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
    
    parser = argparse.ArgumentParser(description="Generate Python→HLO/jaxpr training pairs")
    parser.add_argument("--count", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL")
    parser.add_argument("--ir-type", type=str, default="both", 
                        choices=["jaxpr", "hlo", "stablehlo", "both"],
                        help="IR type(s) to include")
    
    args = parser.parse_args()
    
    if args.jsonl and args.output:
        generate_batch_to_jsonl(args.count, args.output, args.seed, args.ir_type)
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
            print(f"\nJaxpr ({len(pair.jaxpr)} chars):")
            print(pair.jaxpr[:500] + "..." if len(pair.jaxpr) > 500 else pair.jaxpr)
            print(f"\nHLO ({len(pair.hlo)} chars):")
            print(pair.hlo[:500] + "..." if len(pair.hlo) > 500 else pair.hlo)
