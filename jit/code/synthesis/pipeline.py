#!/usr/bin/env python3
"""
Synthesis Pipeline: End-to-end Python→IR training pair generation

Combines kernel generation with IR extraction to create training datasets.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../extraction'))

from generator import KernelGenerator
from ir_extractor import IRExtractor
import json
from typing import List, Dict, Any
import traceback


class SynthesisPipeline:
    """End-to-end pipeline for generating training pairs"""
    
    def __init__(self, ir_type: str = "both", seed: int = 42):
        """
        Initialize pipeline
        
        Args:
            ir_type: Type of IR to extract ("jaxpr", "stablehlo", or "both")
            seed: Random seed for kernel generation
        """
        self.generator = KernelGenerator(seed=seed)
        self.extractor = IRExtractor(ir_type=ir_type)
        self.successful = 0
        self.failed = 0
    
    def generate_pair(self, category: str = None) -> Dict[str, Any]:
        """
        Generate a single training pair
        
        Args:
            category: Optional category to generate from
        
        Returns:
            Training pair dictionary
        """
        # Generate kernel
        if category:
            generator_method = getattr(self.generator, f"generate_{category}")
            kernel = generator_method()
        else:
            kernel = self.generator.generate_random()
        
        # Create training pair
        pair = self.extractor.create_training_pair(
            python_code=kernel['code'],
            func=kernel['function'],
            example_inputs=kernel['example_inputs'],
            metadata={
                "category": kernel['category'],
                "operation": kernel['operation'],
                "description": kernel['description']
            }
        )
        
        return pair
    
    def generate_batch(self, n: int, category: str = None, 
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Generate batch of training pairs
        
        Args:
            n: Number of pairs to generate
            category: Optional category to focus on
            verbose: Print progress
        
        Returns:
            List of training pairs
        """
        pairs = []
        self.successful = 0
        self.failed = 0
        
        for i in range(n):
            try:
                pair = self.generate_pair(category=category)
                pairs.append(pair)
                self.successful += 1
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{n} pairs...")
            
            except Exception as e:
                self.failed += 1
                if verbose:
                    print(f"  Warning: Failed to generate pair {i+1}: {e}")
                # Continue to next iteration
        
        if verbose:
            print(f"  Success: {self.successful}, Failed: {self.failed}")
        
        return pairs
    
    def generate_diverse_batch(self, n_per_category: int = 10,
                               verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Generate diverse batch with examples from each category
        
        Args:
            n_per_category: Number of pairs per category
            verbose: Print progress
        
        Returns:
            List of training pairs
        """
        categories = [
            "arithmetic", "array_op", "math_func", "reduction",
            "linalg", "composite"
        ]
        
        all_pairs = []
        
        for cat in categories:
            if verbose:
                print(f"\nGenerating {cat} kernels...")
            
            pairs = self.generate_batch(n_per_category, category=cat, verbose=verbose)
            all_pairs.extend(pairs)
        
        return all_pairs
    
    def save_pairs(self, pairs: List[Dict[str, Any]], filepath: str):
        """Save training pairs to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(pairs, f, indent=2)
    
    def validate_pairs(self, pairs: List[Dict[str, Any]]) -> bool:
        """Validate that all pairs are properly formatted"""
        required_keys = ["python_code", "input_info"]
        
        for i, pair in enumerate(pairs):
            # Check required keys
            for key in required_keys:
                if key not in pair:
                    print(f"Pair {i} missing key: {key}")
                    return False
            
            # Check IR keys (at least one should be present)
            has_ir = "jaxpr" in pair or "stablehlo" in pair
            if not has_ir:
                print(f"Pair {i} has no IR data")
                return False
        
        return True


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("Synthesis Pipeline Test")
    print("=" * 70)
    
    pipeline = SynthesisPipeline(ir_type="both", seed=42)
    
    # Test 1: Generate single pair
    print("\n1. Generate single pair")
    print("-" * 70)
    pair = pipeline.generate_pair(category="arithmetic")
    print(f"   Generated pair with keys: {list(pair.keys())}")
    print(f"   Category: {pair['metadata']['category']}")
    print(f"   Operation: {pair['metadata']['operation']}")
    print(f"   Code: {pair['python_code']}")
    print(f"   Jaxpr length: {len(pair['jaxpr'])} chars")
    print(f"   StableHLO length: {len(pair['stablehlo'])} chars")
    
    # Test 2: Generate small batch
    print("\n2. Generate batch of 20 pairs")
    print("-" * 70)
    pairs = pipeline.generate_batch(20, verbose=True)
    print(f"   Generated {len(pairs)} pairs")
    
    # Test 3: Validate pairs
    print("\n3. Validate pairs")
    print("-" * 70)
    is_valid = pipeline.validate_pairs(pairs)
    print(f"   All pairs valid: {is_valid}")
    
    # Test 4: Category distribution
    print("\n4. Category distribution")
    print("-" * 70)
    category_counts = {}
    for pair in pairs:
        cat = pair['metadata']['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat:15s}: {count:3d}")
    
    # Test 5: Save and reload
    print("\n5. Save and reload test")
    print("-" * 70)
    test_path = "../../data/samples/pipeline_test.json"
    pipeline.save_pairs(pairs, test_path)
    print(f"   Saved to {test_path}")
    
    with open(test_path, 'r') as f:
        loaded = json.load(f)
    print(f"   Loaded {len(loaded)} pairs successfully")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Synthesis pipeline working!")
    print("=" * 70)
