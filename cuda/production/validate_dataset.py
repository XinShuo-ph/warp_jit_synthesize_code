"""
Dataset Quality Validator

Validate the quality and correctness of generated CUDA IR dataset.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re


class DatasetValidator:
    """Validate CUDA IR dataset quality."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.issues = []
        self.warnings = []
        
    def validate(self) -> Tuple[bool, Dict]:
        """Run all validation checks."""
        print("=" * 70)
        print("Dataset Quality Validation")
        print("=" * 70)
        print(f"Dataset: {self.dataset_dir}")
        print()
        
        results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "issues": [],
            "warnings": []
        }
        
        # Check 1: Directory exists
        print("Check 1: Dataset directory exists")
        if not self.dataset_dir.exists():
            self.issues.append("Dataset directory does not exist")
            print("  ✗ FAIL: Directory not found")
            results["checks_failed"] += 1
            return False, results
        print("  ✓ PASS")
        results["checks_passed"] += 1
        
        # Check 2: Files present
        print("\nCheck 2: Dataset files present")
        json_files = list(self.dataset_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name not in ["generation_stats.json"]]
        
        if len(json_files) == 0:
            self.issues.append("No JSON files found in dataset")
            print("  ✗ FAIL: No data files")
            results["checks_failed"] += 1
            return False, results
        
        results["total_files"] = len(json_files)
        print(f"  ✓ PASS: Found {len(json_files)} files")
        results["checks_passed"] += 1
        
        # Check 3: File format validation
        print("\nCheck 3: File format validation")
        valid_count = 0
        invalid_count = 0
        
        for filepath in json_files[:10]:  # Sample first 10
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    
                # Check required fields
                required = ["python_source", "cuda_ir", "metadata"]
                if all(field in data for field in required):
                    valid_count += 1
                else:
                    invalid_count += 1
                    self.issues.append(f"Missing fields in {filepath.name}")
            except Exception as e:
                invalid_count += 1
                self.issues.append(f"Cannot parse {filepath.name}: {e}")
        
        if invalid_count > 0:
            print(f"  ✗ FAIL: {invalid_count}/{valid_count + invalid_count} files invalid")
            results["checks_failed"] += 1
        else:
            print(f"  ✓ PASS: All sampled files valid")
            results["checks_passed"] += 1
        
        results["valid_files"] = valid_count
        results["invalid_files"] = invalid_count
        
        # Check 4: CUDA patterns present
        print("\nCheck 4: CUDA patterns verification")
        cuda_patterns = ["blockIdx", "threadIdx", "blockDim", "gridDim"]
        files_with_cuda = 0
        
        for filepath in json_files[:20]:  # Sample 20 files
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    ir_code = data.get("cuda_ir", "")
                    
                    if any(pattern in ir_code for pattern in cuda_patterns):
                        files_with_cuda += 1
            except:
                pass
        
        cuda_rate = files_with_cuda / min(20, len(json_files))
        
        if cuda_rate < 0.95:
            print(f"  ✗ FAIL: Only {cuda_rate*100:.1f}% have CUDA patterns")
            results["checks_failed"] += 1
        else:
            print(f"  ✓ PASS: {cuda_rate*100:.1f}% have CUDA patterns")
            results["checks_passed"] += 1
        
        # Check 5: Category distribution
        print("\nCheck 5: Category distribution")
        categories = Counter()
        
        for filepath in json_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    cat = data["metadata"]["category"]
                    categories[cat] += 1
            except:
                pass
        
        print(f"  Categories found: {len(categories)}")
        for cat, count in sorted(categories.items()):
            pct = count / len(json_files) * 100
            print(f"    {cat:15s}: {count:4d} ({pct:5.1f}%)")
        
        # Check if balanced (within 5% variance)
        if len(categories) > 0:
            expected_pct = 100.0 / len(categories)
            max_deviation = max(abs(count/len(json_files)*100 - expected_pct) 
                              for count in categories.values())
            
            if max_deviation > 10:
                self.warnings.append(f"Category imbalance: {max_deviation:.1f}% deviation")
                print(f"  ⚠ WARNING: {max_deviation:.1f}% deviation from balanced")
            else:
                print(f"  ✓ PASS: Balanced distribution ({max_deviation:.1f}% deviation)")
            results["checks_passed"] += 1
        
        # Check 6: Duplicate detection
        print("\nCheck 6: Duplicate detection")
        seen_sources = set()
        duplicates = 0
        
        for filepath in json_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    source = data["python_source"]
                    
                    if source in seen_sources:
                        duplicates += 1
                    else:
                        seen_sources.add(source)
            except:
                pass
        
        if duplicates > len(json_files) * 0.01:  # >1% duplicates
            print(f"  ✗ FAIL: {duplicates} duplicates found ({duplicates/len(json_files)*100:.1f}%)")
            results["checks_failed"] += 1
        else:
            print(f"  ✓ PASS: {duplicates} duplicates ({duplicates/len(json_files)*100:.2f}%)")
            results["checks_passed"] += 1
        
        # Check 7: IR code quality
        print("\nCheck 7: IR code quality")
        empty_ir = 0
        short_ir = 0
        
        for filepath in json_files[:50]:  # Sample 50
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    ir = data["cuda_ir"]
                    
                    if not ir or len(ir.strip()) == 0:
                        empty_ir += 1
                    elif len(ir) < 100:
                        short_ir += 1
            except:
                pass
        
        if empty_ir > 0:
            print(f"  ✗ FAIL: {empty_ir} empty IR codes")
            results["checks_failed"] += 1
        else:
            print(f"  ✓ PASS: No empty IR codes")
            results["checks_passed"] += 1
        
        if short_ir > 5:
            self.warnings.append(f"{short_ir} IR codes are suspiciously short")
            print(f"  ⚠ WARNING: {short_ir} short IR codes")
        
        # Summary
        print("\n" + "=" * 70)
        print("Validation Summary")
        print("=" * 70)
        print(f"Total files: {results['total_files']}")
        print(f"Checks passed: {results['checks_passed']}")
        print(f"Checks failed: {results['checks_failed']}")
        print(f"Issues: {len(self.issues)}")
        print(f"Warnings: {len(self.warnings)}")
        print()
        
        if self.issues:
            print("Issues found:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()
        
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()
        
        results["issues"] = self.issues
        results["warnings"] = self.warnings
        
        overall_pass = results["checks_failed"] == 0
        
        if overall_pass:
            print("✓ Overall: PASS")
            print("Dataset quality is good!")
        else:
            print("✗ Overall: FAIL")
            print("Dataset has quality issues")
        
        return overall_pass, results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CUDA IR dataset quality")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="/workspace/cuda/data/cuda_production",
        help="Dataset directory to validate"
    )
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset_dir)
    passed, results = validator.validate()
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
