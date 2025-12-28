"""
Quality Validator for CUDA IR Dataset

Validates generated CUDA IR pairs for:
- Syntax correctness
- Required CUDA patterns
- Duplicate detection
- Completeness checks
"""
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys


class QualityValidator:
    """Validate CUDA IR dataset quality."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.duplicates = []
        self.hash_map = {}
        
    def validate_pair(self, pair: dict, filename: str) -> Tuple[bool, List[str]]:
        """
        Validate a single CUDA IR pair.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required keys
        required_keys = ['python_source', 'cuda_forward', 'metadata']
        for key in required_keys:
            if key not in pair:
                issues.append(f"Missing required key: {key}")
        
        if issues:
            return False, issues
        
        # Validate Python source
        python_src = pair['python_source']
        if not python_src or len(python_src) < 20:
            issues.append("Python source too short or empty")
        if '@wp.kernel' not in python_src:
            issues.append("Missing @wp.kernel decorator")
        if 'def ' not in python_src:
            issues.append("Missing function definition")
        
        # Validate CUDA code
        cuda_code = pair['cuda_forward']
        if not cuda_code or len(cuda_code) < 100:
            issues.append("CUDA code too short or empty")
        
        # Check CUDA-specific patterns
        required_cuda_patterns = {
            'extern': 'Missing extern "C" declaration',
            '__global__': 'Missing __global__ decorator',
            'cuda_kernel_forward': 'Missing cuda_kernel_forward function',
            'blockDim': 'Missing grid-stride loop (blockDim)',
            'blockIdx': 'Missing grid-stride loop (blockIdx)',
            'threadIdx': 'Missing grid-stride loop (threadIdx)',
        }
        
        for pattern, error_msg in required_cuda_patterns.items():
            if pattern not in cuda_code:
                issues.append(error_msg)
        
        # Validate metadata
        metadata = pair['metadata']
        required_meta = ['kernel_name', 'category', 'device']
        for key in required_meta:
            if key not in metadata:
                issues.append(f"Missing metadata: {key}")
        
        if 'device' in metadata and metadata['device'] != 'cuda':
            issues.append(f"Wrong device: {metadata['device']} (expected cuda)")
        
        # Check for duplicates (by Python source hash)
        src_hash = hashlib.md5(python_src.encode()).hexdigest()
        if src_hash in self.hash_map:
            self.duplicates.append((filename, self.hash_map[src_hash]))
            issues.append(f"Duplicate of {self.hash_map[src_hash]}")
        else:
            self.hash_map[src_hash] = filename
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_dataset(self, data_dir: str) -> Dict:
        """
        Validate entire dataset.
        
        Returns:
            Dictionary with validation results
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return {
                'success': False,
                'error': f"Directory not found: {data_dir}"
            }
        
        # Find all JSON files
        json_files = list(data_path.rglob("*.json"))
        json_files = [f for f in json_files if 'generation_stats' not in f.name and 'progress' not in f.name]
        
        if not json_files:
            return {
                'success': False,
                'error': f"No JSON files found in {data_dir}"
            }
        
        # Validate each file
        results = {
            'total_files': len(json_files),
            'valid': 0,
            'invalid': 0,
            'errors_by_file': {},
            'category_counts': defaultdict(int),
            'duplicates': [],
            'error_types': defaultdict(int),
        }
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    pair = json.load(f)
                
                is_valid, issues = self.validate_pair(pair, str(json_file))
                
                if is_valid:
                    results['valid'] += 1
                    # Track category
                    if 'metadata' in pair and 'category' in pair['metadata']:
                        results['category_counts'][pair['metadata']['category']] += 1
                else:
                    results['invalid'] += 1
                    results['errors_by_file'][str(json_file)] = issues
                    # Count error types
                    for issue in issues:
                        results['error_types'][issue] += 1
                
            except json.JSONDecodeError as e:
                results['invalid'] += 1
                results['errors_by_file'][str(json_file)] = [f"JSON parse error: {e}"]
                results['error_types']['JSON parse error'] += 1
            except Exception as e:
                results['invalid'] += 1
                results['errors_by_file'][str(json_file)] = [f"Unexpected error: {e}"]
                results['error_types']['Unexpected error'] += 1
        
        results['duplicates'] = self.duplicates
        results['success'] = True
        results['valid_percentage'] = 100 * results['valid'] / results['total_files']
        
        return results


def print_validation_report(results: Dict):
    """Print formatted validation report."""
    print("=" * 70)
    print("CUDA IR Dataset Quality Validation Report")
    print("=" * 70)
    
    if not results['success']:
        print(f"\n✗ Validation failed: {results.get('error', 'Unknown error')}")
        return
    
    # Summary
    print(f"\nDataset Summary:")
    print(f"  Total files:     {results['total_files']}")
    print(f"  Valid:           {results['valid']} ({results['valid_percentage']:.1f}%)")
    print(f"  Invalid:         {results['invalid']}")
    print(f"  Duplicates:      {len(results['duplicates'])}")
    
    # Category distribution
    print(f"\nCategory Distribution:")
    for category, count in sorted(results['category_counts'].items()):
        pct = 100 * count / results['valid'] if results['valid'] > 0 else 0
        print(f"  {category:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Error summary
    if results['invalid'] > 0:
        print(f"\nError Types:")
        for error_type, count in sorted(results['error_types'].items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        print(f"\nFirst 10 Invalid Files:")
        for i, (filename, issues) in enumerate(list(results['errors_by_file'].items())[:10], 1):
            print(f"  {i}. {Path(filename).name}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"     - {issue}")
    
    # Duplicates
    if results['duplicates']:
        print(f"\nDuplicate Pairs (first 10):")
        for i, (file1, file2) in enumerate(results['duplicates'][:10], 1):
            print(f"  {i}. {Path(file1).name} == {Path(file2).name}")
    
    # Quality assessment
    print("\n" + "=" * 70)
    if results['valid_percentage'] >= 99:
        print("✓ EXCELLENT: Dataset quality exceeds 99%")
    elif results['valid_percentage'] >= 95:
        print("✓ GOOD: Dataset quality meets 95% threshold")
    elif results['valid_percentage'] >= 90:
        print("⚠ ACCEPTABLE: Dataset quality is 90%+, consider regeneration")
    else:
        print("✗ POOR: Dataset quality below 90%, regeneration recommended")
    print("=" * 70)


def main():
    """Main validation routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CUDA IR dataset quality")
    parser.add_argument('data_dir', help="Directory containing CUDA IR pairs")
    parser.add_argument('-o', '--output', help="Save report to JSON file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    # Run validation
    validator = QualityValidator()
    results = validator.validate_dataset(args.data_dir)
    
    # Print report
    print_validation_report(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.output}")
    
    # Exit code based on quality
    if results['success'] and results['valid_percentage'] >= 95:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
