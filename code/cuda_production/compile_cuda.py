"""
CUDA Compilation Pipeline.
Validates CUDA code generation and produces PTX assembly.
"""
import subprocess
import shutil
from pathlib import Path
import sys


def check_nvcc_available():
    """Check if nvcc is available."""
    return shutil.which('nvcc') is not None


def compile_cuda_to_ptx(cu_file, output_ptx=None, arch="sm_50"):
    """
    Compile CUDA .cu file to PTX assembly.
    
    Args:
        cu_file: Path to .cu file
        output_ptx: Output PTX file (default: same name as .cu)
        arch: CUDA architecture (default: sm_50 for compatibility)
    
    Returns:
        (success, ptx_content, error_message)
    """
    cu_path = Path(cu_file)
    
    if not cu_path.exists():
        return False, None, f"File not found: {cu_file}"
    
    if not check_nvcc_available():
        return False, None, "nvcc not available (no CUDA toolkit installed)"
    
    # Output PTX file
    if output_ptx is None:
        output_ptx = cu_path.with_suffix('.ptx')
    
    # Compile to PTX
    cmd = [
        'nvcc',
        f'-arch={arch}',
        '--ptx',
        str(cu_path),
        '-o', str(output_ptx)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Read PTX content
            with open(output_ptx, 'r') as f:
                ptx_content = f.read()
            return True, ptx_content, None
        else:
            return False, None, result.stderr
    
    except subprocess.TimeoutExpired:
        return False, None, "Compilation timeout"
    except Exception as e:
        return False, None, str(e)


def validate_cuda_syntax(cu_file):
    """
    Validate CUDA syntax without generating output.
    
    Args:
        cu_file: Path to .cu file
    
    Returns:
        (valid, error_message)
    """
    if not check_nvcc_available():
        return None, "nvcc not available (cannot validate syntax)"
    
    cu_path = Path(cu_file)
    
    if not cu_path.exists():
        return False, f"File not found: {cu_file}"
    
    # Compile with syntax check only
    cmd = [
        'nvcc',
        '-arch=sm_50',
        '--ptx',
        '-Xptxas', '-v',
        str(cu_path),
        '-o', '/dev/null'  # Discard output
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
    
    except Exception as e:
        return False, str(e)


def analyze_ptx(ptx_content):
    """
    Analyze PTX assembly and extract statistics.
    
    Returns:
        dict with statistics
    """
    lines = ptx_content.split('\n')
    
    stats = {
        'total_lines': len(lines),
        'instructions': 0,
        'registers_used': 0,
        'shared_memory': 0,
        'functions': []
    }
    
    for line in lines:
        line = line.strip()
        
        # Count instructions (lines that don't start with . or //)
        if line and not line.startswith('.') and not line.startswith('//'):
            stats['instructions'] += 1
        
        # Extract register usage
        if '.reg ' in line:
            stats['registers_used'] += 1
        
        # Extract function names
        if '.entry' in line or '.func' in line:
            match = line.split()
            if len(match) > 1:
                stats['functions'].append(match[1])
    
    return stats


def batch_compile_to_ptx(cu_files, output_dir=None):
    """
    Compile multiple CUDA files to PTX.
    
    Args:
        cu_files: List of .cu file paths
        output_dir: Optional output directory for PTX files
    
    Returns:
        List of (filename, success, error) tuples
    """
    results = []
    
    for cu_file in cu_files:
        cu_path = Path(cu_file)
        
        if output_dir:
            output_ptx = Path(output_dir) / cu_path.with_suffix('.ptx').name
        else:
            output_ptx = None
        
        success, ptx, error = compile_cuda_to_ptx(cu_path, output_ptx)
        results.append((cu_path.name, success, error))
    
    return results


if __name__ == "__main__":
    # Check nvcc availability
    if check_nvcc_available():
        print("✓ nvcc found")
        
        # Get version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        print(result.stdout)
    else:
        print("✗ nvcc not found")
        print("  PTX compilation requires CUDA toolkit")
        print("  Code generation still works without nvcc")
