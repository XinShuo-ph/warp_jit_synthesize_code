"""
IR Extractor for Warp Kernels

This module provides utilities to extract intermediate representation (C++ code)
from Warp kernels for training data generation.
"""

import warp as wp
import os
import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np


class IRExtractor:
    """
    Extracts IR (C++ code) from compiled Warp kernels
    """
    
    def __init__(self):
        self.cache_dir = wp.config.kernel_cache_dir
    
    def extract_ir(self, kernel, force_compile: bool = True) -> Dict:
        """
        Extract IR from a warp kernel
        
        Args:
            kernel: A warp kernel object (decorated with @wp.kernel)
            force_compile: If True, ensures kernel is compiled before extraction
        
        Returns:
            Dictionary containing:
                - python_source: Original Python source code
                - cpp_code: Full generated C++ code
                - forward_function: Extracted forward function
                - backward_function: Extracted backward function
                - signature: Kernel signature
                - kernel_info: Metadata about the kernel
        """
        
        if force_compile:
            # Kernel must be compiled to generate IR
            # If not compiled, we need to launch it at least once
            if not self._is_kernel_compiled(kernel):
                raise RuntimeError(
                    "Kernel not yet compiled. Please launch the kernel at least once "
                    "before extracting IR, or provide dummy inputs to force compilation."
                )
        
        # Get kernel information
        kernel_info = self._get_kernel_info(kernel)
        
        # Find the generated C++ file
        cpp_file = self._find_cpp_file(kernel)
        if not cpp_file:
            raise RuntimeError(f"Could not find generated C++ file for kernel {kernel.key}")
        
        # Read C++ code
        with open(cpp_file, 'r') as f:
            cpp_code = f.read()
        
        # Extract Python source
        python_source = self._get_python_source(kernel)
        
        # Find the specific function hash for this kernel in the C++ code
        kernel_hash = self._find_kernel_hash_in_cpp(cpp_code, kernel.key)
        
        # Extract forward and backward functions using the specific hash
        forward_func = self._extract_function_by_hash(cpp_code, kernel_hash, 'cpu_kernel_forward')
        backward_func = self._extract_function_by_hash(cpp_code, kernel_hash, 'cpu_kernel_backward')
        
        # Load metadata if available
        meta_file = cpp_file.replace('.cpp', '.meta')
        metadata = None
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        
        return {
            'python_source': python_source,
            'cpp_code': cpp_code,
            'cpp_file': cpp_file,
            'forward_function': forward_func,
            'backward_function': backward_func,
            'signature': kernel.sig if hasattr(kernel, 'sig') else '',
            'kernel_info': kernel_info,
            'metadata': metadata,
            'kernel_hash': kernel_hash
        }
    
    def _is_kernel_compiled(self, kernel) -> bool:
        """Check if kernel has been compiled"""
        return self._find_cpp_file(kernel) is not None
    
    def _get_kernel_info(self, kernel) -> Dict:
        """Extract kernel metadata"""
        info = {
            'key': kernel.key,
            'module_name': kernel.module.name if hasattr(kernel, 'module') else 'unknown',
            'is_generic': kernel.is_generic if hasattr(kernel, 'is_generic') else False,
        }
        
        # Get argument information
        if hasattr(kernel, 'adj') and kernel.adj:
            args = []
            for arg in kernel.adj.args:
                args.append({
                    'label': arg.label,
                    'type': str(arg.type) if hasattr(arg, 'type') else 'unknown'
                })
            info['arguments'] = args
        
        return info
    
    def _get_python_source(self, kernel) -> str:
        """Get the original Python source code of the kernel"""
        if hasattr(kernel, 'adj') and kernel.adj and hasattr(kernel.adj, 'source'):
            return kernel.adj.source
        
        # Fallback: try to get source from the function object
        try:
            import inspect
            if hasattr(kernel, 'func'):
                return inspect.getsource(kernel.func)
        except:
            pass
        
        return "Source not available"
    
    def _find_cpp_file(self, kernel) -> Optional[str]:
        """Find the generated C++ file for this kernel"""
        kernel_key = kernel.key
        
        # Search through cache directory
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith('.cpp'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Check if this file contains our kernel
                            if kernel_key in content:
                                return filepath
                    except:
                        pass
        
        return None
    
    def _extract_function(self, cpp_code: str, function_suffix: str) -> Optional[str]:
        """
        Extract a specific function from C++ code
        
        Args:
            cpp_code: Full C++ source
            function_suffix: Suffix to identify function (e.g., 'cpu_kernel_forward')
        
        Returns:
            Extracted function code or None
        """
        lines = cpp_code.split('\n')
        
        # Find function definition - look for the function with this specific suffix
        in_function = False
        function_lines = []
        brace_count = 0
        start_collecting = False
        
        for i, line in enumerate(lines):
            # Look for function definition with the suffix
            if not in_function and function_suffix in line:
                # Check if 'void' is on this line or within previous 2 lines
                start_line = i
                found_void = False
                for j in range(max(0, i-2), i+1):
                    if 'void' in lines[j]:
                        start_line = j
                        found_void = True
                        break
                
                if found_void:
                    in_function = True
                    start_collecting = True
                    # Collect from the void line onwards
                    function_lines = []
                    for k in range(start_line, i+1):
                        function_lines.append(lines[k])
                        brace_count += lines[k].count('{') - lines[k].count('}')
                    continue
            
            if in_function:
                function_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # If we've closed all braces and we've opened at least one, we're done
                if brace_count == 0 and any('{' in l for l in function_lines):
                    break
        
        return '\n'.join(function_lines) if function_lines else None
    
    def _find_kernel_hash_in_cpp(self, cpp_code: str, kernel_key: str) -> Optional[str]:
        """Find the hash suffix used for a specific kernel in the C++ code"""
        # Look for pattern like "kernel_name_HASH"
        import re
        pattern = rf'{re.escape(kernel_key)}_([a-f0-9]+)'
        match = re.search(pattern, cpp_code)
        if match:
            return match.group(1)
        return None
    
    def _extract_function_by_hash(self, cpp_code: str, kernel_hash: str, function_type: str) -> Optional[str]:
        """
        Extract function by looking for specific hash
        
        Args:
            cpp_code: Full C++ source
            kernel_hash: The hash suffix for this specific kernel
            function_type: Type of function (e.g., 'cpu_kernel_forward')
        """
        if not kernel_hash:
            # Fallback to old method
            return self._extract_function(cpp_code, function_type)
        
        # Look for function with this specific hash
        function_pattern = f'_{kernel_hash}_{function_type}'
        return self._extract_function(cpp_code, function_pattern)
    
    def save_pair(self, python_source: str, ir_code: str, output_path: str):
        """
        Save a Python→IR pair to disk
        
        Args:
            python_source: Python kernel code
            ir_code: Generated IR (C++ code)
            output_path: Path to save the pair
        """
        pair_data = {
            'python': python_source,
            'ir': ir_code,
            'metadata': {
                'python_length': len(python_source),
                'ir_length': len(ir_code),
                'python_hash': hashlib.sha256(python_source.encode()).hexdigest()[:16],
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(pair_data, f, indent=2)
    
    def extract_and_save(self, kernel, output_path: str) -> Dict:
        """
        Extract IR and save Python→IR pair
        
        Returns:
            The extracted IR data
        """
        ir_data = self.extract_ir(kernel)
        
        # Save the pair
        self.save_pair(
            ir_data['python_source'],
            ir_data['forward_function'],
            output_path
        )
        
        return ir_data


def compile_kernel_with_dummy_input(kernel, arg_types: List[Tuple]) -> None:
    """
    Force compilation of a kernel by providing dummy inputs
    
    Args:
        kernel: Warp kernel to compile
        arg_types: List of tuples (type, shape) for each argument
                   e.g., [(wp.array(dtype=float), 10), (float, None)]
    """
    inputs = []
    for arg_type, shape in arg_types:
        if isinstance(arg_type, type) and hasattr(arg_type, '__origin__'):
            # It's an array type
            if shape is None:
                shape = 10  # default size
            arr = wp.zeros(shape, dtype=wp.float32)
            inputs.append(arr)
        else:
            # It's a scalar
            inputs.append(0.0)
    
    # Launch with dim=1 just to force compilation
    wp.launch(kernel, dim=1, inputs=inputs)


if __name__ == "__main__":
    # Test the extractor
    wp.init()
    
    @wp.kernel
    def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), scale: float):
        tid = wp.tid()
        b[tid] = a[tid] * scale
    
    # Compile kernel
    n = 5
    a = wp.array(np.ones(n, dtype=np.float32))
    b = wp.zeros(n, dtype=wp.float32)
    wp.launch(test_kernel, dim=n, inputs=[a, b, 2.0])
    
    # Extract IR
    extractor = IRExtractor()
    ir_data = extractor.extract_ir(test_kernel)
    
    print("IR Extraction Test")
    print("=" * 80)
    print(f"Kernel: {ir_data['kernel_info']['key']}")
    print(f"\nPython source:\n{ir_data['python_source']}")
    print(f"\nForward function:\n{ir_data['forward_function'][:500]}...")
    print("\nExtraction successful!")
