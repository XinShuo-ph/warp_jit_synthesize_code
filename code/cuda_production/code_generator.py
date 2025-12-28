"""
Python to CUDA Code Generator.
Converts Python Warp kernels to standalone CUDA C++ code.
"""
import re
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cuda_template import (
    complete_cuda_file, makefile_template, 
    TYPE_MAP, OP_MAP
)


class PythonToCUDATranslator:
    """Translates Python kernel source to CUDA C++."""
    
    def __init__(self):
        self.indent_level = 2  # Start at 2 for kernel body
    
    def translate_kernel(self, python_source):
        """
        Translate a Python Warp kernel to CUDA C++.
        
        Returns: (kernel_name, params, body_code)
        """
        # Parse function signature
        kernel_name, params = self._parse_signature(python_source)
        
        # Parse kernel body
        body = self._parse_body(python_source)
        
        # Translate body to CUDA
        cuda_body = self._translate_body(body)
        
        return kernel_name, params, cuda_body
    
    def _parse_signature(self, source):
        """Extract kernel name and parameters."""
        # Match: def kernel_name(param1: type1, param2: type2, ...):
        match = re.search(r'def\s+(\w+)\s*\((.*?)\):', source, re.DOTALL)
        if not match:
            raise ValueError("Could not parse function signature")
        
        kernel_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            
            # Match: name: wp.array(dtype=type)
            array_match = re.search(r'(\w+)\s*:\s*wp\.array\(dtype=(\w+(?:\.\w+)?)\)', param)
            if array_match:
                pname = array_match.group(1)
                ptype = array_match.group(2)
                cuda_type = TYPE_MAP.get(ptype, "float")
                params.append((pname, cuda_type))
        
        return kernel_name, params
    
    def _parse_body(self, source):
        """Extract kernel body lines."""
        lines = source.split('\n')
        body_lines = []
        in_body = False
        
        for line in lines:
            if 'def ' in line and '(' in line:
                in_body = True
                continue
            
            if in_body and line.strip():
                # Skip decorators and empty lines
                if not line.strip().startswith('@'):
                    body_lines.append(line)
        
        return body_lines
    
    def _translate_body(self, body_lines):
        """Translate Python body to CUDA C++."""
        cuda_lines = []
        
        for line in body_lines:
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                continue
            
            # Translate line
            cuda_line = self._translate_line(stripped)
            if cuda_line:
                cuda_lines.append(f"        {cuda_line}")
        
        return "\n".join(cuda_lines)
    
    def _translate_line(self, line):
        """Translate a single Python line to CUDA."""
        
        # tid = wp.tid()
        if 'wp.tid()' in line:
            return "// Thread ID computed from idx"
        
        # Variable assignment with array indexing
        # var = array[tid]
        match = re.search(r'(\w+)\s*=\s*(\w+)\[tid\]', line)
        if match:
            var_name = match.group(1)
            array_name = match.group(2)
            return f"float {var_name} = {array_name}[idx];"
        
        # Operation: var = op(arg1, arg2, ...)
        # Handle wp.sin, wp.cos, etc.
        for wp_op, cuda_op in OP_MAP.items():
            if wp_op in line and wp_op.startswith('wp.'):
                line = line.replace(wp_op, cuda_op)
        
        # Array store: out[tid] = value
        match = re.search(r'(\w+)\[tid\]\s*=\s*(.+)', line)
        if match:
            array_name = match.group(1)
            value = match.group(2)
            # Replace wp operations in value
            for wp_op, cuda_op in OP_MAP.items():
                value = value.replace(wp_op, cuda_op)
            # Replace [tid] with [idx]
            value = value.replace('[tid]', '[idx]')
            return f"{array_name}[idx] = {value};"
        
        # Variable assignment: var = expr
        match = re.search(r'(\w+)\s*=\s*(.+)', line)
        if match:
            var_name = match.group(1)
            expr = match.group(2)
            # Replace wp operations
            for wp_op, cuda_op in OP_MAP.items():
                expr = expr.replace(wp_op, cuda_op)
            # Replace [tid] with [idx]
            expr = expr.replace('[tid]', '[idx]')
            return f"float {var_name} = {expr};"
        
        return f"// TODO: {line}"
    
    def generate_cuda_file(self, python_source):
        """Generate complete CUDA .cu file from Python source."""
        kernel_name, params, body = self.translate_kernel(python_source)
        return complete_cuda_file(kernel_name, params, body)
    
    def generate_makefile(self, kernel_name):
        """Generate Makefile for the kernel."""
        return makefile_template(kernel_name)


def python_to_cuda(python_source, output_dir=None):
    """
    Convert Python kernel to CUDA code.
    
    Args:
        python_source: Python kernel source code
        output_dir: Optional directory to save .cu file
    
    Returns:
        dict with 'cuda_code', 'makefile', 'kernel_name'
    """
    translator = PythonToCUDATranslator()
    
    # Generate CUDA code
    cuda_code = translator.generate_cuda_file(python_source)
    
    # Extract kernel name
    kernel_name, _, _ = translator.translate_kernel(python_source)
    
    # Generate Makefile
    makefile = translator.generate_makefile(kernel_name)
    
    # Save if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save .cu file
        cu_file = output_path / f"{kernel_name}.cu"
        with open(cu_file, 'w') as f:
            f.write(cuda_code)
        
        # Save Makefile
        makefile_path = output_path / f"Makefile.{kernel_name}"
        with open(makefile_path, 'w') as f:
            f.write(makefile)
        
        # Save original Python
        py_file = output_path / f"{kernel_name}.py"
        with open(py_file, 'w') as f:
            f.write(python_source)
    
    return {
        'cuda_code': cuda_code,
        'makefile': makefile,
        'kernel_name': kernel_name,
        'params': translator.translate_kernel(python_source)[1]
    }


if __name__ == "__main__":
    # Test with simple example
    test_kernel = '''@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]
'''
    
    result = python_to_cuda(test_kernel, "/tmp/cuda_test")
    print("Generated CUDA code:")
    print(result['cuda_code'][:500])
    print("\n✓ Files saved to /tmp/cuda_test")
