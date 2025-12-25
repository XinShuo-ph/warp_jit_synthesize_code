import sys
sys.path.insert(0, '/workspace')

import warp as wp
import numpy as np

wp.init()

# Simple test kernel
@wp.kernel
def simple_test(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0 + 1.0

# Compile it
n = 5
a = wp.array(np.ones(n, dtype=np.float32))
b = wp.zeros(n, dtype=wp.float32)
wp.launch(simple_test, dim=n, inputs=[a, b])

# Manual extraction
from code.extraction.ir_extractor import IRExtractor
extractor = IRExtractor()
ir_data = extractor.extract_ir(simple_test)

print("Kernel hash:", ir_data.get('kernel_hash'))
print("\nForward function lines:", len(ir_data['forward_function'].split('\n')) if ir_data['forward_function'] else 0)
print("\nFirst 30 lines of forward function:")
if ir_data['forward_function']:
    for i, line in enumerate(ir_data['forward_function'].split('\n')[:30]):
        print(f"{i+1:3d}: {line}")
