# CUDA IR Production Dataset Statistics

**Dataset**: /workspace/cuda/data/cuda_production
**Generated**: True

## Overview

- Total pairs: 1200
- Categories: 6
- Perfect balance: True

## Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| arithmetic | 200 | 16.7% |
| atomic | 200 | 16.7% |
| control_flow | 200 | 16.7% |
| math | 200 | 16.7% |
| matrix | 200 | 16.7% |
| vector | 200 | 16.7% |

## Source Code Statistics

### Python Source
- Average lines: 6.1
- Range: 5 - 11 lines
- Average size: 177 characters

### CUDA IR
- Average lines: 39.1
- Range: 32 - 67 lines
- Average size: 1564 characters
- Expansion ratio: 6.4x

## CUDA Pattern Coverage

| Pattern | Count | Percentage |
|---------|-------|------------|
| atomic | 200 | 16.7% |
| blockDim | 1200 | 100.0% |
| blockIdx | 1200 | 100.0% |
| gridDim | 1200 | 100.0% |
| shared_memory | 1200 | 100.0% |
| threadIdx | 1200 | 100.0% |

## Operations Coverage

- trigonometric: 169 (14.1%)
- conditionals: 140 (11.7%)
- dot_product: 63 (5.2%)
- loops: 60 (5.0%)
- cross_product: 22 (1.8%)

## Quality Assessment

- ✓ All files contain CUDA patterns: True
- ✓ Balanced category distribution: True
- ✓ No duplicates detected
- ✓ Ready for LLM training
