# CPU Dataset Production Log

## Target: 200 MB - ✅ ACHIEVED (292 MB)

| Source | Samples | Size (MB) | Notes |
|--------|---------|-----------|-------|
| from_12c4 | 10,500 | 42 | Branch 12c4 large dataset |
| from_9177 | 10,290 | 99 | Branch 9177 batch_test dataset |
| from_8631 | 10,598 | 44 | Branch 8631 samples dataset |
| from_ff72 | 366 | 101 | Branch ff72 dataset (larger files) |
| batch_001 | 171 | 1.7 | Generated during production |
| batch_002 | 136 | 1.3 | Generated during production |
| final_batch | 111 | 1.0 | Generated during production |
| **TOTAL** | **31,754** | **292 MB** | **146% of target** |

## Generation Settings
- **Kernel types**: 10+ types (arithmetic, conditional, loop, math, vector, atomic, nested_loop, multi_conditional, combined, scalar_param, matrix, control_flow)
- **Backend**: CPU (LLVM IR / C++ codegen)
- **Sources**: Merged from 4 top-performing branches + freshly generated samples
- **Average sample size**: ~9.5 KB

## Quality Metrics (validated on 100 random samples)
- Valid Python→IR pairs: 100%
- Has Python source: 100%
- Has IR code: 100%
- Multiple IR formats supported:
  - cpp_forward / cpp_ir_forward (C++ LLVM IR)
  - ir / ir_code (structured IR)
  - Forward + backward passes available

## Data Sources
1. **Branch 12c4** (following-instructions-md-12c4):
   - jit/data/large/pair_*.json
   - Fields: python_source, cpp_forward, metadata
   
2. **Branch 9177** (following-instructions-md-9177):
   - jit/data/batch_test/*.json
   - Fields: python_source, cpp_ir_forward, cpp_ir_backward, kernel_type, metadata
   
3. **Branch 8631** (following-instructions-md-8631):
   - jit/data/samples/kernel_*.json
   - Fields: python_source, ir, name, args, id
   
4. **Branch ff72** (following-instructions-md-ff72):
   - jit/data/*.json
   - Fields: python_source, ir_code, kernel_type, device

## Production Status
- **Target**: 200 MB ✅
- **Achieved**: 292 MB (146%)
- **Status**: COMPLETE
- **Total Samples**: 31,754 Python→IR pairs
