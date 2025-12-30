# JAX JIT Code Synthesis

This directory contains a JAX-based implementation of the training data generation pipeline.

## Files

- `jax_generator.py`: Generates random JAX functions (similar to Warp kernels).
- `jax_ir_extractor.py`: Extracts HLO and other IRs from compiled JAX functions.
- `jax_pipeline.py`: Orchestrates generation and saving to JSONL.

## Usage

The usage is similar to the original Warp pipeline but uses `jax_pipeline.py`.

```bash
# Generate 100 pairs
python3 jit/code/synthesis/jax_pipeline.py --count 100 --output jit/data/jax_data.jsonl --jsonl

# Generate CPU/CUDA specific (Note: JAX auto-detects backend, so this flag doesn't force backend unless configured)
python3 jit/code/synthesis/jax_pipeline.py --count 100 --output jit/data/jax_cpu_only.jsonl --jsonl --device cpu
```

## Outputs

The generated JSONL files contain:
- `python`: JAX source code
- `hlo`: HLO (High Level Optimizer) IR text
- `llvm`: Low-level IR (Optimized HLO or LLVM IR depending on backend)
- `ptx`: PTX assembly (if running on GPU)
