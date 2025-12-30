# Dataset Statistics

## Full Dataset (`data/full/`)
- Total pairs: 10,500
- Success rate: 100%
- Avg jaxpr length: ~340 chars
- Avg HLO length: ~1020 chars

## Sample Dataset (`data/samples/`)
- Total pairs: ~100
- Used for validation/testing

## Data Format
Each JSON pair contains:
- `python_source`: Original Python function
- `jaxpr`: JAX intermediate representation
- `hlo_text`: StableHLO text (MLIR format)
- `input_shapes`: List of input tensor shapes
- `seed`: Generation seed for reproducibility

## Function Categories
- Simple unary ops (sin, cos, exp, sqrt, etc.)
- Binary chains (add, mul, div combinations)
- Reductions (sum, mean, max over axes)
- Matrix operations (dot, matmul)
- Activations (relu, sigmoid, softmax, gelu)
- Normalizations (layer_norm, batch_norm)
- vmap patterns (batched operations)
- scan patterns (sequential/RNN-like)
