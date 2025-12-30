# Dataset Statistics

## Summary
- **Total pairs**: 10,628 (10,500 main + 128 samples)
- **Generation rate**: ~183 pairs/second
- **Success rate**: 100%

## Distribution by Type
| Type | Count |
|------|-------|
| unary | 1,500 |
| binary | 1,500 |
| reduce | 1,500 |
| mixed | 1,500 |
| matmul | 1,500 |
| cond | 1,500 |
| norm | 1,500 |

## IR Statistics
- **Average JAXPR length**: 324 characters
- **Average HLO length**: 909 characters

## File Format
Each JSON file contains:
- `name`: Function identifier
- `source`: Python source code
- `jaxpr`: JAXPR representation
- `hlo`: XLA HLO/StableHLO representation
- `input_shapes`: List of input tensor shapes
