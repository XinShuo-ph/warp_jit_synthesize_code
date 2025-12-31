# Dataset Statistics

## Generation Summary
- **Total pairs**: 10,000
- **Success rate**: 100%
- **Generation time**: ~10 minutes (16.9 pairs/sec)
- **File size**: ~17 MB (JSONL format)

## Category Distribution (top 10)
| Category | Count | % |
|----------|-------|---|
| matmul | 1,450 | 14.5% |
| nn_layer | 1,427 | 14.3% |
| composite | 1,421 | 14.2% |
| vmap | 1,421 | 14.2% |
| max | 463 | 4.6% |
| min | 448 | 4.5% |
| sum | 248 | 2.5% |
| std | 234 | 2.3% |
| sub | 233 | 2.3% |
| mean | 226 | 2.3% |

## Data Format
Each pair contains: `python_source`, `jaxpr`, `stablehlo`, `xla_hlo`, `input_shapes`, `output_shape`
