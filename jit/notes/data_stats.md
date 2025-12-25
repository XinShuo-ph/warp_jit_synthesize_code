# Dataset Statistics

## Summary
- **Total pairs**: 1,505 Python→C++ pairs
- **File size**: 8.2 MB (JSONL format)
- **Generation rate**: 0.84 pairs/sec (CPU-only mode)

## Distribution (10 kernel types)
| Type | Count | % |
|------|-------|---|
| elementwise | 167 | 11.1 |
| nested_branch | 161 | 10.7 |
| vector | 154 | 10.2 |
| compound | 153 | 10.2 |
| reduction | 153 | 10.2 |
| scalar_array | 150 | 10.0 |
| loop | 146 | 9.7 |
| unary | 142 | 9.4 |
| multi_stmt | 141 | 9.4 |
| branch | 138 | 9.2 |

## Size Metrics
- Python: avg 189 chars (125-271), total 278KB
- C++: avg 5.2KB (3.7-7.4KB), total 7.7MB
