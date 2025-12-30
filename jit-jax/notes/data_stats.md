# Dataset Statistics

## Generation Summary
- **Total Pairs**: 10,000 Python→StableHLO pairs
- **Generation Time**: 48 seconds (209 pairs/second)
- **Output Directory**: data/training/

## Distribution by Function Type
| Type | Count | Percentage |
|------|-------|------------|
| Arithmetic | 2,028 | 20.3% |
| Reduction | 2,023 | 20.2% |
| Matrix | 2,011 | 20.1% |
| Compound | 1,992 | 19.9% |
| Unary | 1,946 | 19.5% |

## Content Statistics
- **Avg Python Length**: 58 chars
- **Avg IR Length**: 529 chars
- **Min IR Length**: 274 chars
- **Max IR Length**: 1,669 chars
- **IR/Python Ratio**: ~9x (IR is 9x longer than Python source)
