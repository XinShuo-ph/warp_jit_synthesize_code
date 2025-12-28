# Dataset Statistics

## Sample Dataset (/workspace/jit/data/samples/)
- **Total pairs**: 50
- **Seed**: 42
- **Generation**: Synthesized via pipeline.py

## Category Distribution
| Category     | Count | Percentage |
|--------------|-------|------------|
| vector       | 12    | 24.0%      |
| matrix       | 8     | 16.0%      |
| combined     | 7     | 14.0%      |
| arithmetic   | 6     | 12.0%      |
| atomic       | 6     | 12.0%      |
| control_flow | 6     | 12.0%      |
| math         | 5     | 10.0%      |

## Notes
- Sample data included for testing/validation
- Use `batch_generator.py` for large-scale generation (10k+ pairs)
- Generation rate: ~180 pairs/second on CPU
- 7 kernel types available (merged from ff72)
