# Branch 9177 Analysis

## Quick Stats
- Milestone: M2/M3
- Data generated: 10,270 pairs
- Pipeline works: Yes

## Unique Features
- **10 Kernel Types**: Adds nested, multi_cond, combined, scalar_param.
- **Backward IR**: `pipeline.py` and `batch_generator.py` extract backward kernels.
- **SynthesisPair**: Dataclass for structured pair handling.
- **Batch Generator**: `batch_generator.py` with `BatchConfig` class.

## Code Quality
- Clean: Yes.
- Tests: `examples/` folder.
- Docs: `data_stats.md` confirms 10k pairs.

## Recommended for Merge
- [x] generator.py - Has more kernel types (10 vs 7 in 12c4).
- [x] pipeline.py - Supports backward IR extraction.
- [x] batch_generator.py - Advanced batch generation with backward support.
- [x] extraction/ir_extractor.py - Likely needed for backward extraction logic if not in pipeline.

## Comparison with 12c4
- 9177 has more kernel types (10 vs 6-7).
- 9177 supports backward IR.
- 12c4 has a larger generated dataset (10.5k vs 10.2k) but 9177 is more feature-rich.
- **Decision**: 9177 seems like a better base for code, 12c4 for data structure ideas?
- Actually, 9177 code seems superior due to backward support and more types.

## Skip
- None.
