# Merge State
- **Working Branch**: (run `git branch --show-current` and record here)
- **Phase**: P1
- **Current Branch**: 12c4 (first in queue)
- **Branches Completed**: []
- **Status**: ready_for_next

## Next Action
1. Create merge_notes directory:
   ```bash
   mkdir -p merge_notes
   ```
2. Start analyzing branch 12c4:
   ```bash
   git show origin/cursor/following-instructions-md-12c4:jit/notes/data_stats.md
   git ls-tree --name-only -r origin/cursor/following-instructions-md-12c4 | grep -E '\.(py|md)$' | head -30
   ```
3. Test pipeline from 12c4
4. Document findings in `merge_notes/12c4_notes.md`

Note: This merge workflow assumes the codebase is **JAX-based** (IR extraction via JAX lowering). If a branch is still Warp-based, capture that in the branch notes and skip merging Warp-specific code.

**First action**: Run `git branch --show-current` and record the branch name above.

## Branch Queue (from branch_progresses.md)
### Tier 1 - Must Process
- [ ] 12c4 (10,727 pairs)
- [ ] 9177 (10,320 pairs)
- [ ] 8631 (10,101 pairs)

### Tier 2 - Process for Features
- [ ] 82cf (775 pairs, README)
- [ ] aa30 (628 pairs, QUICKSTART)
- [ ] ff72 (371 pairs, clean docs)
- [ ] 3576 (239 pairs, test categories)
- [ ] 3a5b (100 pairs)

### Tier 3-4 - Quick Scan
- [ ] 25e7, 5d09, a4fd, 0fbe, 7288, 3f34, 4b76, d623

## Key Findings This Session
(none yet)

## Merge Decisions Made
(none yet)

## Session Log
- (initial): Merge workflow initialized, ready to begin P1 with branch 12c4

