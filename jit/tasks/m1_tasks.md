# Milestone 1 Tasks

## Task 1: Environment sanity + warp install
- [x] Step 1.1: Confirm Python + pip work (`python --version`, `python -m pip --version`)
- [x] Step 1.2: Install Warp (`python -m pip install --upgrade pip` then `python -m pip install warp-lang`)
- [x] Step 1.3: Verify import + basic device init (`python -c "import warp as wp; wp.init(); print(wp.__version__)"`)
- **Done when**: `import warp` and `wp.init()` succeed and report a version.

## Task 2: Access Warp repo examples locally (non-vendored)
- [x] Step 2.1: Clone Warp repo to a git-ignored location (e.g. `jit/_deps/warp`)
- [x] Step 2.2: Identify 3 examples that run headless (no GUI/USD) on CPU
- **Done when**: `jit/_deps/warp/warp/examples/` exists and 3 runnable example scripts are chosen.

## Task 3: Run 3+ examples successfully (twice)
- [x] Step 3.1: Run example A twice (prefer CPU-only env)
- [x] Step 3.2: Run example B twice
- [x] Step 3.3: Run example C twice
- [x] Step 3.4: Capture exact commands + outcomes in `jit/STATE.md` session log
- **Done when**: 3 distinct example scripts complete successfully two consecutive runs each.

## Task 4: Write minimal kernel compilation notes
- [x] Step 4.1: Create `jit/notes/warp_basics.md` (<= 50 lines) covering: kernel compilation flow, where to locate generated artifacts/IR
- **Done when**: `jit/notes/warp_basics.md` exists and is <= 50 lines.

