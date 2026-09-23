## Solvers

### About
- Time-budgeted nonogram solvers sharing a common `BaseSolver` API.
- Each solver exposes `solve(clues, solution, max_time)` which runs iteratively
  until the puzzle is solved, progress stalls, or the wall-clock deadline is hit.
  `solve` returns `(steps, len(steps), StoppingReason)`.
- Solvers have a persistent per-solve state (`_reset_state`), reset once at the
  start of each `solve` call.
- The `generate_traces` package builds a trace dataset from a base dataset
  (`data/base_dataset/nonograms`) using the `NonogramsDataset` reader. Run it
  with `python -m puzzle_solvers.generate_traces` (see `data/README.md` for the
  dataset format).

### Assumptions
- Puzzles are square grids described by padded clues of shape `(2, H, K)` with
  `K = (H + 1) // 2`.
- The time limit is a safety net: speed beats fairness, and most solvers end
  early via `SOLVED` or `NO_CHANGE` before the deadline.
- No core pinning or CPU-time accounting.

### Solvers
- `MAC` — belief propagation over line probabilities (numpy/lru_cache, CPU).
- `GeneticAlgorithmDET` / `GeneticAlgorithmDEP` — tournament/proportionate
  selection with elitism (vectorised numpy, CPU).
- `GDGlobalAdam` / `GDGlobalSGDSolver` / `GDBangBang` — gradient descent on a
  clue-reconstruction loss with persistent torch optimizer state (GPU if
  available). Gauss-Seidel/Jacobi variants are kept but not registered.
- `BacktrackingSearch` — persistent single-trajectory incremental DFS (CPU).
- `ModelSolver` — recursive grid model. When constructed without an explicit
  model, it auto-loads a checkpoint (`--model-path` or `$MODEL_DIR/recursive_gridmlp.pt`)
  and infers the architecture from the checkpoint (clue width, output channels,
  hidden size). Included in the generator's default solver set.

### TODO
- parallelism
- MAC heuristics
- constraint programming
- variable elimination
- PPO
- random forest