## Solvers

### About
- Time-budgeted nonogram solvers, all built from one `Anatomy` by picking a value
  on each of four independent axes.
- `BaseSolver.solve(clues, solution, max_time, seed)` runs
  iteratively until the puzzle is solved, progress stalls, or the wall-clock
  deadline is hit. It returns `(steps, len(steps), StoppingReason)`.
- There are no per-solver subclasses and no solver-held state. `reset(problem)`
  returns a `State` and `advance(state, problem)` returns a *new* one, so one
  instance can be reused across puzzles without leaking anything between them.
  `solve` is the driver that turns that pair into a trace, and is the only thing
  `generate_traces` needs to know about.
- The `generate_traces` package builds a trace dataset from a base dataset
  (`data/base_dataset/nonograms`) using the `NonogramsDataset` reader. Run it
  with `python -m puzzle_solvers.generate_traces` (see `data/README.md` for the
  dataset format).

### The four axes
A solver is a row in the declarative registry, not a class:

| axis | values |
| --- | --- |
| proposal | `belief_propagation`, `gradient_descent`, `evolutionary`, `backtracking`, `recursive_model` |
| schedule | `global`, `parallel`, `sequential`, `local` |
| aggregation | `identity`, `mean(k)`, `cr_weighted_mean(k)` |
| coupling | `none`, `evolutionary` |

Add a solver by adding a row, not by subclassing. The coupling axis is what
separates an ensemble of independent searches from a genetic algorithm: the
members are either stepped one at a time and never interact, or replaced
jointly by a generation drawn from the whole set.

```python
from puzzle_solvers.solvers import build_solver, describe_all

solver = build_solver("genetic_tournament")  # no per-solver subclass
steps, n, reason = solver.solve(clues, grid, max_time=3.0, seed=0)
describe_all()["genetic_tournament"]  # the anatomy, for trace metadata
```

### Registered solvers
| name | proposal | schedule | aggregation | coupling |
| --- | --- | --- | --- | --- |
| `mac` | belief propagation | global | identity | none |
| `mac_random_line` | belief propagation | local | identity | none |
| `mac_population` | belief propagation | local | mean(100) | none |
| `genetic_tournament` | evolutionary | global | mean(1000) | evolutionary (tournament) |
| `genetic_proportionate` | evolutionary | global | mean(1000) | evolutionary (proportionate) |
| `gd_adam` / `gd_sgd` / `gd_bangbang` | gradient descent | global | identity | none |
| `gd_adam_random_init` | gradient descent | global | identity | none |
| `gd_adam_random_init_population` | gradient descent | global | mean(10) | none |
| `backtracking_search` | backtracking | local | identity | none |
| `backtracking_population` | backtracking | local | mean(100) | none |
| `model_solver` | recursive model | global | identity | none |

The gradient-descent rows split on `init`: the first three start from a zero
grid of logits, the last two from a random one (see below). Every registered
gradient-descent row uses `Global`; `Parallel` and `Sequential` are reachable via
`anatomy_of` as the unregistered ablations `gd_jacobi_adam`, `gd_jacobi_sgd`,
`gd_gauss_seidel_adam`, `gd_gauss_seidel_sgd`, and are not run by
`generate_traces` unless asked for by name. `Sequential` in particular costs one
gradient per cell, so a step goes from one loss evaluation to one per cell.

### Assumptions
- Puzzles are square grids described by padded clues of shape `(2, H, K)` with
  `K = (H + 1) // 2`.
- The time limit is a safety net: speed beats fairness, and most solvers end
  early via `SOLVED` or `NO_CHANGE` before the deadline. Because it is
  wall-clock, a run that is cut off mid-search is *not* reproducible even with a
  fixed seed; a solver that terminates on its own is.
- Every step acts on every component its proposal can move. There is no
  sampling ratio: `Parallel` is `Global` under another name and `Sequential`
  chains one move per component.
- Every random draw comes from `Problem.rng`, seeded per solve. The solvers do
  not touch the global numpy or torch RNGs, so traces are reproducible and
  parallel workers cannot interfere with each other. `Mean(k)` gives each member
  its own generator, seeded from `(seed, member_index)`, so members are
  independent of each other and of the order they are stepped in.
- No core pinning or CPU-time accounting.

### Ensembles need a stochastic pair
`Mean(k)` is only meaningful when the (proposal, schedule) pair actually draws
from `Problem.rng`, and only a *one-axis* change if the scalar row already is.
Averaging `k` runs of a deterministic policy reproduces that policy exactly: the
row would advertise an ensemble in `describe()` and in the trace metadata while
carrying the information of a single solver. Measured distinct members after
three steps at k=8:

| proposal + schedule | distinct |
| --- | --- |
| `mac` + `global` | 1/8 |
| `gd_adam` + `global` | 1/8 |
| `mac_random_line` + `local` | 8/8 |
| `backtracking_search` + `local` | 8/8 |

`Local` supplies the variation: it shuffles the sweep order, so each member picks
a different first component. That makes an ensemble a one-axis change only when
the scalar solver is *already* on `Local` -- which is the case for
`backtracking_search` and `mac_random_line`, and not for `gd_adam` or
`model_solver`. Both are deterministic, so neither has a population row.

Forcing one anyway is a trap. `gd_adam`'s gradient is taken over the whole grid,
so moving it to `Local` is not a mild perturbation but a different algorithm. On
10 puzzles:

| variant | solved | mean steps | time |
| --- | --- | --- | --- |
| `global` + `identity` (= `gd_adam`) | 3/10 | 72.8 | 7.0s |
| `local` + `identity` | 0/10 | 52.4 | 3.9s |
| `local` + `mean(10)` | 0/10 | 87.9 | 60.6s |
| `global` + `mean(10)` | 3/10 | 72.8 | 44.4s |

The schedule change alone is the whole regression, and the ensemble on top buys
nothing; the clean `global` + `mean(10)` version reproduces `gd_adam`'s 72.8
steps exactly, at six times the cost. The only entropy a `global` gradient row
has is its initialisation, which is what the registered random-init pair supplies.
Tests pin both halves: every registered `Mean(k)` row must have more than one
distinct member, and must not have changed its schedule.

### Loss-weighted mean: fixing the readout, not the search
`Mean(k)` reports the arithmetic mean of its members, which is the whole point
of `mac_population` -- and its whole weakness. The members are `k` independent
trajectories and most never converge, so the recorded grid is close to all-0.5
everywhere and the `SOLVED`/`NO_CHANGE` checks in `keep_solving` almost never
trip: the population row looks unsolved even when some member has solved it.

`LossWeightedMean(k)` keeps the geometry identical and changes only how the
members are averaged: `w_i = softmax(-loss_i / temperature)` against each
member's clue-reconstruction loss. A member that has nailed the puzzle has loss
~0 and dominates the recorded grid; `temperature -> inf` recovers the plain
`Mean` and `temperature -> 0` picks the single best member. It is a pure
readout -- members still step independently, the weights only change what gets
recorded -- so it can never make the take longer; it can only stop the trace,
correctly, when a member reached the solution.

Every uncoupled `Mean(k)` row has a `cr_weighted_mean` twin registered as an
ablation (`mac_population_loss_weighted`, `backtracking_population_loss_weighted`,
`gd_adam_random_init_population_loss_weighted`) plus a `_loss_weighted` twin of
each genetic solver. On three trace puzzles with `temperature=0.05`,
`backtracking_population_loss_weighted` reported `SOLVED` on a puzzle where
`backtracking_population` (cr_loss 0.26) never did, and `mac_population`
converged from cr 0.0056 to 0.0008. The gradient-descent twin slightly reduced
the residual without reaching `SOLVED`.

### Random initialisation: the one-axis way to ensemble a deterministic solver
There is a second option that does not touch the schedule: perturb where the
trajectory *starts*. `GradientDescent(init="random")` draws its initial logits
from `Problem.rng`, and since each `Mean(k)` member holds its own stream, all `k`
members differ -- a genuine ensemble, reached by changing the proposal alone.
That is what the registered pair
`gd_adam_random_init` (single run) / `gd_adam_random_init_population` (`mean(10)`)
does, and it is the only way to ensemble gradient descent without also swapping
the schedule.

It helps. 200 puzzles, 10s budget, three seed bases per
scale (mean solved, and the per-base counts so the spread is visible):

| `init_scale` | solved /200 | per-base | sd |
| --- | --- | --- | --- |
| `zeros` (= `gd_adam`) | 45.0 | -- (deterministic) | 0 |
| +/-0.10 | 51.7 | 51, 56, 48 | 4.0 |
| **+/-0.25** | **53.0** | 51, 57, 51 | 3.5 |
| +/-0.50 | 51.3 | 52, 47, 55 | 4.0 |
| +/-0.75 | 48.7 | 48, 47, 51 | 2.1 |
| +/-1.00 | 44.0 | 40, 44, 48 | 4.0 |
| +/-2.00 | 32.3 | 29, 34, 34 | 2.9 |

The effect is real: +8/200 at the optimum, roughly 4 SE against a per-base sd of
3.5, and all nine runs at scales 0.10-0.50 beat the baseline. It is also free on
this budget -- 58 to 60 mean steps, and `timeout` stays at 0-1 everywhere. The
inverted-U is the part worth keeping: a perturbation too small stays in the same
basin, and one too large is worse than not starting randomly at all, so 0.25 is an
interior optimum of a real mechanism rather than a constant fitted to one run.
That is why `init_scale` is a proposal parameter and why 0.25 is its default.

The `no_change` column from the earlier 20-puzzle table is still the more
interesting line. 145-171 of 200 runs per configuration stop because the grid
stopped *changing*, not because the deadline hit. For belief propagation and
backtracking a fixed point means done; for gradient descent it means it converged
to a wrong local optimum, and `keep_solving` throws the trace away. Random init
helps precisely because it moves more runs into basins worth being in -- but the
stop rule is still reporting 97% of those stops on a *revisit* of an earlier step
rather than a genuine plateau, so it is not measuring what its name says. A
stopping rule that distinguishes "converged" from "gave up" would likely move
every number in the table above.

`Mean(10)` is registered alongside the single run so the question is answerable
from the dataset: whether averaging diverse restarts beats one of them is exactly
what the pair is for, and the k=1 sibling is the reference. Against a 2s budget
the mean row's traces are short by construction (gradient descent costs ~10x more
per member), so compare solve rate rather than step counts.

### Notes on individual solvers
- `mac`'s line DP is memoised per (clue, known pattern), so the population
  variants stay cheap.
- `backtracking_search` records one more step than it used to: the old solver
  advanced once inside `_first_step` without recording it.
- `backtracking_population` is the same search averaged over 100 independent
  shuffles. On 5x5 puzzles it is both more reliable and much shorter than
  `backtracking_search` (5/5 solved in 19-43 steps against 4/5 in 89-143,
  seed 1-5): the members disagree about which cell to commit, so the mean
  survives a wrong early commitment that a single trajectory cannot.
- `model_solver` loads its checkpoint lazily, on first use, and infers the
  architecture from the checkpoint (clue width, output channels, hidden size).
  Registry rows declare what they need in `Anatomy.requires`, so callers pass
  `build_solver(..., clues_shape=..., model_path=...)` without special-casing it.
- `mac_parallel` was removed. It never had an implementation -- the `mac_plus.py`
  it was named after was an empty placeholder that was never committed -- and as a
  registered row it was a byte-for-byte duplicate of `mac` in every dataset.
  Traces already generated under that name still load; only new runs stop
  emitting the column.

### TODO
- parallelism
- MAC heuristics
- constraint programming
- variable elimination
- PPO
- random forest
