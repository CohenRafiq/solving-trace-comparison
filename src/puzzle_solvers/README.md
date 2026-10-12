## Solvers

### About
- Time-budgeted nonogram solvers, each assembled from three independent parts:
  a `State`, a `Strategy`, and an `Aggregator`.
- `Solver.solve(clues, solution, max_time)` runs iteratively until the puzzle is
  solved, progress stalls, or the wall-clock deadline is hit. It returns
  `(steps, len(steps), StoppingReason)`.
- There are no per-solver subclasses and no solver-held state. `State` is a
  frozen dataclass and `Strategy.__call__(state)` returns a *new* one, so one
  instance can be reused across puzzles without leaking anything between them.
- A solver is a **config file**, not a class: `configs/solvers/<name>.yaml`
  names the three parts and hydra instantiates them.
- The `generate_traces` package builds a trace dataset from a base dataset
  (`data/base_dataset/nonograms`) using the `NonogramsDataset` reader. Run it
  with `python -m puzzle_solvers.generate_traces` (see `data/README.md` for the
  dataset format).

### The three axes
A solver is a row in a declarative config group, not a class:

| axis | classes |
| --- | --- |
| state | `ACState`, `BSState`, `GAState`, `GDSGDState`, `GDAdamState`, `NNState` |
| strategy | `ACParallelStrategy`, `ACPerLineStrategy`, `BSStrategy`, `GATournamentStrategy`, `GARankStrategy`, `GARouletteStrategy`, `GDStrategy`, `NNStrategy` |
| aggregator | `FirstAggregator`, `MeanAggregator`, `WeightedMeanAggregator` |

Add a solver by adding a file under `configs/solvers/`, not by subclassing. The
state carries the population size `k`, because that is what determines whether a
solver is an ensemble: `FirstAggregator` reads member 0, `MeanAggregator`
averages all `k`.

```python
from hydra import compose, initialize_config_dir
from puzzle_solvers.generate_traces.factory import build_solvers

with initialize_config_dir(config_dir="src/puzzle_solvers/configs", version_base=None):
    cfg = compose(config_name="generate_traces", overrides=["solvers=[genetic_tournament]"])

solver = build_solvers(cfg)["genetic_tournament"]
steps, n, reason = solver.with_seed(0).solve(clues, grid, max_time=3.0)
solver.describe()  # the anatomy, for trace metadata
```

Every part in the table above is named by a `_target_`, so a solver file is:

```yaml
ac_population:
  name: ac_population
  strategy:
    _target_: puzzle_solvers.solvers.arc_consistency.ACPerLineStrategy
  state:
    _target_: puzzle_solvers.solvers.arc_consistency.ACState
    k: 100
    seed: 42
  aggregator:
    _target_: puzzle_solvers.aggregation.MeanAggregator
```

Note the solver name as the top-level key: it is what lets several solver files
compose without their `strategy` / `state` / `aggregator` nodes overwriting one
another, and what makes `solvers=[a,b]` on the command line work.

### Configured solvers
| name | state | strategy | aggregator |
| --- | --- | --- | --- |
| `ac_parallel` | arc consistency (k=1) | parallel | first |
| `ac_population` | arc consistency (k=100) | per-line | mean(100) |
| `backtracking_search` | backtracking (k=1) | per-cell | first |
| `backtracking_population` | backtracking (k=100) | per-cell | mean(100) |
| `genetic_tournament` | GA (k=100) | tournament | mean(100) |
| `genetic_rank` | GA (k=100) | rank | mean(100) |
| `genetic_roulette` | GA (k=100) | roulette | mean(100) |
| `gd_adam` | Adam (k=1) | gradient descent | first |
| `gd_sgd` | SGD (k=1) | gradient descent | first |
| `gd_adam_population` | Adam (k=10) | gradient descent | mean(10) |
| `nn_recursive_mlp` | checkpointed MLP (k=1) | neural network | first |

The default `solvers` selection covers all of these, so a bare
`python -m puzzle_solvers.generate_traces` runs the whole catalogue -- which
means it needs the checkpoint (below) and takes minutes rather than seconds.
Narrow it with `solvers=[ac_parallel,backtracking_search]` for a quick run.

An ensemble needs a stochastic partner: averaging `k` copies of a deterministic
trajectory reproduces that trajectory. `ACParallelStrategy` and the
gradient-descent state both start from a fixed grid and never draw from
`state.rng`, so neither can be meaningfully averaged -- `ac_parallel` is
therefore paired with `ACPerLineStrategy` (which shuffles its line order), and
`gd_adam_population` is kept only to make that cost visible.

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
- Every random draw comes from `state.rng`, rebuilt from `(seed, step)` on every
  transition. The solvers do not touch the global numpy or torch RNGs, so traces
  are reproducible and parallel workers cannot interfere with each other. `k`
  members share one `state.rng` and are stepped in index order, so a member's
  draws depend on how many earlier members drew -- which is why the population
  configs below are paired with a strategy whose shuffle is what varies.
- No core pinning or CPU-time accounting.

### Running `generate_traces`
Hydra config, overridable from the command line:

```
python -m puzzle_solvers.generate_traces max_size=100 'solvers=[ac_parallel,gd_adam]'
python -m puzzle_solvers.generate_traces --cfg job     # show the composed config
python -m puzzle_solvers.generate_traces --help        # list the solver group
```

The output lands in `$DATA_DIR/traces_dataset/nonograms/<output>`, resolved
against the directory the command was run from rather than hydra's run
directory, so `data/traces_dataset/nonograms/` stays where the existing
consumers and DVC tracking expect it. Each selected solver becomes one column in
`traces`.

### Historical notes
Everything below this line was written against the previous `Anatomy` registry
(`mac`, `gd_bangbang`, `model_solver`, `anatomy_of`, `Problem.rng`, and the
four axes). The measurements still describe why the configurations above are
shaped the way they are, but the names no longer exist; the equivalents are in
the table above (`mac` ~ `ac_parallel`, `Mean(k)` ~ `MeanAggregator`, and so
on).

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
- `nn_recursive_mlp` loads its checkpoint lazily, on first use, and infers the
  architecture from the checkpoint rather than from the config: `hidden_size`
  and the output width come out of the state dict, so a checkpoint cannot be
  paired with a mismatched model. See below for the rest.

### Checkpoints
`state.model_path` is required in the solver config and is resolved against
`CHECKPOINT_DIR` -- `$MODEL_DIR` if set, else `<repo>/checkpoints`, made absolute
so it does not depend on the directory the process was launched from. That
directory has to match what `train.py` writes, which is
`<dataset>/<model_class>/<nn>.pt` with both names the class lowercased and
stripped of underscores (`NonogramsDataset` -> `nonogramsdataset`,
`RecursiveMLP` -> `recursivemlp`). So the 5x5 MLP is
`nonogramsdataset/recursivemlp/default.pt`, *not* `recursive_mlp/default.pt`.

Two things are deliberately not inferred:

- `num_layers`. `RecursiveMLP` applies one shared `ResidualBlock` in a loop, so
  it owns no parameters. It also is not the same thing as the solver's step
  count: training applied it 9 times inside one forward pass, whereas here each
  solver step applies it once, so a trace comparable to a training forward runs
  for about 9 steps.
- The grid size. The clue width is baked into `clue_embed.pos_embed`, so a 5x5
  checkpoint cannot drive a 10x10 puzzle; that is rejected on load with the two
  sizes named, rather than as an opaque size mismatch from `load_state_dict`.

The model emits `num_classes` logits per cell rather than one score, because
`AbstainLoss` trains two classes plus an abstain channel. `NNState.grids`
therefore softmaxes over the class axis and renormalises by the probability the
model did *not* abstain -- the same renormalisation the loss optimises -- so the
trace carries a probability per cell like every other solver.

### Readout and stopping
`State.grids` is the readout, and what a trace records; `State.raw_grids` is the
state as stored. They differ wherever a readout is lossy, and that difference
matters for stopping: `sigmoid'` is at most 0.25, so an Adam step that moves
the logits by about `lr` lands *inside* the `1e-3` stop tolerance once mapped to
[0, 1]. Testing "did this move" on the readout therefore ended `gd_adam`'s trace
two steps in instead of ~190. `Solver.solve` decides `SOLVED` on the readout
(where `solution` lives) and `NO_CHANGE` on the raw state (where the search
moves), so changing a readout cannot silently change how long a solver runs.
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
