## Datasets

### TODO:
- Diagram showing example traces
- change data generation code to be ideal style
    - base datset
    - traces dataset
- reduce number of decimal places for traces dataset

### File Structure
```bash
data
├── base_dataset # puzzles + solution datasets 
│   └── nonograms
└── traces_dataset # base_dataset + solving traces for different solvers
    └── nonograms
```

### base_dataset
Base dataset conatining puzzles and solutions with some additional metadata. Used for training neural models and as a base for the trace dataset. Code for constructing the dataset can be found at `pipeline/base_dataset` and running instructions can be found in `pipeline/base_dataset\README.md`. Files are stored in jsonl format. Below is an example entry:

Current
```json
{
    "id": 0,
    "height": 5,
    "width": 5,
    "rows": [[1, 2], [1], [1, 3], [3], [2]],
    "cols": [[3, 1], [1], [1, 2], [1, 2], [2]],
    "grid": [
        [1, 0, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ]
}

```

Ideal
```json
{
    "clues": [
        [[1, 2], [1], [1, 3], [3], [2]],
        [[3, 1], [1], [1, 2], [1, 2], [2]]
    ],
    "grid": [
        [1, 0, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ],
    "meta": {
        "shape": [5, 5],
        "generation_time": 0.00234,
        "grid_density": 0.4,
        "mean_clue_runs": 1.5
    }
}
```
Naming convention is `[dimensions]-[entries]-[extra].jsonl` such as `5x5-100_000-no_search.jsonl`. Current code produces only puzzles with a single unique solution.

### traces_dataset
Dataset containing puzzles, solutions, step-by-step solving traces and metadata. Solving traces are the steps a solver would take to solve the puzzle (each step is the updated grid from the previous step). Same naming convention as `base_dataset` and also stored as jsonl. Code for construction is found in `src/puzzle_solvers/generate_traces` and running instructions in `src/puzzle_solvers/README.md`.

Each solver run is one key in `traces`, named `{base}_{seed}` (e.g. `ac_sgl_1`); a stochastic solver config with `seeds: [1, 2, 3]` produces three runs. Each run maps directly to one run record (no wrapping list).

```json
{
  "clues": [
    [[1], [2, 1], [1, 1, 1], [1], [2]],
    [[2], [1, 1], [1, 3], [1], [1]]
  ],
  "grid": [
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0]
  ],
  "meta": {
    "shape": [5, 5],
    "density": 0.4,
    "mean_clue_runs": 1.5,
    "solver_anatomy": {
      "ac_sgl_1": {
        "name": "ac_sgl_1",
        "strategy": "ACPerLineStrategy",
        "state": "ACState",
        "aggregator": "FirstAggregator",
        "seed": 1,
        "stopping_condition": "close",
        "solved_threshold": 0.2,
        "stopping_threshold": 0.001
      }
    }
  },
  "traces": {
    "ac_sgl_1": {
      "steps": [
        [[0.25, 0.5, 1.0, 0.2, 0.2],
         [0.5, 0.333, 0.0, 0.2, 0.2],
         [0.5, 0.333, 1.0, 0.2, 0.2],
         [0.2, 0.111, 1.0, 0.059, 0.059],
         [0.1, 0.5, 1.0, 0.2, 0.077]],
        "... one grid per step, converging to the grid ..."
      ],
      "clues": [[[1], [2, 1], [1, 1, 1], [1], [2]], [[2], [1, 1], [1, 3], [1], [1]]],
      "solution": [[0, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0]],
      "cr_losses": [0.995, 0.947, "...", 0.929],
      "mse_losses": [0.127, 0.080, "...", 0.0],
      "num_steps": 15,
      "solved": true,
      "stopping_reason": "SOLVED"
    },
    "ac_sgl_2": { "..." : "..." },
    "gd_adm_1": { "..." : "..." }
  }
}
```

Ideal
```json
{
  "clues": [
    [[1], [2, 1], [1, 1, 1], [1], [2]],
    [[2], [1, 1], [1, 3], [1], [1]]
  ],
  "grid": [
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0]
  ],
  "meta": {
    "shape": [5, 5],
    "generation_time": 0.00234,
    "density": 0.4,
    "mean_clue_runs": 1.5
  },
  "ac3": {
    "steps": [
      [[0.25, 0.5, 1.0, 0.2, 0.2],
       [0.5, 0.333, 0.0, 0.2, 0.2],
       [0.5, 0.333, 1.0, 0.2, 0.2],
       [0.2, 0.111, 1.0, 0.059, 0.059],
       [0.1, 0.5, 1.0, 0.2, 0.077]],
      "… only as many steps as needed to solve the puzzle / give up …"
    ],
    "rounded_steps": [
      [[0, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 1, 0, 0]],
      …
    ],
    "solver": "constraint_propagation",
    "solver_subtype": "ac-3",
    "sample_type": "mean_samples",
    "loss_per_step": [0.127, 0.080, 0.049, 0.020, 0.0, …, 0.0],
    "final_loss": 0.0,
    "solved": true,
    "step_ratio": 2
  },
  "genetic_tournament": { … },
  "gradient_descent_adam": { … },
  "gradient_descent_sgd": { … },
  "gradient_descent_bang_bang": { … },
  "recursive_mlp": { … },
  "backtracking_search": { … }
}
```

Example solving traces:

`EXAMPLE SOLVING TRACE PNG`


head -n1 base_1000.jsonl | jq '
def shape:
  if type == "array" then
    "[array len=\(length)]" + (if length > 0 then " of " + (.[0] | shape) else "" end)
  elif type == "object" then
    with_entries(.value |= shape)
  else
    type
  end;
shape
'