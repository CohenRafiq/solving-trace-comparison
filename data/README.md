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
Dataset containing puzzles, solutions, step-by-step solving traces and metadata. Solving traces are the steps a solver would take to solve the puzzle (each step is the updated grid from the previous step). Same naming convention as `base_dataset` and also stored as jsonl. Code for construction is found `pipeline/solvers` and running instruction are found `pipeline/solvers/README.md`.

Current
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
    "mean_clue_runs": 1.5
  },
  "traces": {
    "mac": [{
      "steps": [
        [[0.25, 0.5, 1.0, 0.2, 0.2],
         [0.5, 0.333, 0.0, 0.2, 0.2],
         [0.5, 0.333, 1.0, 0.2, 0.2],
         [0.2, 0.111, 1.0, 0.059, 0.059],
         [0.1, 0.5, 1.0, 0.2, 0.077]],
        … 15 steps total, converging to the "grid" …
      ],
      "steps_to_solve": 5,
      "cr_losses": [0.995, 0.947, 0.930, …, 0.929],
      "mse_losses": [0.127, 0.080, 0.049, 0.020, 0.0, …, 0.0],
      "num_steps": 15,
      "solved": true,
      "step_ratio": 2
    }],
    "genetic_algorithm_det": [{
      "steps": [
        [[0.506, 0.513, 0.519, 0.497, 0.512],
         [0.506, 0.510, 0.470, 0.526, 0.479],
         … 5×5 grid per step, 15 steps …
        ],
        …
      ],
      "steps_to_solve": 7,
      "cr_losses": [1.280, 1.195, …, 0.936],
      "mse_losses": [0.247, 0.193, …, 0.0001],
      "num_steps": 15,
      "solved": true,
      "step_ratio": 2
    }],
    "genetic_algorithm_dep": [{ … }],
    "gradient_descent_global_adam": [{ … }],
    "gradient_descent_global_sgd": [{ … }],
    "gradient_descent_bang_bang": [{ … }],
    "model_solver": [{ … }],
    "backtracking_search": [{ … }]
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