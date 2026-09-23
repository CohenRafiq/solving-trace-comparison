import numpy as np

from puzzle_solvers.solvers.base import BaseSolver
from utils.grid import batch_line_clues


class BaseGeneticAlgorithm(BaseSolver):
    def __init__(
        self,
        population_size: int = 1000,
        max_samples: int = 1,
        mutation_rate: float = 0.01,
        sampling_modifier: float = 0.2,
    ):
        self.population_size = population_size
        self.max_samples = max_samples
        self.mutation_rate = mutation_rate
        self.sampling_modifier = sampling_modifier
        super().__init__()

    def _generate_initial_population(
        self,
        grid_shape: tuple[int, int],
        probabilities: np.ndarray,
        population_size,
        concurrent_samples,
    ) -> np.ndarray:
        shape = (population_size, concurrent_samples, *grid_shape)
        return (
            (np.random.random(size=shape) < probabilities)
            .astype(np.int8)
            .reshape(concurrent_samples, population_size, *grid_shape)
        )

    def _fitness(
        self,
        grids: np.ndarray,
        grid_shape: tuple[int, int],
        clues: np.ndarray,
        expected_num_runs: np.ndarray,
    ) -> np.ndarray:
        rows, cols = grid_shape
        row_lines = grids.reshape(-1, cols)
        col_lines = grids.transpose(0, 2, 1).reshape(-1, rows)

        def line_violations(
            lines: np.ndarray, clues: np.ndarray, expected_num_runs: np.ndarray
        ) -> np.ndarray:
            reps = lines.shape[0] // clues.shape[0]  # number of grids in this batch

            runs, num_runs = batch_line_clues(lines, clues.shape[1])
            tiled_clues = np.tile(clues, (reps, 1))
            tiled_expected = np.tile(expected_num_runs, reps)

            ok = np.all(runs == tiled_clues, axis=1) & (num_runs == tiled_expected)
            return ~ok

        row_violations = line_violations(row_lines, clues[0], expected_num_runs[0])
        col_violations = line_violations(col_lines, clues[1], expected_num_runs[1])
        total_row_violations = row_violations.reshape(grids.shape[0], rows).sum(axis=1)
        total_col_violations = col_violations.reshape(grids.shape[0], cols).sum(axis=1)
        return 1.0 / (1.0 + total_row_violations + total_col_violations)

    def _population_to_grid(self, population: np.ndarray) -> np.ndarray:
        combined_population = population.reshape(-1, *population.shape[-2:])
        mean_grid = combined_population.mean(axis=0)
        return mean_grid

    def _select_parents(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _apply_elitism(
        self, population: np.ndarray, fitness_scores: np.ndarray, offspring: np.ndarray
    ) -> np.ndarray:
        return offspring  # Default: no elitism, just return offspring

    def _crossover(self, parents1, parents2):
        cross_mask = np.random.random(size=parents1.shape) < 0.5
        return np.where(cross_mask, parents1, parents2)

    def _mutate(self, children):
        flip_mask = np.random.random(size=children.shape) < self.mutation_rate
        return np.where(flip_mask, 1 - children, children).astype(np.int8)

    def _step(self, clues, prev, num_steps, sampling_ratio):
        num_samples = self.max_samples
        population = self._generate_initial_population(
            prev.shape, prev, self.population_size, num_samples
        )
        steps = [self._population_to_grid(population)]
        expected_num_runs = (np.count_nonzero(clues[0], axis=1), np.count_nonzero(clues[1], axis=1))

        for _ in range(num_steps):
            fitness_scores = self._fitness(
                population.reshape(-1, *prev.shape), prev.shape, clues, expected_num_runs
            ).reshape(num_samples, self.population_size)
            noise_scale = self.sampling_modifier * (1.0 - sampling_ratio)
            multiplicative_noise = np.random.lognormal(0, noise_scale, size=fitness_scores.shape)
            noisy_fitness_scores = fitness_scores * multiplicative_noise
            noisy_fitness_scores = np.clip(noisy_fitness_scores, 0, 1)

            parents1 = self._select_parents(population, noisy_fitness_scores)
            parents2 = self._select_parents(population, noisy_fitness_scores)
            offspring = self._mutate(self._crossover(parents1, parents2))

            population = self._apply_elitism(population, fitness_scores, offspring)
            steps.append(self._population_to_grid(population))

        return np.stack(steps, axis=0)


class TournamentSelectionMixin:
    tournament_size: int = 3

    def _select_parents(self, population, fitness_scores):
        num_samples, pop_size = fitness_scores.shape
        num_children = pop_size - (1 if getattr(self, "_uses_elitism", False) else 0)
        s_range = np.arange(num_samples)

        contenders = np.random.randint(
            0, pop_size, size=(num_samples, num_children, self.tournament_size)
        )
        fit_c = fitness_scores[s_range[:, None, None], contenders]
        winner_local = np.argmax(fit_c, axis=2)
        winner_idx = np.take_along_axis(contenders, winner_local[..., None], axis=2).squeeze(-1)
        return population[s_range[:, None], winner_idx]


class ProportionateSelectionMixin:
    def _select_parents(self, population, fitness_scores):
        num_samples, pop_size = fitness_scores.shape
        num_children = pop_size - (1 if getattr(self, "_uses_elitism", False) else 0)
        s_range = np.arange(num_samples)
        weights = np.clip(fitness_scores, a_min=0, a_max=None)
        row_sums = weights.sum(axis=1, keepdims=True)
        uniform = np.ones_like(weights) / pop_size
        probs = np.where(row_sums > 0, weights / np.where(row_sums == 0, 1, row_sums), uniform)

        cum_probs = np.cumsum(probs, axis=1)

        draws = np.random.random(size=(num_samples, num_children))
        winner_idx = np.array(
            [np.searchsorted(cum_probs[i], draws[i], side="right") for i in range(num_samples)]
        )
        winner_idx = np.clip(winner_idx, 0, pop_size - 1)
        return population[s_range[:, None], winner_idx]


class ElitismMixin:
    _uses_elitism = True

    def _apply_elitism(self, population, fitness_scores, offspring):
        s_range = np.arange(fitness_scores.shape[0])
        elite_idx = np.argmax(fitness_scores, axis=1)
        elite = population[s_range, elite_idx]
        return np.concatenate([elite[:, None], offspring], axis=1)


class GeneticAlgorithmDET(TournamentSelectionMixin, ElitismMixin, BaseGeneticAlgorithm):
    pass


class GeneticAlgorithmDEP(ProportionateSelectionMixin, ElitismMixin, BaseGeneticAlgorithm):
    pass
