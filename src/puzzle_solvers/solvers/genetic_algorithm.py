import numpy as np

from puzzle_solvers.solvers.base import BaseSolver
from utils.grid import batch_line_clues


class BaseGeneticAlgorithm(BaseSolver):
    def __init__(
        self,
        population_size: int = 1000,
        mutation_rate: float = 0.01,
        sampling_modifier: float = 0.2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_modifier = sampling_modifier
        super().__init__()

    def _generate_initial_population(
        self,
        grid_shape: tuple[int, int],
        population_size: int,
    ) -> np.ndarray:
        shape = (population_size, *grid_shape)
        return (
            (np.random.random(size=shape) < 0.5)
            .astype(np.int8)
            .reshape(population_size, *grid_shape)
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

    def _step(self, clues, prev, sampling_ratio):
        expected_num_runs = (np.count_nonzero(clues[0], axis=1), np.count_nonzero(clues[1], axis=1))
        fitness_scores = self._fitness(
            self.population, prev.shape, clues, expected_num_runs
        )
        if sampling_ratio < 1.0:
            noise_scale = self.sampling_modifier * (1.0 - sampling_ratio)
            multiplicative_noise = np.random.lognormal(0, noise_scale, size=fitness_scores.shape)
            fitness_scores *= multiplicative_noise
            fitness_scores = np.clip(fitness_scores, 0, 1)
        parents1 = self._select_parents(self.population, fitness_scores)
        parents2 = self._select_parents(self.population, fitness_scores)
        offspring = self._mutate(self._crossover(parents1, parents2))

        self.population = self._apply_elitism(self.population, fitness_scores, offspring)
        return self.population.mean(axis=0)

    def _first_step(self, clues, grid_shape, sampling_ratio):
        self.population = self._generate_initial_population(
            grid_shape, self.population_size
        )
        return self.population.mean(axis=0)


class TournamentSelectionMixin:
    tournament_size: int = 3

    def _select_parents(self, population, fitness_scores):
        pop_size = len(fitness_scores)
        num_children = pop_size - (1 if getattr(self, "_uses_elitism", False) else 0)

        contenders = np.random.randint(
            0, pop_size, size=(num_children, self.tournament_size)
        )
        fit_c = fitness_scores[contenders]
        winner_local = np.argmax(fit_c, axis=1)
        winner_idx = np.take_along_axis(contenders, winner_local[:, None], axis=1).squeeze(-1)
        return population[winner_idx]


class ProportionateSelectionMixin:
    def _select_parents(self, population, fitness_scores):
        pop_size = len(fitness_scores)
        num_children = pop_size - (1 if getattr(self, "_uses_elitism", False) else 0)

        weights = np.clip(fitness_scores, a_min=0, a_max=None)
        row_sum = weights.sum()
        if row_sum > 0:
            probs = weights / row_sum
        else:
            probs = np.ones(pop_size) / pop_size

        cum_probs = np.cumsum(probs)

        draws = np.random.random(size=num_children)
        winner_idx = np.searchsorted(cum_probs, draws, side="right")
        winner_idx = np.clip(winner_idx, 0, pop_size - 1)
        return population[winner_idx]


class ElitismMixin:
    _uses_elitism = True

    def _apply_elitism(self, population, fitness_scores, offspring):
        elite_idx = np.argmax(fitness_scores)
        elite = population[elite_idx]
        return np.concatenate([elite[None], offspring], axis=0)


class GeneticTournament(TournamentSelectionMixin, ElitismMixin, BaseGeneticAlgorithm):
    pass


class GeneticProportionate(ProportionateSelectionMixin, ElitismMixin, BaseGeneticAlgorithm):
    pass
