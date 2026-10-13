from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np

from ...types import Strategy
from .fitness import population_fitness
from .state import GAState


@dataclass(frozen=True)
class _GABaseStrategy(Strategy, ABC):

    mutation_rate: float = field(default=0.01)

    def _crossover(self, rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Crosses over two parents to produce a child."""
        axis = int(rng.integers(2))
        take_p2 = rng.random(p1.shape[axis]) < 0.5
        mask = take_p2[:, None] if axis == 0 else take_p2[None, :]
        return np.where(mask, p2, p1)

    def _mutate(self, rng: np.random.Generator, individual: np.ndarray) -> np.ndarray:
        """Mutates an individual."""
        mask = rng.random(individual.shape) < self.mutation_rate
        return individual ^ mask

    @abstractmethod
    def _select(self, rng: np.random.Generator, fitnesses: np.ndarray, n: int) -> np.ndarray:
        """Returns an (n, 2) array of parent index pairs to breed from, given their fitnesses."""
        ...

    def __call__(self, state: GAState) -> GAState:
        """Applies the genetic algorithm strategy to the given state."""
        fitnesses = population_fitness(state.pop_of_grids, state.clue_blocks)
        new_population = np.empty_like(state.pop_of_grids)

        new_population[0] = state.pop_of_grids[np.argmax(fitnesses)]  # elitism

        parents = self._select(state.rng, fitnesses, state.k - 1)
        for i, (a, b) in enumerate(parents, start=1):
            child = self._crossover(
                state.rng, state.pop_of_grids[a], state.pop_of_grids[b])
            new_population[i] = self._mutate(state.rng, child)

        return replace(state, pop_of_grids=new_population, step=state.step + 1)


@dataclass(frozen=True)
class GATournamentStrategy(_GABaseStrategy):
    tournament_size: int = 3

    def _select(self, rng, fitnesses, n):
        contenders = rng.integers(len(fitnesses), size=(n, 2, self.tournament_size))
        winner = fitnesses[contenders].argmax(axis=-1)
        return np.take_along_axis(contenders, winner[..., None], axis=-1)[..., 0]


@dataclass(frozen=True)
class _WeightedStrategy(_GABaseStrategy):
    """Parents drawn with probability proportional to a per-individual weight."""

    @abstractmethod
    def _weights(self, fitnesses: np.ndarray) -> np.ndarray: ...

    def _select(self, rng, fitnesses, n):
        weights = self._weights(fitnesses)
        total = weights.sum()
        p = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
        return rng.choice(len(weights), size=(n, 2), p=p)


@dataclass(frozen=True)
class GARankStrategy(_WeightedStrategy):
    def _weights(self, fitnesses):
        ranks = np.argsort(np.argsort(-fitnesses))
        return 1.0 / (ranks + 1)


@dataclass(frozen=True)
class GARouletteStrategy(_WeightedStrategy):
    def _weights(self, fitnesses):
        return fitnesses - fitnesses.min()
