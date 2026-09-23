from puzzle_solvers.solvers.base import BaseSolver, StoppingReason
from puzzle_solvers.solvers.genetic_algorithm import (
    GeneticTournament,
    GeneticProportionate,
)
from puzzle_solvers.solvers.gradient_descent import (
    GDBangBangSolver,
    GDGaussSeidelAdamSolver,
    GDGaussSeidelSGDSolver,
    GDGlobalAdamSolver,
    GDGlobalSGDSolver,
    GDJacobiAdamSolver,
    GDJacobiSGDSolver,
)
from puzzle_solvers.solvers.mac import MAC
from puzzle_solvers.solvers.model_solver import ModelSolver
from puzzle_solvers.solvers.search import BacktrackingSearch

__all__ = [
    "BaseSolver",
    "StoppingReason",
    "MAC",
    "GeneticAlgorithmDET",
    "GeneticAlgorithmDEP",
    "GDGlobalAdamSolver",
    "GDGlobalSGDSolver",
    "GDGaussSeidelAdamSolver",
    "GDGaussSeidelSGDSolver",
    "GDJacobiAdamSolver",
    "GDJacobiSGDSolver",
    "GDBangBangSolver",
    "BacktrackingSearch",
    "ModelSolver",
]

SOLVERS = {
    "mac": MAC,
    "genetic_tournament": GeneticTournament,
    "genetic_proportionate": GeneticProportionate,
    "gd_adam": GDGlobalAdamSolver,
    "gd_sgd": GDGlobalSGDSolver,
    "gd_bangbang": GDBangBangSolver,
    # "gradient_descent_gauss_seidel_adam": GDGaussSeidelAdamSolver,
    # "gradient_descent_gauss_seidel_sgd": GDGaussSeidelSGDSolver,
    # "gradient_descent_jacobi_adam": GDJacobiAdamSolver,
    # "gradient_descent_jacobi_sgd": GDJacobiSGDSolver,
    "model_solver": ModelSolver,
    "backtracking_search": BacktrackingSearch,
}
