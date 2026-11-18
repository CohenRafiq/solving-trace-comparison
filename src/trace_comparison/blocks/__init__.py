from .implementations.alignment import Resample, MeanPool, DynamicTimeWarping
from .implementations.comparison import MeanAbsoluteError, CosineSimilarity, CanonicalCorrelationAnalysis, CentredKernalAlignment, MaximumMeanDiscrepancy, ProcrustesCompare
from .implementations.other import ClassificationWeights, ImitationWeights, Centroid
from .implementations.same_shape import L2Norm, FrobeniusNorm, ZScoreNorm, MinmaxNorm, RankNorm, SubtractPuzzle, LeaceNorm, ProcrustesTransform
from .implementations.trace_emb import ClassificationLatent, ImitationLatent, ImitationPrediction, Fourier, PrincipalComponentAnalysis, UniformManifoldApproximation
from .implementations.matrix_norm import MatrixMinmax, MatrixRank

BLOCKS = {
    # Alignment
    "resample": Resample,
    "meanpool": MeanPool,
    "dtw": DynamicTimeWarping,

    # Comparison
    "mae": MeanAbsoluteError,
    "cosine": CosineSimilarity,
    "cca": CanonicalCorrelationAnalysis,
    "cka": CentredKernalAlignment,
    "mmd": MaximumMeanDiscrepancy,
    "procrustes_compare": ProcrustesCompare,

    # Other
    "classification_weights": ClassificationWeights,
    "imitation_weights": ImitationWeights,
    "centroid": Centroid,

    # Same shape
    "l2norm": L2Norm,
    "frobenius_norm": FrobeniusNorm,
    "zscore_norm": ZScoreNorm,
    "minmax_norm": MinmaxNorm,
    "rank_norm": RankNorm,
    "subtract_puzzle": SubtractPuzzle,
    "leace_norm": LeaceNorm,
    "procrustes_transform": ProcrustesTransform,

    # Trace embedding
    "classification_latent": ClassificationLatent,
    "imitation_latent": ImitationLatent,
    "imitation_prediction": ImitationPrediction,
    "fourier": Fourier,
    "pca": PrincipalComponentAnalysis,
    "umap": UniformManifoldApproximation,

    # Matrix norm
    "matrix_minmax": MatrixMinmax,
    "matrix_rank": MatrixRank
}

__all__ = [
    # Alignment
    "Resample", "MeanPool", "DynamicTimeWarping",
    # Comparison
    "MeanAbsoluteError", "CosineSimilarity", "CanonicalCorrelationAnalysis",
    "CentredKernalAlignment", "MaximumMeanDiscrepancy", "ProcrustesCompare",
    # Other
    "ClassificationWeights", "ImitationWeights", "Centroid",
    # Same shape
    "L2Norm", "FrobeniusNorm", "ZScoreNorm", "MinmaxNorm", "RankNorm",
    "SubtractPuzzle", "LeaceNorm", "ProcrustesTransform",
    # Trace embedding
    "ClassificationLatent", "ImitationLatent", "ImitationPrediction",
    "Fourier", "PrincipalComponentAnalysis", "UniformManifoldApproximation"
    # Matrix norm
    "MatrixMinmax", "MatrixRank"
]