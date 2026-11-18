import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
import torch
from umap import UMAP

from ..block_types import TraceEmbeddingBlock, ModelBlock

@dataclass
class Latent(TraceEmbeddingBlock, ModelBlock):
    def _apply(self, x: np.ndarray) -> np.ndarray:
        s, p, f = x.shape
        flat = torch.from_numpy(x.reshape(s * p, f)).float().to(self.device)
        hidden = self.model.latent(flat)
        return hidden.reshape(s, p, -1).cpu().numpy()

    def get_context(self):
        cfg = self.payload.get("model_config", None)
        in_features = cfg.get("in_features", None)
        return {"model_input_dim": in_features}


@dataclass
class ClassificationLatent(Latent, ModelBlock):
    model_type: str = "classification"
    
    def __post_init__(self):
        super().__post_init__()
        self.emb_dim = int(self.payload["model_config"]["hidden"])


@dataclass
class ImitationLatent(Latent, ModelBlock):
    model_type: str = "imitation"

    def output_shape(self, input_shape: np.ndarray) -> np.ndarray:
        self.check_shape(3, input_shape, "min")
        return torch.flatten(input_shape, start_dim=2).shape


class ImitationPrediction(TraceEmbeddingBlock, ModelBlock):
    ...


class Fourier(TraceEmbeddingBlock):
    ...


@dataclass(kw_only=True)
class PrincipalComponentAnalysis(TraceEmbeddingBlock):
    whiten: bool

    def __post_init__(self):
        self.pca = PCA(n_components=self.emb_dim, whiten=self.whiten)

    def _apply(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1])
        return self.pca.fit_transform(flat).reshape(*x.shape[:-1], self.emb_dim)

@dataclass(kw_only=True)
class UniformManifoldApproximation(TraceEmbeddingBlock):
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int | None = 0

    def __post_init__(self):
        self.umap = UMAP(
            n_components=self.emb_dim,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

    def _apply(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1])
        # UMAP needs n_neighbors < n_samples; clamp for small inputs
        self.umap.n_neighbors = max(2, min(self.n_neighbors, len(flat) - 1))
        out = self.umap.fit_transform(flat).astype(np.float32, copy=False)
        return out.reshape(*x.shape[:-1], self.emb_dim)