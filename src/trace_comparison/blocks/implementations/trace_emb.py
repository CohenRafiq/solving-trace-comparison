import numpy as np
from sklearn.decomposition import PCA
from dataclasses import dataclass
import torch

from ..block_types import TraceEmbeddingBlock, ModelBlock

class Latent(TraceEmbeddingBlock, ModelBlock):
    def forward(self, x: np.ndarray) -> np.ndarray:
        s, p = x.shape[:2]
        flat = torch.from_numpy(x.reshape(s * p, -1)).to(self.device)
        hidden = self.model.latent(flat)
        return hidden.reshape(s, p, -1).cpu().numpy()

    def get_context(self):
        cfg = self.payload.get("model_config", None)
        in_features = cfg.get("in_features", None)
        return {"model_input_dim": in_features}


class ClassificationLatent(TraceEmbeddingBlock, ModelBlock):
    model_type: str = "classification"
    
    def __post_init__(self):
        super().__post_init__()
        self.emb_dim = int(self.payload["model_config"]["hidden"])


class ImitationLatent(TraceEmbeddingBlock, ModelBlock):
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