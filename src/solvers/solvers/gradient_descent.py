import numpy as np
import torch

from neural_models_old.criterion.base import BaseCriterion
from solvers.solvers import BaseSolver
from trace_comparison.criterion.clue_reconstruction import ClueReconstructionLoss


class BaseGradientDescentSolver(BaseSolver):
    def __init__(self, loss_fn: BaseCriterion | None = None, **kwargs):
        self.loss_fn = loss_fn if loss_fn is not None else ClueReconstructionLoss(reduction="none")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(**kwargs)

    def _sample_grid(self, n: int, sampling_ratio: float) -> torch.Tensor:
        return (torch.rand(n, device=self.device) < sampling_ratio).float()

    def _sample_mask(self, n: int, sampling_ratio: float) -> torch.Tensor:
        return self._sample_grid(n, sampling_ratio)

    def _reset_optimizer_state(self, n: int) -> None:
        """Called once at the start of each `_step` call."""
        raise NotImplementedError

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """Return the parameter delta (same shape as `grad` / `idxs`).
        Called inside a `torch.no_grad()` block by the update mixins."""
        raise NotImplementedError

    def _optimise(
        self, prev: torch.Tensor, n: int, sampling_ratio: float, tensor_clues: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _step(
        self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float = 1.0
    ) -> np.ndarray:
        tensor_clues = torch.tensor(
            np.asarray(clues).flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        prev_t = torch.tensor(prev, dtype=torch.float32, device=self.device)
        prev_t = torch.clamp(prev_t, 1e-6, 1 - 1e-6)
        logit_grid = torch.logit(prev_t, eps=1e-6)

        steps_torch = [logit_grid]
        n = logit_grid.numel()

        self._reset_optimizer_state(n)

        for _ in range(num_steps):
            current = steps_torch[-1]
            new_logits = self._optimise(current, n, sampling_ratio, tensor_clues)
            steps_torch.append(new_logits)

        steps = [torch.sigmoid(step).detach().cpu().numpy() for step in steps_torch[1:]]
        return np.array(steps)


# =============================================================================
# Optimiser mixins
# =============================================================================


class AdamOptimizerMixin:
    default_step_ratio: int = 5
    device: torch.device
    _m: torch.Tensor | None
    _v: torch.Tensor | None
    _t: torch.Tensor | None

    def __init__(
        self,
        step_size: float = 1.5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None
        self._v = None
        self._t = None
        super().__init__(**kwargs)

    def _reset_optimizer_state(self, n: int) -> None:
        self._m = torch.zeros(n, device=self.device)
        self._v = torch.zeros(n, device=self.device)
        self._t = torch.zeros(n, dtype=torch.long, device=self.device)

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        assert self._m is not None and self._v is not None and self._t is not None
        self._t[idxs] = self._t[idxs] + 1
        t = self._t[idxs]

        m, v = self._m[idxs], self._v[idxs]

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)

        self._m[idxs] = m
        self._v[idxs] = v

        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        return self.step_size * m_hat / (v_hat.sqrt() + self.eps)

    def _compute_update_batch(self, grads: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """Vectorized Adam update for a batch of cells."""
        assert self._m is not None and self._v is not None and self._t is not None
        self._t[idxs] = self._t[idxs] + 1
        t = self._t[idxs].float()

        m = self._m[idxs]
        v = self._v[idxs]

        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * grads.pow(2)

        self._m[idxs] = m
        self._v[idxs] = v

        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        return self.step_size * m_hat / (v_hat.sqrt() + self.eps)


class SGDOptimizerMixin:
    default_step_ratio: int = 3

    def __init__(self, step_size: float = 70.0, **kwargs):
        self.step_size = step_size
        super().__init__(**kwargs)

    def _reset_optimizer_state(self, n: int) -> None:
        pass  # SGD is stateless

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        return self.step_size * grad

    def _compute_update_batch(self, grads: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """Vectorized SGD update for a batch of cells."""
        return self.step_size * grads


class BangBangOptimizerMixin:
    default_step_ratio: int = 3

    def __init__(self, step_size: float = 10.0, **kwargs):
        self.step_size = step_size
        super().__init__(**kwargs)

    def _reset_optimizer_state(self, n: int) -> None:
        pass

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        return self.step_size * torch.sign(grad) * torch.sqrt(torch.abs(grad) + 1e-8)

    def _compute_update_batch(self, grads: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        return self.step_size * torch.sign(grads) * torch.sqrt(torch.abs(grads) + 1e-8)


# =============================================================================
# Update-strategy mixins
# =============================================================================


class GlobalMixin:
    device: torch.device
    loss_fn: BaseCriterion

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        weight = self._sample_grid(n, sampling_ratio)  # type: ignore[attr-defined]

        anchor = prev.flatten().detach()
        flat_logits = anchor.clone().requires_grad_(True)

        loss = self.loss_fn(flat_logits.unsqueeze(0), tensor_clues)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss.sum()

        grad = torch.autograd.grad(loss, flat_logits)[0]

        with torch.no_grad():
            all_idxs = torch.arange(anchor.numel(), device=self.device)
            update = self._compute_update(grad, all_idxs)  # type: ignore[attr-defined]
            flat_opt = flat_logits - update
            blended = anchor + weight * (flat_opt - anchor)

        return blended.view(prev.shape)


class GaussSeidelMixin:
    device: torch.device
    loss_fn: BaseCriterion

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._sample_mask(n, sampling_ratio)  # type: ignore[attr-defined]
        idxs = mask.nonzero(as_tuple=True)[0]

        grid = prev.flatten().clone().detach()
        if idxs.numel() == 0:
            return grid.view(prev.shape)

        order = idxs[torch.randperm(idxs.numel(), device=self.device)]

        # Pre-allocate reusable tensor for the candidate grid
        candidate_buf = grid.clone()

        for _, i in enumerate(order):
            i_scalar = i.item()

            fixed_val = grid[i].clone()

            # Reuse pre-allocated buffer: reset to current grid state
            candidate_buf.copy_(grid)
            candidate_buf[i_scalar] = fixed_val
            candidate_buf.requires_grad_(True)

            loss = self.loss_fn(candidate_buf.unsqueeze(0), tensor_clues)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss = loss.sum()
            grad = torch.autograd.grad(loss, candidate_buf)[0]

            with torch.no_grad():
                grad_i = grad[i_scalar : i_scalar + 1]
                update = self._compute_update(grad_i, i.unsqueeze(0))  # type: ignore[attr-defined]
                grid[i] = fixed_val - update

            # Detach for next iteration
            candidate_buf.detach_()

        return grid.view(prev.shape)


class JacobiMixin:
    device: torch.device
    loss_fn: BaseCriterion

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._sample_mask(n, sampling_ratio)  # type: ignore[attr-defined]
        idxs = mask.nonzero(as_tuple=True)[0]

        grid = prev.flatten().clone().detach()
        if idxs.numel() == 0:
            return grid.view(prev.shape)

        anchor = grid.clone()  # frozen starting point for all gradients
        order = idxs[torch.randperm(idxs.numel(), device=self.device)]
        num_selected = order.numel()

        # Vectorized Jacobi: create batch of candidates, one per selected cell
        # Each candidate differs from anchor at exactly one position
        batch_candidates = anchor.unsqueeze(0).expand(num_selected, -1).clone()

        # Set each candidate's selected cell to a leaf variable with requires_grad
        fixed_vals = anchor[order].clone()
        free_vals = fixed_vals.clone().requires_grad_(True)

        # Scatter free_vals into the batch: candidate[i, order[i]] = free_vals[i]
        batch_idx = torch.arange(num_selected, device=self.device)
        batch_candidates[batch_idx, order] = free_vals

        # Batch forward pass through loss
        batch_clues = tensor_clues.expand(num_selected, -1)
        losses = self.loss_fn(batch_candidates, batch_clues)
        if isinstance(losses, tuple):
            losses = losses[0]
        # losses shape: (num_selected,)

        # Backward pass: get gradient of total loss w.r.t. free_vals
        total_loss = losses.sum()
        total_loss.backward()

        grads = free_vals.grad  # (num_selected,)
        assert grads is not None

        with torch.no_grad():
            updates = self._compute_update_batch(grads, order)

        # Apply all updates simultaneously
        grid[order] = fixed_vals - updates

        return grid.view(prev.shape)

    def _compute_update_batch(self, grads: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """Vectorized update computation for a batch of cells.
        Default implementation falls back to per-cell computation."""
        updates = []
        for i in range(idxs.numel()):
            update = self._compute_update(grads[i : i + 1], idxs[i : i + 1])  # type: ignore[attr-defined]
            updates.append(update)
        return torch.cat(updates)


# =============================================================================
# Concrete solvers
# =============================================================================


class GDGlobalAdamSolver(AdamOptimizerMixin, GlobalMixin, BaseGradientDescentSolver):
    pass


class GDGlobalSGDSolver(SGDOptimizerMixin, GlobalMixin, BaseGradientDescentSolver):
    pass


class GDGaussSeidelAdamSolver(AdamOptimizerMixin, GaussSeidelMixin, BaseGradientDescentSolver):
    pass


class GDGaussSeidelSGDSolver(SGDOptimizerMixin, GaussSeidelMixin, BaseGradientDescentSolver):
    pass


class GDJacobiAdamSolver(AdamOptimizerMixin, JacobiMixin, BaseGradientDescentSolver):
    pass


class GDJacobiSGDSolver(SGDOptimizerMixin, JacobiMixin, BaseGradientDescentSolver):
    pass


class GDBangBangSolver(BangBangOptimizerMixin, GlobalMixin, BaseGradientDescentSolver):
    pass
