import torch
from torch import nn

class NonogramPosEmbed(nn.Module):
    def __init__(self, shape, d_model):
        super().__init__()
        _, width, max_runs = shape
        self.axis_embed = nn.Embedding(2, d_model)
        self.line_embed = nn.Embedding(width, d_model)
        self.slot_embed = nn.Embedding(max_runs, d_model)

        axis_idx = torch.arange(2).view(2, 1, 1).expand(2, width, max_runs)
        line_idx = torch.arange(width).view(1, width, 1).expand(2, width, max_runs)
        slot_idx = torch.arange(max_runs).view(1, 1, max_runs).expand(2, width, max_runs)
        self.register_buffer('axis_idx', axis_idx)
        self.register_buffer('line_idx', line_idx)
        self.register_buffer('slot_idx', slot_idx)

    def forward(self, batch_size: int) -> torch.Tensor:
        pos = (self.axis_embed(self.axis_idx)
               + self.line_embed(self.line_idx)
               + self.slot_embed(self.slot_idx))          # (2, w, k, d_model)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return pos.flatten(start_dim=1, end_dim=3)        # (B, 2*w*k, d_model)  ← only first 3 dims


class NonogramClueEmbed(nn.Module):
    def __init__(self, shape, d_model):
        super().__init__()
        self.pos_embed = NonogramPosEmbed(shape, d_model)
        self.clue_embed = nn.Linear(1, d_model)

    def forward(self, clues, pad_mask):
        B = clues.shape[0]
        pos_embedded = self.pos_embed(B)                                  # (B, P, d_model)
        clue_embedded = self.clue_embed(
            clues.unsqueeze(-1).float()
        ).flatten(start_dim=1, end_dim=-2)                                # (B, P, d_model)

        out = pos_embedded + clue_embedded                                # (B, P, d_model)
        
        mask_flat = pad_mask.flatten(start_dim=1).unsqueeze(-1)       # (B, P, 1)
        out = out * mask_flat                                         # zero padded slots

        return out.mean(dim=1)  # (B, d_model)