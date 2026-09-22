import numpy as np
import torch


def display_grid(ax, grid, clues=None, title=""):
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    ax.imshow(grid, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)

    # Draw grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1, clip_on=False)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color="black", linewidth=1, clip_on=False)

    ax.axis("off")

    # Draw clues
    if clues is not None:
        for i, row_clue in enumerate(clues[0]):
            clue_str = ",".join(str(int(num)) for num in row_clue if num > 0) or "0"
            ax.text(-0.6, i, clue_str, va="center", ha="right", fontsize=12)
        for j, col_clue in enumerate(clues[1]):
            clue_str = ",".join(str(int(num)) for num in col_clue if num > 0) or "0"
            ax.text(j, -0.6, clue_str, va="bottom", ha="center", fontsize=12)

    ax.set_title(title, fontsize=14, pad=20)


def display_blended(ax, grid, clues=None, title=""):
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)

    smooth = torch.softmax(grid, dim=0)
    smooth = smooth.permute(1, 2, 0)

    base_colours = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # empty (0) → white
            [0.0, 0.0, 0.0],  # filled (1) → black
            [0.0, 0.0, 0.5],  # unknown (-1) → blue
        ]
    )  

    flat_weights = smooth.reshape(-1, 3)
    blended_flat = flat_weights @ base_colours
    blended = blended_flat.reshape(smooth.shape)
    img = blended.detach().cpu().numpy()

    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(-0.5, grid.shape[2] - 0.5)
    ax.set_ylim(grid.shape[1] - 0.5, -0.5)

    # Draw grid lines
    for i in range(grid.shape[1] + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1, clip_on=False)
    for j in range(grid.shape[2] + 1):
        ax.axvline(j - 0.5, color="black", linewidth=1, clip_on=False)

    ax.axis("off")

    # Draw clues
    if clues is not None:
        for i, row_clue in enumerate(clues[0]):
            clue_str = ",".join(str(int(num)) for num in row_clue if num > 0) or "0"
            ax.text(-0.6, i, clue_str, va="center", ha="right", fontsize=12)
        for j, col_clue in enumerate(clues[1]):
            clue_str = ",".join(str(int(num)) for num in col_clue if num > 0) or "0"
            ax.text(j, -0.6, clue_str, va="bottom", ha="center", fontsize=12)

    ax.set_title(title, fontsize=14, pad=20)

def display_blended_greyscale(ax, grid, clues=None, title=""):
    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)

    smooth = torch.softmax(grid[:2], dim=0)  # ignore the last channel
    smooth = smooth.permute(1, 2, 0)  # [H, W, 2]

    base_colours = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # empty (0) → white
            [0.0, 0.0, 0.0],  # filled (1) → black
        ]
    )

    flat_weights = smooth.reshape(-1, 2)          # [H*W, 2]
    blended_flat = flat_weights @ base_colours     # [H*W, 3]
    blended = blended_flat.reshape(*smooth.shape[:2], 3)  # [H, W, 3]
    img = blended.detach().cpu().numpy()

    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(-0.5, grid.shape[2] - 0.5)
    ax.set_ylim(grid.shape[1] - 0.5, -0.5)

    # Draw grid lines
    for i in range(grid.shape[1] + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1, clip_on=False)
    for j in range(grid.shape[2] + 1):
        ax.axvline(j - 0.5, color="black", linewidth=1, clip_on=False)

    ax.axis("off")

    # Draw clues
    if clues is not None:
        for i, row_clue in enumerate(clues[0]):
            clue_str = ",".join(str(int(num)) for num in row_clue if num > 0) or "0"
            ax.text(-0.6, i, clue_str, va="center", ha="right", fontsize=12)
        for j, col_clue in enumerate(clues[1]):
            clue_str = ",".join(str(int(num)) for num in col_clue if num > 0) or "0"
            ax.text(j, -0.6, clue_str, va="bottom", ha="center", fontsize=12)

    ax.set_title(title, fontsize=14, pad=20)


def ascii_print_grid(grid):
    if hasattr(grid, "tolist"):
        grid = grid.tolist()

    for row in grid:
        print(" ".join({1: "■", 0: "□", -1: "·"}.get(c, "?") for c in row))
