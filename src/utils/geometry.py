import numpy as np
import torch

def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def denormalize_xy(x, y, pitch_length=120, pitch_width=80):
    return x * pitch_length, y * pitch_width

def xy_to_cell(x, y, grid_x, grid_y):
    """
    Convert normalized (x, y) coordinates to flat cell indices.

    Args:
        x: normalized x coordinates in [0, 1] — tensor of shape [N]
        y: normalized y coordinates in [0, 1] — tensor of shape [N]
        grid_x: cells along pitch length
        grid_y: cells along pitch width

    Returns:
        Long tensor of cell indices in [0, NUM_CELLS), shape [N]
    """
    ix = torch.clamp((x * grid_x).long(), max=grid_x - 1)
    iy = torch.clamp((y * grid_y).long(), max=grid_y - 1)
    return iy * grid_x + ix