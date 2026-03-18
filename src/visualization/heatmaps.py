import numpy as np
from scipy.ndimage import gaussian_filter


def logits_to_heatmap(probs, grid_x, grid_y):
    """
    Convert flat probability vector to 2D heatmap
    """
    return probs.view(grid_y, grid_x).cpu().numpy()


def smooth_heatmap(heatmap, sigma=1.0):
    """
    Apply Gaussian smoothing to heatmap
    """
    return gaussian_filter(heatmap, sigma=sigma)


def get_pred_coordinates(heatmap, grid_x, grid_y, pitch_length, pitch_width):
    """
    Convert heatmap argmax to pitch coordinates
    """
    idx = heatmap.argmax()
    iy, ix = divmod(int(idx), grid_x)

    x = (ix + 0.5) / grid_x * pitch_length
    y = (iy + 0.5) / grid_y * pitch_width

    return x, y