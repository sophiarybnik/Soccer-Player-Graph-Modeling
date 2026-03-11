import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.geometry import xy_to_cell
from src.models.config import GRID_X, GRID_Y

def pass_location_mse(pred, target):
    return F.mse_loss(pred, target)


def pass_location_ce(logits, target_xy):
    # Categorical cross-entropy: treat each cell as a separate class, and the true pass destination (x, y) is mapped to a target cell index (a single integer from 0 to NUM_CELLS-1)
    target_cell = xy_to_cell(target_xy[:, 0], target_xy[:, 1], GRID_X, GRID_Y)
    return F.cross_entropy(logits, target_cell)