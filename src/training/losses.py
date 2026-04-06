import torch.nn.functional as F
import torch
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter


# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.geometry import xy_to_cell

def pass_location_mse(pred, target):
    return F.mse_loss(pred, target)

def pass_location_ce(logits, target_xy, grid_x, grid_y, sigma):
    # build soft target distribution
    target_cell = xy_to_cell(target_xy[:, 0], target_xy[:, 1], grid_x, grid_y)
    
    # one-hot then smooth
    soft_targets = torch.zeros_like(logits)
    soft_targets.scatter_(1, target_cell.unsqueeze(1), 1.0)
    
    # apply gaussian smoothing per sample in batch
    soft_targets_np = soft_targets.view(-1, grid_y, grid_x).cpu().numpy()
    for i in range(len(soft_targets_np)):
        soft_targets_np[i] = gaussian_filter(soft_targets_np[i], sigma=sigma)
    
    soft_targets = torch.tensor(soft_targets_np).view(-1, grid_x * grid_y).to(logits.device)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)  # renormalize
    
    # KL divergence instead of CE
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")