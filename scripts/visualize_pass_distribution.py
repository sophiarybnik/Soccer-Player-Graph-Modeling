from pathlib import Path
import sys
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pass_gnn import PassPredictionGNN
from src.graphs.visualize import plot_prediction_heatmap
from src.models.config import GRID_X, GRID_Y

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

############## CONFIG ##############
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_pass_gnn.pt" 

############## LOAD DATA ##############
graphs_dir = PROJECT_ROOT / "data" / "processed" / "graphs"
val_graphs = torch.load(graphs_dir / "val.pt")

############## LOAD MODEL ##############
with open(PROJECT_ROOT / "data" / "processed" / "event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

model = PassPredictionGNN(
    node_dim=5,
    edge_dim=4,
    hidden_dim=128,
    out_dim=128,
    num_event_types=len(event_type_to_idx)
)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

"""
checkpoint = torch.load(CHECKPOINT)

model = PassPredictionGNN(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
"""

model.to(DEVICE)
model.eval()

print(f"Loaded model from {CHECKPOINT}")

############## VISUALIZE PREDICTIONS ##############
val_graphs = val_graphs[:100]
plot_prediction_heatmap(
    val_graphs,
    model,
    device=DEVICE,
    grid_x=GRID_X,
    grid_y=GRID_Y,
    sigma=1.5,
    save_dir=PROJECT_ROOT / "outputs" / "predictions"
)