from pathlib import Path
import sys
import json
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pass_gnn import PassPredictionGNN
from src.visualization.engine import plot_prediction_heatmap
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_pass_gnn.pt"

# --- Load data ---
graphs_dir = PROJECT_ROOT / "data" / "processed" / "graphs"
val_graphs = torch.load(graphs_dir / "val.pt")

# --- Load vocab ---
with open(PROJECT_ROOT / "data" / "processed" / "event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

# --- Load grid config ---
with open(PROJECT_ROOT / "checkpoints" / "best_config.json") as f:
    grid_config = json.load(f)
    GRID_X = grid_config["grid_x"]
    GRID_Y = grid_config["grid_y"]

# --- Load model ---
model = PassPredictionGNN(
    node_dim=5,
    edge_dim=4,
    hidden_dim=128,
    out_dim=128,
    num_event_types=len(event_type_to_idx),
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

print(f"Loaded model from {CHECKPOINT}")

# --- Run visualization ---
plot_prediction_heatmap(
    val_graphs[:100],
    model,
    grid_x=GRID_X,
    grid_y=GRID_Y,
    sigma=1.5,
    save_dir=PROJECT_ROOT / "outputs" / "predictions",
    device=DEVICE,
)
print(f"Saved visualizations to {PROJECT_ROOT}\\outputs\\predictions")
