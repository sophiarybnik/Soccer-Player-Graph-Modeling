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
print("PROJECT_ROOT:", PROJECT_ROOT)


from src.models.pass_gnn import PassPredictionGNN
from src.visualization.engine import plot_prediction_heatmap
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_pass_gnn.pt"

# Load data
graphs_dir = PROJECT_ROOT / "data" / "processed" / "graphs"
val_graphs = torch.load(graphs_dir / "val.pt")

#  Load vocab
with open(PROJECT_ROOT / "data" / "processed" / "event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

# Load checkpoint and config
checkpoint = torch.load(PROJECT_ROOT / "checkpoints/best_pass_gnn.pt")
config = checkpoint["config"]
print(f"[checkpoint] loaded from checkpoints/best_pass_gnn.pt")
print(f"[config] {config}")

GRID_X = config["grid_x"]
GRID_Y = config["grid_y"]
NUM_CELLS = GRID_X * GRID_Y
print(f"[grid] {GRID_X} x {GRID_Y} = {NUM_CELLS} cells")


model = PassPredictionGNN(
    node_dim=config["node_dim"],
    edge_dim=config["edge_dim"],
    hidden_dim=config["hidden_dim"],
    out_dim=config["out_dim"],
    num_event_types=len(event_type_to_idx),
    grid_x=GRID_X,
    grid_y=GRID_Y,
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

print(f"Loaded model from {CHECKPOINT}")

# Run visualization
plot_prediction_heatmap(
    val_graphs[:100],
    model,
    grid_x=GRID_X,
    grid_y=GRID_Y,
    sigma=1,
    save_dir=PROJECT_ROOT / "outputs" / "predictions",
    device=DEVICE,
)
print(f"[saved] visualizations to → outputs/predictions")