import torch
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.train_loop import run_training
from src.utils.config import load_config
from src.visualization.plot_utils import plot_loss_curves


config = load_config(PROJECT_ROOT / "checkpoints/best_config.json")
config['epochs'] = 20  # Set to a reasonable number for training

print(f"Training with config:\n{json.dumps(config, indent=2)}")
graphs_dir = PROJECT_ROOT / "data/processed/graphs"
train_graphs = torch.load(graphs_dir / "train.pt")
val_graphs   = torch.load(graphs_dir / "val.pt")

with open(PROJECT_ROOT / "data/processed/event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

model, history = run_training(config, train_graphs, val_graphs, event_type_to_idx, DEVICE)

torch.save({
    "model_state_dict": model.state_dict(),
    "config": config
}, PROJECT_ROOT / "checkpoints/best_pass_gnn.pt")

print("\n\u2705 Training complete!")

# Plot and save curves
plot_loss_curves(
    history,
    save_path=PROJECT_ROOT / "outputs/train_val_curves.png"
)