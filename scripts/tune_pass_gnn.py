import torch
from itertools import product
from pathlib import Path
import sys
import json

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.training.train_loop import run_training
from src.utils.config import load_config, merge_config

# Load configs
base_config = load_config(PROJECT_ROOT / "configs/base_config.json")
search_space = load_config(PROJECT_ROOT / "configs/search_space.json")


# Load graphs
graphs_dir = PROJECT_ROOT / "data/processed/graphs"
train_graphs = torch.load(graphs_dir / "train.pt")
val_graphs   = torch.load(graphs_dir / "val.pt")

# Infer Dimensions
sample = train_graphs[0]
base_config["node_dim"] = sample.x.shape[1]
base_config["edge_dim"] = sample.edge_attr.shape[1]

# Load event type vocab
with open(PROJECT_ROOT / "data/processed/event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

keys = search_space.keys()
best_val = float("inf")
best_config = None

# Total combinations for progress tracking
total = 1
for v in search_space.values():
    total *= len(v)

for i, values in enumerate(product(*search_space.values()), 1):
    updates = dict(zip(keys, values))
    config = merge_config(base_config, updates)

    # Print the config being tested
    print(f"\n[{i}/{total}] Testing config:\n{json.dumps(updates, indent=2)}")

    # Run training and get validation loss
    model, history = run_training(config, train_graphs, val_graphs, event_type_to_idx, DEVICE)
    val_loss = min(history["val_loss"])  # Best val loss in this run

    # Track best config
    if val_loss < best_val:
        best_val = val_loss
        best_config = config

# Print final best config
print("\n\u2705 Hyperparameter tuning complete!")
print("Best validation loss:", best_val)
print("Best configuration:")
print(json.dumps(best_config, indent=2))

# Save the best config
with open(PROJECT_ROOT / "checkpoints/best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)
