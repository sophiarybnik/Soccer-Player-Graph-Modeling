import torch
import json
from pathlib import Path
import sys
from torch_geometric.loader import DataLoader
import functools
import math
import numpy as np
import warnings
import csv

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEVICE}")

from src.models.pass_gnn import PassPredictionGNN
from src.datasets.graph_dataset import GraphDataset
from src.training.eval import Evaluator
from src.training.losses import pass_location_ce
from src.graphs.config import GraphConfig

# Load checkpoint and config
checkpoint = torch.load(PROJECT_ROOT / "checkpoints/best_pass_gnn.pt")
config = checkpoint["config"]
print(f"[checkpoint] loaded from checkpoints/best_pass_gnn.pt")
print(f"[config] {config}")

GRID_X = config["grid_x"]
GRID_Y = config["grid_y"]
NUM_CELLS = GRID_X * GRID_Y
print(f"[grid] {GRID_X} x {GRID_Y} = {NUM_CELLS} cells")

# Load test graphs
graphs_dir = PROJECT_ROOT / "data/processed/graphs"
test_graphs = torch.load(graphs_dir / "test.pt")
print(f"[data] loaded {len(test_graphs)} test graphs")


with open(PROJECT_ROOT / "data/processed/event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

# Load model
model = PassPredictionGNN(
    node_dim=config["node_dim"],
    edge_dim=config["edge_dim"],
    hidden_dim=config["hidden_dim"],
    out_dim=config["out_dim"],
    num_event_types=len(event_type_to_idx),
    grid_x=GRID_X,
    grid_y=GRID_Y,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"[model] loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

# Dataloader
loader = DataLoader(
    GraphDataset(test_graphs),
    batch_size=config["batch_size"],
    shuffle=False,
)
print(f"[loader] {len(loader)} batches at batch_size={config['batch_size']}")


# Loss function
loss_fn = functools.partial(pass_location_ce, grid_x=GRID_X, grid_y=GRID_Y)

# Run evaluation
print("\n[eval] running full evaluation on test set...")
evaluator = Evaluator(
    model=model,
    loss_fn=loss_fn,
    grid_x=GRID_X,
    grid_y=GRID_Y,
    pitch_length=120.0,
    pitch_width=80.0,
    device=DEVICE,
)

metrics = evaluator.evaluate(loader)

# Print evaluation results
print("=" * 40)
print("  Test set evaluation")
print("=" * 40)
print(f"  Loss (CE):         {metrics['loss']:.4f}")
print(f"  ADE (argmax):      {metrics['ade_argmax']:.2f} m")
print(f"  ADE (centroid):    {metrics['ade_centroid']:.2f} m")
print(f"  Top-1 accuracy:    {metrics['top1_acc']*100:.1f}%")
print(f"  Top-3 accuracy:    {metrics['top3_acc']*100:.1f}%")
print(f"  Top-5 accuracy:    {metrics['top5_acc']*100:.1f}%")
print(f"  Mean rank:         {metrics['mean_rank']:.1f} / {NUM_CELLS}")
print(f"  Median rank:       {metrics['median_rank']:.1f} / {NUM_CELLS}")
print("=" * 40)

# Baselines
pitch_length = GraphConfig.pitch_length
pitch_width = GraphConfig.pitch_width
uniform_ade = (
    (pitch_length / GRID_X) ** 2 + (pitch_width / GRID_Y) ** 2
) ** 0.5 * math.sqrt(2) / 2

# centre-of-pitch baseline ADE
true_xs, true_ys = [], []
for g in test_graphs:
    true_xs.append(g.y[0, 0].item() * pitch_length)
    true_ys.append(g.y[0, 1].item() * pitch_width)

centre_ade = np.mean(np.sqrt(
    (np.array(true_xs) - pitch_length/2) ** 2 + (np.array(true_ys) - pitch_width/2) ** 2
))


print("\n  Baselines (for reference)")
print(f"  Random cell ADE:   {uniform_ade:.2f} m")
print(f"  Centre-of-pitch baseline ADE: {centre_ade:.2f} m")
print(f"  Random rank:      {NUM_CELLS / 2:.0f} / {NUM_CELLS}")
print("=" * 40)


results = {
    "loss":               metrics["loss"],
    "ade_argmax":         metrics["ade_argmax"],
    "ade_centroid":       metrics["ade_centroid"],
    "top1_acc":           metrics["top1_acc"],
    "top3_acc":           metrics["top3_acc"],
    "top5_acc":           metrics["top5_acc"],
    "mean_rank":          metrics["mean_rank"],
    "median_rank":        metrics["median_rank"],
    "baseline_random_ade":       uniform_ade,
    "baseline_centre_pitch_ade": centre_ade,
    "baseline_random_rank":      NUM_CELLS / 2,
    "grid_x":      GRID_X,
    "grid_y":      GRID_Y,
    "hidden_dim":  config["hidden_dim"],
    "out_dim":     config["out_dim"],
    "lr":          config["lr"],
    "sigma":       config["sigma"],
    "batch_size":  config["batch_size"],
    "epochs":      config["epochs"],
}

csv_path = PROJECT_ROOT / "checkpoints" / "eval_results.csv"
file_exists = csv_path.exists()

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(results)

print(f"[saved] eval results → checkpoints/eval_results.csv")