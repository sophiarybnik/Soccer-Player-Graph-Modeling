import torch
import json
from pathlib import Path
import sys
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.models.pass_gnn import PassPredictionGNN
from src.datasets.graph_dataset import GraphDataset
from src.training.eval import Evaluator
from src.training.losses import pass_location_ce


checkpoint = torch.load(PROJECT_ROOT / "checkpoints/best_pass_gnn.pt")

config = checkpoint["config"]

graphs_dir = PROJECT_ROOT / "data/processed/graphs"
test_graphs = torch.load(graphs_dir / "test.pt")

with open(PROJECT_ROOT / "data/processed/event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

model = PassPredictionGNN(
    node_dim=config["node_dim"],
    edge_dim=config["edge_dim"],
    hidden_dim=config["hidden_dim"],
    out_dim=config["out_dim"],
    num_event_types=len(event_type_to_idx)
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

loader = DataLoader(GraphDataset(test_graphs), batch_size=config["batch_size"])

evaluator = Evaluator(model, pass_location_ce, DEVICE)

print("Test Loss:", evaluator.evaluate(loader))