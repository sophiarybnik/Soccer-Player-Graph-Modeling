from pathlib import Path
import sys
import json
import torch
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pass_gnn import PassPredictionGNN
from src.datasets.graph_dataset import GraphDataset
from src.training.trainer import Trainer
from src.training.eval import Evaluator
from src.training.losses import pass_location_ce

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

############## HYPERPARAMETERS & CONFIG ##############
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 3e-3
SAVE_DIR    = PROJECT_ROOT / "checkpoints"
SAVE_DIR.mkdir(exist_ok=True)
LR_PATIENCE =4
ES_PATIENCE = 10

############## DATALOADER ##############
graphs_dir  = PROJECT_ROOT / "data" / "processed" / "graphs"
train_graphs = torch.load(graphs_dir / "train.pt")
val_graphs   = torch.load(graphs_dir / "val.pt")

train_loader = DataLoader(GraphDataset(train_graphs), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(GraphDataset(val_graphs),   batch_size=BATCH_SIZE, shuffle=False)

############## MODEL & TRAINING SETUP ##############
with open(PROJECT_ROOT / "data" / "processed" / "event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

model = PassPredictionGNN(
    node_dim=5,
    edge_dim=4, 
    hidden_dim=128,
    out_dim=128,
    num_event_types=len(event_type_to_idx)
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, min_lr=1e-5)

trainer   = Trainer(model=model, optimizer=optimizer, loss_fn=pass_location_ce, device=DEVICE)
evaluator = Evaluator(model=model, loss_fn=pass_location_ce, device=DEVICE)

############## TRAINING LOOP ##############
best_val_loss = float("inf")
epoch_since_improvement = 0

for epoch in range(1,NUM_EPOCHS+1):

    train_loss = trainer.train_epoch(train_loader)
    val_loss   = evaluator.evaluate(val_loader)

    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

    # Save latest checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, SAVE_DIR / "latest_pass_gnn.pt")

    """
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "val_loss": val_loss,
    "model_config": {
        "node_dim": 5,
        "edge_dim": 4,
        "hidden_dim": 128,
        "out_dim": 128,
        "num_event_types": len(event_type_to_idx)
    }
}, SAVE_DIR / "best_pass_gnn.pt")
    """

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_since_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }, SAVE_DIR / "best_pass_gnn.pt")
    else:
        epoch_since_improvement += 1

    if epoch_since_improvement >= ES_PATIENCE:
        print(f"Early stopping triggered after {epoch} epochs")
        break

print("\nTraining complete.")
