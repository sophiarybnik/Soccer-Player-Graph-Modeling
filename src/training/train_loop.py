# src/training/train_loop.py
import torch
from torch_geometric.loader import DataLoader
from src.models.pass_gnn import PassPredictionGNN
from src.datasets.graph_dataset import GraphDataset
from src.training.trainer import Trainer
from src.training.eval import Evaluator
from src.training.losses import pass_location_ce
from functools import partial

def run_training(config, train_graphs, val_graphs, event_type_to_idx, device):
    """
    Train the model, report train and val loss per epoch.
    Returns the best model and history dictionary.
    """
    model = PassPredictionGNN(
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        hidden_dim=config["hidden_dim"],
        out_dim=config["out_dim"],
        num_event_types=len(event_type_to_idx),
        grid_x=config["grid_x"],
        grid_y=config["grid_y"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["lr_patience"], min_lr=1e-5
    )

    train_loader = DataLoader(GraphDataset(train_graphs), batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(GraphDataset(val_graphs), batch_size=config["batch_size"], shuffle=False)
    
    loss_fn = partial(pass_location_ce, grid_x=config["grid_x"], grid_y=config["grid_y"], sigma=1.5)
    trainer   = Trainer(model, optimizer, loss_fn, device)
    evaluator = Evaluator(model, loss_fn, grid_x=config["grid_x"], grid_y=config["grid_y"], device=device)


    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}


    for epoch in range(1, config["epochs"] + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss   = evaluator.evaluate(val_loader)["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["es_patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Load best model
    model.load_state_dict(best_state)

    return model, history