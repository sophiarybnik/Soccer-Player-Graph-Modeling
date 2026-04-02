import torch
from pathlib import Path
import sys
import json
import optuna
from optuna.samplers import TPESampler

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

# Load search space
search_config = load_config(PROJECT_ROOT / "configs/optuna_search_space.json")
N_TRIALS = search_config.get("n_trials", 30)

# Load data
graphs_dir = PROJECT_ROOT / "data/processed/graphs"
train_graphs = torch.load(graphs_dir / "train.pt")
val_graphs   = torch.load(graphs_dir / "val.pt")

# Infer dims from data
sample = train_graphs[0]
NODE_DIM = sample.x.shape[1]
EDGE_DIM = sample.edge_attr.shape[1]

with open(PROJECT_ROOT / "data/processed/event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

print(f"[device] {DEVICE}")
print(f"[data] {len(train_graphs)} train / {len(val_graphs)} val graphs")
print(f"[dims] node_dim={NODE_DIM}, edge_dim={EDGE_DIM}")
print(f"[vocab] {len(event_type_to_idx)} event types")


def build_config(trial):
    """
    Construct a full config for one trial by combining:
      - dims inferred from the data (always fixed)
      - fixed params from search_config["fixed"]
      - sampled params from categorical / float / int sections
    """
    config = {
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "num_event_types": len(event_type_to_idx),
    }

    # Fixed param
    for name, value in search_config.get("fixed", {}).items():
        config[name] = value

    # Sampled params
    for name, choices in search_config.get("categorical", {}).items():
        config[name] = trial.suggest_categorical(name, choices)

    for name, bounds in search_config.get("float", {}).items():
        config[name] = trial.suggest_float(
            name, bounds["low"], bounds["high"], log=bounds.get("log", False)
        )

    for name, bounds in search_config.get("int", {}).items():
        config[name] = trial.suggest_int(name, bounds["low"], bounds["high"])

    return config


def objective(trial):
    config = build_config(trial)
    print(f"\n[trial {trial.number}] config: {json.dumps(config, indent=2, default=str)}")

    try:
        model, history = run_training(
            config, train_graphs, val_graphs, event_type_to_idx, DEVICE
        )
        val_loss = min(history["val_loss"])
    except Exception as e:
        print(f"[trial {trial.number}] failed: {e}")
        raise optuna.exceptions.TrialPruned()

    print(f"[trial {trial.number}] best val loss: {val_loss:.4f}")
    return val_loss


# Run study
storage = f"sqlite:///{PROJECT_ROOT / 'checkpoints' / 'optuna_study.db'}"

study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    study_name="pass_gnn_tuning",
    storage=storage,
    load_if_exists=True,   # resume if study already exists in the db
)

print(f"\n[optuna] starting study — {N_TRIALS} trials")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Results
best_trial = study.best_trial
print("\n" + "=" * 50)
print("  Hyperparameter tuning complete")
print("=" * 50)
print(f"  Best val loss:  {best_trial.value:.4f}")
print(f"  Best trial:     #{best_trial.number}")
print(f"  Best params:")
for k, v in best_trial.params.items():
    print(f"    {k}: {v}")
print("=" * 50)

# Save best config
best_config = build_config(best_trial)
with open(PROJECT_ROOT / "checkpoints/best_config.json", "w") as f:
    json.dump(best_config, f, indent=2, default=str)
print(f"\n[saved] best config → checkpoints/best_config.json")

# Save full study summary
summary = [
    {
        "trial":    t.number,
        "val_loss": t.value,
        "params":   t.params,
        "state":    str(t.state),
    }
    for t in study.trials
]
with open(PROJECT_ROOT / "checkpoints/tuning_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[saved] full study summary → checkpoints/tuning_summary.json")
