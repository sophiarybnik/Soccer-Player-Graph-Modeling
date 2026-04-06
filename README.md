# Soccer Pass Prediction with Graph Neural Networks

This repository implements a **graph neural network (GNN)** for soccer pass predictions using StatsBomb 360 event data. It includes data processing, graph construction, model training, hyperparameter tuning, inference, and visualization.

Each passing situation is represented as a player graph — nodes are players, edges encode spatial relationships — and the model learns to predict where the ball carrier will pass next.



---

## Motivation

The primary goal of this project is to model and understand passing behavior using GNNs. By learning spatial and relational patterns between players, this model aims to:

- Predict likely pass destinations given a game state  
- Capture the structure of team play and decision-making  
- Provide a foundation for assessing similarity between players and tactical patterns

Ultimately, this work is a step toward quantifying **how players play**, enabling comparisons in style, strategy, and decision-making across matches and competitions.



---

## Project Structure

```
├── checkpoints/
│   ├── best_pass_gnn.pt        # Best trained model weights
│   ├── best_config.json        # Config from best tuning trial
│   ├── optuna_study.db         # Optuna study database
│   └── tuning_summary.json     # Full hyperparameter search history
├── configs/
│   └── optuna_search_space.json  # Bayesian search space + fixed params
├── data/
│   ├── graphs/
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   └── processed/
│       ├── events_full.parquet
│       ├── pass_events.parquet
│       └── event_type_vocab.json
├── notebooks/                  # Exploratory analysis
├── outputs/
│   └── predictions/            # Heatmap visualizations
├── scripts/                    # Pipeline entry points
│   ├── build_canonical_tables.py
│   ├── build_pass_graphs.py
│   ├── tune_pass_gnn.py
│   ├── train_pass_gnn.py
│   ├── evaluate_pass_gnn.py
│   └── visualize_predictions.py
└── src/
    ├── data/                   # StatsBomb loaders & canonical tables
    ├── datasets/               # PyG dataset wrappers
    ├── graphs/                 # Graph construction
    ├── inference/              # Prediction utilities
    ├── models/                 # GNN architecture
    ├── training/               # Trainer, Evaluator, loss functions
    ├── utils/                  # Config loading and helpers
    └── visualization/          # Heatmaps and pitch plotting
└── statsbomb_data/         # Raw StatsBomb JSON

```

---

## Installation

```bash
git clone https://github.com/sophiarybnik/statsbombpredictions.git
cd statsbombpredictions

conda create -n pass_gnn python=3.10
conda activate pass_gnn
pip install -r requirements.txt
```

---

## Pipeline

Run the full pipeline in order:

**1. Process raw StatsBomb data**
```bash
python scripts/build_canonical_tables.py
```

**2. Build pass graphs**
```bash
python scripts/build_pass_graphs.py
```

**3. Tune hyperparameters**
```bash
python scripts/tune_pass_gnn.py
```
Uses Bayesian optimization (Optuna TPE sampler) to search over architecture, training, and spatial parameters. To launch the dashboard, open a second terminal while tuning is running:

```bash
optuna-dashboard sqlite:///checkpoints/optuna_study.db
```

Then open http://localhost:8080 in browser.

**4. Train the model**
```bash
python scripts/train_pass_gnn.py
```
Trains using the best config from tuning. Best weights are saved to `checkpoints/best_pass_gnn.pt`.

**5. Evaluate**
```bash
python scripts/evaluate_pass_gnn.py
```
Reports loss, ADE (argmax and centroid), top-k accuracy, and rank metrics against random and centre-of-pitch baselines.

**6. Visualize predictions**
```bash
python scripts/visualize_predictions.py
```
Outputs pass predictions to `outputs/predictions/`.

---

## Model

`PassPredictionGNN` represents each passing situation as a graph and predicts a probability distribution over a discretized pitch grid.

**Architecture:**
- Node features are projected into a shared hidden space via a linear layer
- The actor node (ball carrier) receives an additional event type embedding
- Two stacked `GINEConv` layers propagate information across the graph, conditioning messages on edge attributes
- The actor's final embedding feeds into a prediction head that outputs logits over all grid cells

**Edge attributes** encode the spatial relationship between each pair of players: distance, angle, and binary flags for pressure (opponent bearing down on the actor) and support (available teammate).

**Loss function** (`pass_location_ce`): custom loss function over the discretized pitch grid, that incorporates Gaussian smoothing and KL divergence to train the model to understand that pass destinations exist in continuous space, not completely independent discrete classes.
---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ADE (argmax) | Distance in metres from predicted cell peak to true destination |
| ADE (centroid) | Distance using probability-weighted centroid |
| Top-k accuracy | Whether true destination appears in model's top k cells |
| Mean / median rank | Rank of true destination cell in the sorted probability distribution (out of all possible cells in discrete grid) |

Baselines: uniform random predictor and centre-of-pitch predictor.

---

## Configuration

All tunable parameters are defined in `configs/optuna_search_space.json`:

- `fixed` — parameters held constant across trials (e.g. `lr_patience`, `es_patience`)
- `categorical` — discrete choices (e.g. `hidden_dim`, `grid_x`, `batch_size`)
- `float` — continuous ranges (e.g. `lr`, `sigma`)
- `int` — integer ranges (e.g. `epochs`)

Node and edge dimensions are inferred automatically from the dataset at runtime.

---

## Notes

- `Trainer` and `Evaluator` are decoupled from model architecture, making it straightforward to swap loss functions or evaluation strategies independently.
- The pitch grid resolution (`grid_x`, `grid_y`) and Gaussian smoothing (`sigma`) are tunable; coarser grids are easier to classify, finer grids give more spatial precision.
- Pitch dimensions are fixed to StatsBomb coordinates (120×80m). The proximity threshold for edge construction can be adjusted in `src/graphs/config.py`.

---

## Next Steps

- Extend to other event types: shots, carries, defensive actions
- Add attention mechanisms (GAT / transformer-based GNN) for richer relational reasoning
- Incorporate temporal context — sequence of events leading up to the pass
- Predict pass success probability alongside destination
- Add k-fold cross-validation for more robust evaluation
- Animate passing sequences rather than static heatmaps