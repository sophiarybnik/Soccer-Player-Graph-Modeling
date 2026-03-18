# ⚽ Soccer Ball Pass Prediction Model

This repository implements a **graph neural network (GNN)** for soccer pass predictions using StatsBomb 360 event data. It includes data processing, graph construction, model training, hyperparameter tuning, inference, and visualization.

---

## 🎯 Project Motivation

The primary goal of this project is to model and understand passing behavior using GNNs. By learning spatial and relational patterns between players, this model aims to:

- Predict likely pass destinations given a game state  
- Capture the structure of team play and decision-making  
- Provide a foundation for assessing similarity between players and tactical patterns

Ultimately, this work is a step toward quantifying **how players play**, enabling comparisons in style, strategy, and decision-making across matches and competitions.

---

## 📁 Project Structure

```
C:.
├── checkpoints/            # Saved models
│   ├── best_pass_gnn.pt
│   └── latest_pass_gnn.pt
│   └── best_config.json
├── configs/                # Configuration files
│   ├── base_config.json
│   └── search_space.json
├── data/                   # Processed events data and pass graphs
│   ├── graphs/
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   └── processed/
│       ├── events_full.parquet
│       ├── pass_events.parquet
│       └── event_type_vocab.json
├── notebooks/              # Exploratory notebooks
├── outputs/
│   └── predictions/        # Prediction visualization output
├── scripts/                # Pipeline entry points
│   ├── build_canonical_tables.py
│   ├── build_pass_graphs.py
│   ├── evaluate_pass_gnn.py
│   ├── train_pass_gnn.py
│   ├── tune_pass_gnn.py
│   └── visualize_predictions.py
├── src/                    # Core source code
│   ├── data/               # StatsBomb loaders & canonical tables
│   ├── datasets/           # PyG datasets
│   ├── graphs/             # Graph construction
│   ├── inference/          # Prediction scripts
│   ├── models/             # GNN architectures
│   ├── training/           # Trainer, evaluator, and loops
│   ├── utils/              # Helper functions
│   └── visualization/      # Heatmaps and plotting utilities
└── statsbomb_data/         # Raw StatsBomb JSON
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
cd C:\Users\YourUser\Projects
git clone https://github.com/sophiarybnik/statsbombpredictions.git
```

2. Create a Python environment and install dependencies:

```bash
conda create -n pass_gnn python=3.10
conda activate pass_gnn
pip install -r requirements.txt
```
---

## 🏁 Quickstart

Run the full pipeline from raw StatsBomb data to visualized pass predictions:

1. **Load and process StatsBomb data:**

```bash
python scripts/build_canonical_tables.py
```

2. **Construct pass event graphs:**

```bash
python scripts/build_pass_graphs.py
```

3. **Run grid search over the hyperparameter space:**
```bash
python scripts/tune_pass_gnn.py
```

4. **Train the PassPredictionGNN model:**

```bash
python scripts/train_pass_gnn.py
```

5. **Visualize heatmaps of pass predictions:**

```bash
python scripts/visualize_predictions.py
```
* Outputs are saved to `outputs/predictions/`.


## 🎯 Evaluation / Inference

Evaluate or make predictions on new graphs:

```bash
python scripts/evaluate_pass_gnn.py
python src/inference/predict.py
```

* Uses a trained model (`best_pass_gnn.pt`) to compute metrics or generate predictions.


---

## 🧩 Configuration

* `configs/base_config.json` — default hyperparameters and dimensions.

* `configs/search_space.json` — grid search space for tuning.

* `src/graphs/config.py` — pitch and graph parameters. 

    **Note:** Pitch dimensions should remain fixed to the Statsbomb dimensions, but the proximity threshold can be tuned. Node and edge dimensions are inferred directly from the dataset.

---
## 🔧 Notes

* `Trainer` and `Evaluator` are modular and decoupled from model architecture, enabling easy experimentation with different loss functions and training strategies.

* The loss function (`pass_location_ce`) operates over a discretized pitch grid, converting continuous pass targets into classification over spatial cells. The grid resolution and gaussian smoothing can be controlled from `base_config.json`.
---

## 🚀 Next Steps / Improvements
* Extend the framework beyond passes to **include other event types** such as:
  - Shots
  - Carries / dribbles
  - Defensive actions (pressures, tackles)

* Add **attention mechanisms (GAT / transformer-based GNNs)** for better relational reasoning.
* Incorporate **temporal context** (sequence of events before the pass).
* Predict **pass success probability** alongside destination.
* **Replace grid search** with:
  - Random search
  - Bayesian optimization (e.g., Optuna)
* Add **experiment tracking** (e.g., MLflow, Weights & Biases).
* Add **k-fold cross-validation** for more robust evaluation.
* Animate **passing sequences** instead of static plots.

