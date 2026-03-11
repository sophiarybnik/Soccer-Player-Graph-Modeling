import json
import pandas as pd
import torch
from pathlib import Path

import sys
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.graphs.builder import build_event_graph
from src.graphs.config import GraphConfig

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def build_graph_dataset(
    pass_events,
    event_type_to_idx,
    output_dir: Path,
    train_frac=0.7,
    val_frac=0.15,
    seed=42
):
    rng = np.random.default_rng(seed)
    matches = pass_events.match_id.unique()
    rng.shuffle(matches)

    n_train = int(len(matches) * train_frac)
    n_val = int(len(matches) * val_frac)

    splits = {
        "train": matches[:n_train],
        "val": matches[n_train:n_train + n_val],
        "test": matches[n_train + n_val:]
    }

    config = GraphConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, match_ids in splits.items():
        graphs = []

        df_split = pass_events[pass_events.match_id.isin(match_ids)]

        for _, row in df_split.iterrows():
            g = build_event_graph(
                row,
                event_type_to_idx,
                config
            )
            if g is not None:
                graphs.append(g)

        torch.save(graphs, output_dir / f"{split}.pt")
        print(f"{split}: {len(graphs)} graphs")
