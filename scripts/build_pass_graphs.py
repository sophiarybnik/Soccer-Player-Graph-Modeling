from pathlib import Path
import pandas as pd
import sys
import json

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.graph_dataset import build_graph_dataset
from src.graphs.config import GraphConfig


pass_events = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "pass_events.parquet"
)
print(f"[data] loaded {len(pass_events)} pass events")


with open(PROJECT_ROOT / "data" / "processed" / "event_type_vocab.json") as f:
    event_type_to_idx = json.load(f)

build_graph_dataset(
    pass_events=pass_events,
    event_type_to_idx=event_type_to_idx,
    output_dir=PROJECT_ROOT / "data" / "processed" / "graphs",
    config=GraphConfig()
)

print(f"[saved] pass graph objects → data/processed/graphs")

