import sys
from pathlib import Path
import json


PROJECT_ROOT = Path.cwd().parent 
sys.path.insert(0, str(PROJECT_ROOT))

ROOT_DIR = Path(__file__).parent.parent / "statsbomb_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

from src.data.loaders import (
    load_threesixty, 
    load_events, 
    load_matches)
from src.data.canonical import (
    build_events_with_360,
    enrich_events,
    extract_pass_events,
    build_event_type_vocab)

# Data transformation: raw data -> enrich
threesixty_df = load_threesixty(ROOT_DIR)

events_df = load_events(ROOT_DIR, set(threesixty_df.match_id))
print(f"[data] loaded {len(events_df)} events")

matches_df = load_matches(ROOT_DIR)
print(f"[data] loaded {len(matches_df)} matches")

events_360 = build_events_with_360(events_df, threesixty_df)

events_full = enrich_events(events_360, matches_df)
print(f"[data] enriched {len(events_full)} events with 360 spatial data")

pass_events = extract_pass_events(events_full)
print(f"[data] extracted {len(pass_events)} pass events with 360 spatial data")

event_type_to_idx = build_event_type_vocab(events_full)

# Persist
events_full.to_parquet(OUTPUT_DIR / "events_full.parquet", engine="pyarrow")
pass_events.to_parquet(OUTPUT_DIR / "pass_events.parquet", engine="pyarrow")
print(f"[saved] enriched events data → data/processed/events_full.parquet")
print(f"[saved] pass event data → data/processed/pass_events.parquet")

# Save down event vocab
with open(OUTPUT_DIR / "event_type_vocab.json", "w") as f:
    json.dump(event_type_to_idx, f, indent=2)