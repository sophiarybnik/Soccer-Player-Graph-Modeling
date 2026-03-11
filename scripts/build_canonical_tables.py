import sys
from pathlib import Path
import json


project_root = Path.cwd().parent 
sys.path.insert(0, str(project_root))

root_dir = Path(__file__).parent.parent / "statsbomb_data"
output_dir = project_root / "data" / "processed"

from src.data.loaders import (
    load_threesixty, 
    load_events, 
    load_matches)
from src.data.canonical import (
    build_events_with_360,
    enrich_events,
    extract_pass_events,
    build_event_type_vocab)


threesixty_df = load_threesixty(root_dir)
events_df = load_events(root_dir, set(threesixty_df.match_id))

matches_df = load_matches(root_dir)

events_360 = build_events_with_360(events_df, threesixty_df)
events_full = enrich_events(events_360, matches_df)

pass_events = extract_pass_events(events_full)

event_type_to_idx = build_event_type_vocab(events_full)


# Persist
events_full.to_parquet(output_dir / "events_full.parquet", engine="pyarrow")
pass_events.to_parquet(output_dir / "pass_events.parquet", engine="pyarrow")

# Save vocab
with open(output_dir / "event_type_vocab.json", "w") as f:
    json.dump(event_type_to_idx, f, indent=2)