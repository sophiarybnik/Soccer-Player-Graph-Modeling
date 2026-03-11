import torch
from src.graphs.config import GraphConfig

def build_nodes(freeze_frame: list[dict], config: GraphConfig) -> tuple[torch.Tensor, int]:
    node_features = []
    actor_idx = None

    for i, p in enumerate(freeze_frame):
        x, y = p["location"]

        # Normalize positions to [0, 1]
        if config.normalize_positions:
            x /= config.pitch_length
            y /= config.pitch_width

        # Feature flags
        is_teammate = int(p.get("teammate", False))
        is_actor = int(p.get("actor", False))
        is_keeper = int(p.get("keeper", False))

        # Identify actor index for later use in message passing
        if is_actor:
            actor_idx = i

        node_features.append([
            x, y,
            is_teammate,
            is_actor,
            is_keeper
        ])

    if actor_idx is None:
        raise ValueError("No actor in freeze_frame")

    return torch.tensor(node_features, dtype=torch.float32), actor_idx


