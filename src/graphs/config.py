from dataclasses import dataclass
import math

@dataclass
class GraphConfig:
    # Pitch geometry (StatsBomb standard)
    pitch_length: float = 120.0
    pitch_width: float = 80.0

    # Train/val/test split for graph construction
    train_frac = 0.7
    val_frac = 0.15
    test_frac = 0.15

    # Edge construction
    proximity_threshold: float = 10.0  # meters

    # Normalization flags
    normalize_positions: bool = True
    normalize_edge_distance: bool = True

    @property
    def pitch_diagonal(self) -> float:
        return math.sqrt(
            self.pitch_length ** 2 + self.pitch_width ** 2
        )
