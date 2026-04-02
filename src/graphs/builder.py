import numpy as np
import torch
from torch_geometric.data import Data

from .nodes import build_nodes
from .edges import proximity_edges, pressure_edges, support_edges
from .config import GraphConfig

def build_event_graph(
    event_row: dict,
    event_type_to_idx: dict,
    config: GraphConfig
) -> Data | None:
    """    
    :param event_row: Description
    :type event_row: dict
    :param event_type_to_idx: Description
    :type event_type_to_idx: dict
    :param config: Description
    :type config: GraphConfig
    :return: Description
    :rtype: Data | None
    """
    
    
    freeze_frame = event_row.get("freeze_frame")
    if len(freeze_frame)==0:
        return None
    
    # Nodes
    x, actor_idx = build_nodes(freeze_frame, config) # x: [num_nodes, 5]

    # Target: normalized pass end location
    pass_end = event_row.get("end_location", None)
    if pass_end is None:
        return None  # skip graph if target missing
    
    end_x, end_y = pass_end

    if config.normalize_positions:
        end_x /= config.pitch_length
        end_y /= config.pitch_width

    y = torch.from_numpy(np.array([end_x, end_y], dtype=np.float32)).unsqueeze(0)    # normalized [x_end, y_end] -> shape [1, 2]
    
    # Event type
    event_type = event_row['type']['name']
    event_type_idx = event_type_to_idx[event_type]
    

    edge_list, edge_attr = [], []

    for fn in (
        lambda: proximity_edges(freeze_frame, config.proximity_threshold, config),
        lambda: pressure_edges(freeze_frame, actor_idx, config),
        lambda: support_edges(freeze_frame, actor_idx, config),
    ):
        e, a = fn()
        edge_list.extend(e)
        edge_attr.extend(a)

    if not edge_list:
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x, # node features [num_nodes, 5]
        edge_index=edge_index,
        edge_attr=edge_attr,
        actor_idx=torch.tensor([actor_idx]),
        actor_player_id=torch.tensor([event_row["player"]["id"]]),
        event_type_idx=torch.tensor([event_type_idx]),
        y=y,  # target (x_end, y_end)
        metadata={
            "player_name": event_row["player"]["name"],
            "event_type": event_type,
            "match_id": event_row["match_id"],
        }
    )

