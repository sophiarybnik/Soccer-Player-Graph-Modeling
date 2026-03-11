from src.utils.geometry import euclidean
from src.graphs.config import GraphConfig

def proximity_edges(freeze_frame: list[dict], threshold: float, config: GraphConfig):
    """
    Connect players who are close together on the pitch to represent potential interactions (e.g. marking, tackling, etc.)
    Edges are undirected since proximity is a mutual relationship.
    
    Args:
    - freeze_frame: list of player dicts in the freeze frame
    - threshold: distance in meters under which two players are considered close
    - config: GraphConfig object containing graph construction settings
    
    Returns:
    - edges: list of [source, target] pairs representing undirected edges between close players
    - attrs: list of edge attribute lists [distance, is_proximity, is_pressure, is_support] where distance is the (normalized) distance between players, and the binary flags indicate the type of edge

    """
    edges, attrs = [], []

    # Loop over all unique pairs of players in the freeze frame
    for i in range(len(freeze_frame)):
        for j in range(i + 1, len(freeze_frame)):
            d = euclidean(
                freeze_frame[i]["location"],
                freeze_frame[j]["location"]
            )

            if config.normalize_edge_distance:
                d /= config.pitch_diagonal
                threshold  /=config.pitch_diagonal # Normalize threshold by the same amount
            if d < threshold:
                edges.extend([[i, j], [j, i]]) # Add undirected edges in both directions
                attrs.extend([[d, 1, 0, 0], [d, 1, 0, 0]]) # d = actual distance between players, is_proximity = 1, is_pressure = 0, is_support = 0

    return edges, attrs


def pressure_edges(freeze_frame: list[dict], actor_idx: int, config: GraphConfig):
    """
    Connect opponents to the actor to represent pressure on the actor.
    
    Args:
    - freeze_frame: list of player dicts in the freeze frame
    - actor_idx: index of the actor in the freeze frame
    - config: GraphConfig object containing graph construction settings
    
    Returns:
    - edges: list of [source, target] pairs representing directed edges from opponents to the actor
    - attrs: list of edge attribute lists [distance, is_proximity, is_pressure, is_support] where distance is the (normalized) distance between players, and the binary flags indicate the type of edge
    """
    edges, attrs = [], []
    actor_loc = freeze_frame[actor_idx]["location"] # location of actor

    # Loop over all players in the freeze frame and connect opponents to the actor
    for i, p in enumerate(freeze_frame):
        if not p.get("teammate", False):
            d = euclidean(p["location"], actor_loc) 
            if config.normalize_edge_distance:
                d /= config.pitch_diagonal
            edges.append([i, actor_idx]) # Directed edge from opponent -> actor
            attrs.append([d, 0, 1, 0]) # d = distance between opponent and actor, is_proximity = 0, is_pressure = 1, is_support = 0

    return edges, attrs

def support_edges(freeze_frame: list[dict], actor_idx: int, config: GraphConfig):
    """
    Connect the actor to teammates to represent passing options.
    We consider only forward passes to be supportive since they represent realistic passing options that can advance the play.
    
    Args:
    - freeze_frame: list of player dicts in the freeze frame
    - actor_idx: index of the actor in the freeze frame
    - config: GraphConfig object containing graph construction settings
    
    Returns:
    - edges: list of [source, target] pairs representing directed edges from the actor to teammates
    - attrs: list of edge attribute lists [distance, is_proximity, is_pressure, is_support] where distance is the (normalized) distance between players, and the binary flags indicate the type of edge
    """
    edges, attrs = [], []

    actor_x = freeze_frame[actor_idx]["location"][0]
    actor_loc = freeze_frame[actor_idx]["location"] # location of actor

    for i, p in enumerate(freeze_frame):
        # Add directed edge from actor -> teammate if teammate is ahead on the pitch (i.e. has greater x coordinate)
        if p.get("teammate", False) and not p.get("actor", False): 
            if p["location"][0] > actor_x:
                d = euclidean(actor_loc, p["location"])
                if config.normalize_edge_distance:
                    d /= config.pitch_diagonal
                edges.append([actor_idx, i]) # Directed edge from actor -> teammate
                attrs.append([d, 0, 0, 1]) # d = distance between actor and teammate, is_proximity = 0, is_pressure = 0, is_support = 1

    return edges, attrs


