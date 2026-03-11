from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch
import torch
from torch_geometric import data
from src.utils.geometry import denormalize_xy
from src.models.config import GRID_X, GRID_Y, NUM_CELLS
from src.graphs.config import GraphConfig
from torch_geometric.data import Batch


def pyg_to_nx(graph):
    """
    Convert a torch_geometric Data graph into a networkx DiGraph for visualization.
    Args:
        graph: torch_geometric.data.Data object with attributes:
            - x: [num_nodes, node_dim] node features (must include x,y positions)
            - edge_index: [2, num_edges] source and target node indices
            - edge_attr: [num_edges, edge_dim] edge features (must include relationship types)
    Returns:
        G: networkx DiGraph with node attributes (x, y, is_teammate, is_actor) and edge attributes (distance, is_proximity, is_pressure, is_support)    
    
    """

    G = nx.DiGraph()
    
    # Add nodes with attributes
    for i, node_feat in enumerate(graph.x):
        G.add_node(
            i,
            x=node_feat[0].item(),
            y=node_feat[1].item(),
            is_teammate=bool(node_feat[2].item()),
            is_actor=bool(node_feat[3].item())
        )
    
    # Add edges with attributes
    for src, dst, attr in zip(graph.edge_index[0], graph.edge_index[1], graph.edge_attr):
        G.add_edge(
            src.item(),
            dst.item(),
            distance=attr[0].item(),
            is_proximity=bool(attr[1].item()),
            is_pressure=bool(attr[2].item()),
            is_support=bool(attr[3].item())
        )
    return G


def plot_event_graph(graph, title="Event Graph"):
    """
    Visualize a single pass event graph
    Args:
        graph: torch_geometric.data.Data object
        title: title for the plot
    Returns:
        None
    """

    G = pyg_to_nx(graph)
    pos = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}

    plt.figure(figsize=(12, 8))

    for condition, color, style, width in [
        ("is_proximity", "gray", "solid", 1),
        ("is_pressure", "orange", "dashed", 2),
        ("is_support", "purple", "solid", 3),
    ]:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d[condition]]
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               edge_color=color, style=style, width=width)

    node_colors = [
        "red" if G.nodes[n]["is_actor"]
        else "green" if G.nodes[n]["is_teammate"]
        else "blue"
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Edge types
    edge_definitions = [
        ("is_proximity", "gray", "solid", 1, "Proximity"),
        ("is_pressure", "orange", "dashed", 2, "Pressure"),
        ("is_support", "purple", "solid", 3, "Support"),
    ]

    # Custom legend for edges
    legend_elements = [
        Line2D([0], [0], color=color, lw=width, linestyle=style, label=label)
        for _, color, style, width, label in edge_definitions
    ]
    plt.legend(handles=legend_elements, loc='upper right')


    plt.title(title)
    plt.xlim(0, 120)
    plt.ylim(0, 80)
    plt.gca().set_aspect("equal")
    plt.show()


def plot_prediction_heatmap(
    graphs,
    model,
    grid_x,
    grid_y,
    save_dir,
    device="cpu",
    sigma=1.0,
    figsize=(16, 7)
):
    """
    Visualize both player configuration and predicted pass destination heatmap side-by-side.
    
    Left subplot:  Player graph on football pitch with edges representing relationships
    Right subplot: Heatmap of predicted pass destinations with pass start and true end locations
    
    Args:
        graphs:     single torch_geometric Data object or list of Data objects representing pass events
        model:      trained PassPredictionGNN
        device:     "cpu" or "cuda"
        grid_x:     number of grid cells in x-direction, should match GRID_X config (default: 120)
        grid_y:     number of grid cells in y-direction, should match GRID_Y config (default: 80)
        save_dir:   directory to save visualizations
        sigma:      Gaussian smoothing parameter for heatmap (default: 1.0)
        figsize:    figure size tuple (default: (16, 7))
        
    Returns:
        None
    """
    model.eval()

    # StatsBomb pitch dimensions
    config = GraphConfig()

    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white")
    
    # Handle both single graph and list of graphs
    if not isinstance(graphs, list):
        graphs = [graphs]

    
    for i, data in enumerate(graphs):
        data = Batch.from_data_list([data])
        data = data.to(device)

        with torch.no_grad():
            logits = model(data) # [1, NUM_CELLS] raw scores
            probs  = torch.softmax(logits.squeeze(0), dim=-1) # [NUM_CELLS] sum to 1, probability distribution over grid cells
            heatmap = probs.view(grid_y, grid_x).cpu().numpy() # reshape to 2D grid for visualization

        # Apply gaussian smoothing (does not affect argmax prediction) then get max probability cell as predicted pass destination
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        pred_cell        = heatmap.argmax()
        pred_iy, pred_ix = divmod(int(pred_cell), grid_x) # convert flat index to 2D grid indices
        # +0.5 to place marker at center of cell
        pred_x           = (pred_ix + 0.5) / grid_x * config.pitch_length
        pred_y           = (pred_iy + 0.5) / grid_y * config.pitch_width
        
        # Denormalize true pass end coordinates from [0,1] to pitch dimensions for plotting
        actor_idx = data.actor_idx.item()
        start_x = data.x[actor_idx, 0].item() * config.pitch_length
        start_y = data.x[actor_idx, 1].item() * config.pitch_width

        y_coords = data.y.view(-1)
        true_x = y_coords[0].item() * config.pitch_length
        true_y = y_coords[1].item() * config.pitch_width
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Player graph subplot
        pitch.draw(ax=ax1)
        ax1.set_title("Player Configuration", fontsize=14, fontweight="bold")
        ax1.set_xlim(0, config.pitch_length)
        ax1.set_ylim(0, config.pitch_width)
        ax1.set_aspect("equal") 

        # Convert PyG graph to NetworkX for drawing
        G = pyg_to_nx(data)

        # Denormalize node positions from [0,1] to pitch dimensions for plotting
        pos = {
            n: (
                G.nodes[n]['x'] * config.pitch_length,
                G.nodes[n]['y'] * config.pitch_width
            )
            for n in G.nodes()
        }
        
        # Draw edges with different styles based on relationship types
        edge_configs = [
            ("is_proximity", "gray",   "solid",  1),  # spatial neighbours
            ("is_pressure",  "orange", "dashed", 2),  # opponent pressing actor
            ("is_support",   "purple", "solid",  3),  # teammate passing options
        ]
        for condition, color, style, width in edge_configs:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d[condition]]
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, ax=ax1,
                edge_color=color, style=style, width=width, alpha=0.7
            )
        
        # Colour nodes by role: red = actor (passer), green = teammate, blue = opponent
        node_colors = [
            "red"   if G.nodes[n]["is_actor"]
            else "green" if G.nodes[n]["is_teammate"]
            else "blue"
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax1)
        
        # Heatmap subplot
        pitch.draw(ax=ax2)
        ax2.set_title("Predicted Pass Destination Heatmap", fontsize=14, fontweight="bold")
        ax2.set_xlim(0, config.pitch_length)
        ax2.set_ylim(0, config.pitch_width)
        ax2.set_aspect("equal")

        # Overlay heatmap
        im = ax2.imshow(
            heatmap,
            extent=[0, config.pitch_length, 0, config.pitch_width],
            origin="lower",
            cmap="YlOrRd",
            alpha=0.7,
            aspect="equal"
        )

        ax2.scatter(start_x, start_y,
            c="black", s=100, marker="o", edgecolors="black", linewidths=2,
            label="Pass start", zorder=5)

        ax2.scatter(true_x, true_y,
            c="lime", s=100, marker="*", edgecolors="darkgreen", linewidths=2,
            label="True end", zorder=5)

        ax2.scatter(pred_x, pred_y,
            c="blue", s=100, marker="*", edgecolors="darkblue", linewidths=2,
            label="Predicted end", zorder=5)

        # Arrow parameters for pass vectors
        arrow_kwargs = dict(head_width=2, head_length=2, alpha=0.7, linewidth=2)

        # True pass
        ax2.arrow(start_x, start_y,
            true_x - start_x, true_y - start_y,
            fc="black", ec="black", **arrow_kwargs)

        # Predicted pass
        ax2.arrow(start_x, start_y,
            pred_x - start_x, pred_y - start_y,
            fc="black", ec="black", **arrow_kwargs)
        
        ax2.legend(loc="upper right", fontsize=11)

        # Probability scale reference
        plt.colorbar(im, ax=ax2, label="Probability")

        # Title
        player_name = data.metadata["player_name"][0]
        match_id    = data.metadata["match_id"][0].item()

        fig.suptitle(
            f"Pass Prediction Analysis\n{player_name} | Match {match_id}",
            fontsize=16, fontweight="bold"
        )
        

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        player_slug = player_name.replace(" ", "_")
        filename = save_dir / f"{i:03d}_{player_slug}_match{match_id}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)