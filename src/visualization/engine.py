from pathlib import Path
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Batch
from mplsoccer import Pitch

from src.visualization.heatmaps import (
    logits_to_heatmap,
    smooth_heatmap,
    get_pred_coordinates,
)

from src.graphs.config import GraphConfig
from src.graphs.visualize import pyg_to_nx
from src.inference.predict import predict_single



def plot_prediction_heatmap(
    graphs,
    model,
    grid_x,
    grid_y,
    save_dir,
    device="cpu",
    sigma=1.0,
    figsize=(16, 7),
):
    """
    Visualize player graph + predicted pass heatmap side-by-side
    """

    model.eval()
    config = GraphConfig()
    pitch = Pitch(pitch_type="statsbomb", line_color="black", pitch_color="white")

    if not isinstance(graphs, list):
        graphs = [graphs]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(graphs):
        batch = Batch.from_data_list([data]).to(device)

        probs = predict_single(model, batch, device)

        # --- Heatmap processing ---
        heatmap = logits_to_heatmap(probs, grid_x, grid_y)
        heatmap = smooth_heatmap(heatmap, sigma)

        pred_x, pred_y = get_pred_coordinates(
            heatmap,
            grid_x,
            grid_y,
            config.pitch_length,
            config.pitch_width,
        )

        # --- True + start positions ---
        actor_idx = batch.actor_idx.item()

        start_x = batch.x[actor_idx, 0].item() * config.pitch_length
        start_y = batch.x[actor_idx, 1].item() * config.pitch_width

        y_coords = batch.y.view(-1)
        true_x = y_coords[0].item() * config.pitch_length
        true_y = y_coords[1].item() * config.pitch_width

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ===== LEFT: PLAYER GRAPH =====
        pitch.draw(ax=ax1)
        ax1.set_title("Player Configuration", fontweight="bold")

        G = pyg_to_nx(batch)

        pos = {
            n: (
                G.nodes[n]["x"] * config.pitch_length,
                G.nodes[n]["y"] * config.pitch_width,
            )
            for n in G.nodes()
        }

        edge_configs = [
            ("is_proximity", "gray", "solid", 1),
            ("is_pressure", "orange", "dashed", 2),
            ("is_support", "purple", "solid", 3),
        ]

        for condition, color, style, width in edge_configs:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d[condition]]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                ax=ax1,
                edge_color=color,
                style=style,
                width=width,
                alpha=0.7,
            )

        node_colors = [
            "red"
            if G.nodes[n]["is_actor"]
            else "green"
            if G.nodes[n]["is_teammate"]
            else "blue"
            for n in G.nodes()
        ]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax1)

        # ===== RIGHT: HEATMAP =====
        pitch.draw(ax=ax2)
        ax2.set_title("Predicted Pass Heatmap", fontweight="bold")

        im = ax2.imshow(
            heatmap,
            extent=[0, config.pitch_length, 0, config.pitch_width],
            origin="lower",
            cmap="YlOrRd",
            alpha=0.7,
        )

        # Points
        ax2.scatter(start_x, start_y, c="black", s=100, label="Start")
        ax2.scatter(true_x, true_y, c="lime", s=100, marker="*", label="True")
        ax2.scatter(pred_x, pred_y, c="blue", s=100, marker="*", label="Pred")

        # Arrows
        ax2.arrow(start_x, start_y, true_x - start_x, true_y - start_y)
        ax2.arrow(start_x, start_y, pred_x - start_x, pred_y - start_y)

        ax2.legend()
        plt.colorbar(im, ax=ax2)

        # --- Title ---
        player_name = batch.metadata["player_name"][0]
        match_id = batch.metadata["match_id"][0].item()

        fig.suptitle(f"{player_name} | Match {match_id}")

        # --- Save ---
        filename = save_dir / f"{i:03d}_{player_name.replace(' ', '_')}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)