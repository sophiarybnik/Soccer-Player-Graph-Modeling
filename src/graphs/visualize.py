import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
