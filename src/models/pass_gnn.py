import torch.nn as nn
from torch_geometric.nn import GINEConv
from .config import NUM_CELLS

class PassPredictionGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_event_types: int,
        hidden_dim: int = 64,
        out_dim: int = 64
    ):
        super().__init__()

        # Event conditioning
        self.event_emb = nn.Embedding(num_event_types, hidden_dim)

        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # GNN layers
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ),
            edge_dim=edge_dim
        )

        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            ),
            edge_dim=edge_dim
        )

        # Predicting P(pass ends in cell i) for i = 1 ... N, where the pitch is discretized into a grid (GRID_X × GRID_Y)
        self.pass_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, NUM_CELLS)
        )

    def forward(self, data):
        x = self.node_proj(data.x)

        # Inject event embedding into actor node
        event_vec = self.event_emb(data.event_type_idx)

        # PyG's DataLoader merges all graphs in a batch into one giant graph
        # Each graph in the batch has its own set of node indices starting from 0, so we need to add the graph's start index (from data.ptr) to get the global node index in the batch.
        global_actor_idx = data.ptr[:-1] + data.actor_idx  # [B]

        # For batched graphs, data.actor_idx gives the index of the actor node for each graph in the batch, so we can directly add the event vector to those nodes.
        x = x.clone()
        x[global_actor_idx] = x[global_actor_idx] + event_vec

        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index, data.edge_attr)

        actor_emb = x[global_actor_idx]  # shape: [B, out_dim]
        
        # Actor node's embedding feeds into the head to predict pass location probabilities
        # Actor is the one making the decision, and the GNN layers have already aggregated neighborhood context (teammates, opponents, pressure) 
        logits = self.pass_head(actor_emb) # shape: [NUM_CELLS], unnormalized score for one pitch cell
        return logits

