import torch
from torch_geometric.data import Dataset

class PassGraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        g = self.graphs[idx]

        # target: pass end location
        g.y = torch.tensor(g.end_location, dtype=torch.float)
        return g
