import torch
import numpy as np


class Evaluator:
    def __init__(
        self,
        model,
        loss_fn,
        grid_x,
        grid_y,
        pitch_length=120.0,
        pitch_width=80.0,
        device="cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.device = device

        # precompute cell centre coordinates for centroid ADE [num_cells]
        ix = torch.arange(grid_x).float()
        iy = torch.arange(grid_y).float()
        grid_y_coords, grid_x_coords = torch.meshgrid(iy, ix, indexing="ij")
        self.cell_x = ((grid_x_coords + 0.5) / grid_x * pitch_length).reshape(-1)  # [num_cells]
        self.cell_y = ((grid_y_coords + 0.5) / grid_y * pitch_width).reshape(-1)

    def _cell_to_coords(self, ix, iy):
        """Convert grid cell indices to pitch coordinates (cell centre)."""
        x = (ix + 0.5) / self.grid_x * self.pitch_length
        y = (iy + 0.5) / self.grid_y * self.pitch_width
        return x, y

    def _true_coords(self, batch):
        """Extract true pass destination in pitch coordinates from batch.y."""
        y_coords = batch.y.view(-1, 2)
        true_x = y_coords[:, 0] * self.pitch_length
        true_y = y_coords[:, 1] * self.pitch_width
        return true_x, true_y

    def _true_cell(self, true_x, true_y):
        """Convert true pitch coords to the corresponding flat grid cell index."""
        ix = (true_x / self.pitch_length * self.grid_x).long().clamp(0, self.grid_x - 1)
        iy = (true_y / self.pitch_width  * self.grid_y).long().clamp(0, self.grid_y - 1)
        return iy * self.grid_x + ix  # flat index, shape [B]

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Run full evaluation over a DataLoader.

        Returns a dict with:
            loss          - mean cross-entropy loss
            ade           - average displacement error (metres)
            top1_acc      - fraction where argmax cell == true cell
            top3_acc      - fraction where true cell is in top-3 predicted cells
            top5_acc      - fraction where true cell is in top-5 predicted cells
            mean_rank     - mean rank of true cell in sorted probability distribution
            median_rank   - median rank of true cell
        """
        self.model.eval()

        total_loss = 0.0
        all_ade_argmax = []
        all_ade_centroid = []
        all_top1 = []
        all_top3 = []
        all_top5 = []
        all_ranks = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)                          # [B, num_cells]
            probs  = torch.softmax(logits, dim=-1)             # [B, num_cells]

            # --- loss ---
            loss = self.loss_fn(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs

            # --- true destination ---
            true_x, true_y = self._true_coords(batch)          # [B]
            true_cell = self._true_cell(true_x, true_y)        # [B]

            # --- argmax ADE ---
            pred_cell = probs.argmax(dim=-1)                    # [B]
            pred_iy, pred_ix = divmod(pred_cell.cpu().numpy(), self.grid_x)
            pred_x_argmax = torch.tensor(
                (pred_ix + 0.5) / self.grid_x * self.pitch_length
            )
            pred_y_argmax = torch.tensor(
                (pred_iy + 0.5) / self.grid_y * self.pitch_width
            )
            ade_argmax = torch.sqrt(
                (pred_x_argmax - true_x.cpu()) ** 2 + (pred_y_argmax - true_y.cpu()) ** 2
            )
            all_ade_argmax.extend(ade_argmax.tolist())

            # --- centroid ADE ---
            probs_cpu = probs.cpu()
            pred_x_centroid = (probs_cpu * self.cell_x).sum(dim=-1)  # [B]
            pred_y_centroid = (probs_cpu * self.cell_y).sum(dim=-1)
            ade_centroid = torch.sqrt(
                (pred_x_centroid - true_x.cpu()) ** 2 + (pred_y_centroid - true_y.cpu()) ** 2
            )
            all_ade_centroid.extend(ade_centroid.tolist())


            # --- cell accuracy ---
            pred_cell_cpu = pred_cell.cpu()
            true_cell_cpu = true_cell.cpu()
            all_top1.extend((pred_cell_cpu == true_cell_cpu).tolist())

            # --- top-k accuracy ---
            topk_indices = probs_cpu.topk(k=5, dim=-1).indices  # [B, 5]
            for i in range(batch.num_graphs):
                tc = true_cell_cpu[i].item()
                top5 = topk_indices[i].tolist()
                all_top3.append(tc in top5[:3])
                all_top5.append(tc in top5)

            # --- rank of true cell ---
            sorted_indices = probs_cpu.argsort(dim=-1, descending=True)  # [B, num_cells]
            for i in range(batch.num_graphs):
                tc = true_cell_cpu[i].item()
                rank = (sorted_indices[i] == tc).nonzero(as_tuple=True)[0].item() + 1
                all_ranks.append(rank)

        n = len(loader.dataset)
        return {
            "loss":         total_loss / n,
            "ade_argmax":   float(np.mean(all_ade_argmax)),
            "ade_centroid": float(np.mean(all_ade_centroid)),
            "top1_acc":     float(np.mean(all_top1)),
            "top3_acc":     float(np.mean(all_top3)),
            "top5_acc":     float(np.mean(all_top5)),
            "mean_rank":    float(np.mean(all_ranks)),
            "median_rank":  float(np.median(all_ranks)),
        }