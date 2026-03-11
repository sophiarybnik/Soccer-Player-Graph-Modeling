import torch

class Evaluator:
    def __init__(
        self,
        model,
        loss_fn,
        device="cpu"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)

            pred = self.model(batch)

            loss = self.loss_fn(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)
