class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(batch)

            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)
