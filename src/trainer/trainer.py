import torch
from tqdm import tqdm


class Trainer:
    """
    This trainer suits the following paradigms:
    - Normal training
    - Curriculum learning
    - Hypernetwork training
    """

    def __init__(self, model, criterion, optimizer, cfg,
                 train_loader, epochs, logger=None, val_loader=None, lr_scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """

        :type device: object
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.device = device

    def train(self):
        for epoch in tqdm(range(self.cfg.trainer.epochs)):
            train_loss = self._train_epoch(epoch)
            # valid_loss = self._valid_epoch(epoch)
            print(f"Epoch: {epoch}, Train Loss: {train_loss}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        total_loss = (total_loss / len(self.train_loader))
        if self.lr_scheduler:
            self.lr_scheduler.step(total_loss)
        self._progress(epoch, total_loss)
        return total_loss

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(self.val_loader):
                output = self.model(data)
                loss = self.criterion(output, data)  # Assuming data is the target
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _progress(self, epoch, loss):
        if self.logger:
            self.logger.log_metric("train_loss", loss, step=epoch)


if __name__ == '__main__':
    pass
