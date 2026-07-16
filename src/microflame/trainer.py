import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: str | None = None,
        scheduler: lr_scheduler.LRScheduler | None = None,
        step_scheduler_per_batch: bool = False
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps" if (hasattr(torch, "mps") and torch.mps.is_available()) else "cpu"
            )
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_scheduler_per_batch = step_scheduler_per_batch
        self.loss_fn = loss_fn

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def accuracy(self, prediction, target):
        return (prediction.argmax(1) == target).float().mean().item()

    def _train_loop(self, current_epoch: int, num_epochs: int):
        dataloader = self.train_dataloader
        train_set_size = len(dataloader.dataset)
        num_batches = len(dataloader)

        train_loss, correct_preds = 0, 0
        loop = tqdm(
            dataloader, desc=f"Epoch [{current_epoch+1}/{num_epochs}]", leave=False
        )
        self.model.train()
        for X, y in loop:
            X, y = X.to(self.device), y.to(self.device)

            # forward pass
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            if self.scheduler and self.step_scheduler_per_batch:
                self.scheduler.step()

            train_loss += loss.item()
            correct_preds += self.accuracy(pred, y)

            loop.set_postfix(Loss=loss.item())

        train_loss /= num_batches
        train_acc = 100 * correct_preds / train_set_size
        return train_loss, train_acc

    def _val_loop(self, current_epoch: int, num_epochs: int):
        dataloader = self.val_dataloader
        val_set_size = len(dataloader.dataset)
        num_batches = len(dataloader)

        val_loss, correct_preds = 0, 0
        self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                val_loss += loss.item()
                correct_preds += self.accuracy(pred, y)

            val_loss /= num_batches
            val_acc = 100 * correct_preds / val_set_size

        return val_loss, val_acc

    def fit(
        self,
        num_epochs: int,
        save_path: str = "checkpoint.pth",
        save_frequency: int | None = None,
        start_epoch: int = 0,
    ):

        if start_epoch >= num_epochs:
            raise ValueError("Can't run training loop. Training is already complete")

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self._train_loop(
                current_epoch=epoch, num_epochs=num_epochs
            )
            if self.scheduler and not self.step_scheduler_per_batch:
                self.scheduler.step()
            val_loss, val_acc = self._val_loop(
                current_epoch=epoch, num_epochs=num_epochs
            )

            tqdm.write(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"loss={train_loss:.3f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)

            if save_frequency and (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(save_path, epoch)

        if save_frequency:
            self.save_checkpoint(save_path, epoch)

    def save_checkpoint(self, path: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # automated warning messages for loading w/ schedulers at checkpoints
        if checkpoint["scheduler_state_dict"] is not None and self.scheduler is None:
            print(
                "Warning: checkpoint has scheduler state but this Trainer has no scheduler — skipping."
            )
        elif self.scheduler is not None and checkpoint["scheduler_state_dict"] is None:
            print(
                "Warning: this Trainer has a scheduler but checkpoint has none — scheduler state not restored."
            )
        elif (
            self.scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint["history"]
        return checkpoint["epoch"]

    def resume(
        self,
        path: str,
        num_epochs: int,
        save_frequency: int | None = None,
        save_path: str = "checkpoint.pth",
    ):
        last_epoch = self.load_checkpoint(path)
        self.fit(
            num_epochs=num_epochs,
            start_epoch=last_epoch + 1,
            save_frequency=save_frequency,
            save_path=save_path,
        )

    def plot(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.history["train_loss"], label="Train loss")
        axes[0].plot(epochs, self.history["val_loss"], label="val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, self.history["train_accuracy"], label="Train accuracy")
        axes[1].plot(epochs, self.history["val_accuracy"], label="val accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
