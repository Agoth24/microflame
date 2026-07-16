---
title: Quickstart
sidebar_title: Quickstart
---

This walks through everything `Trainer` exposes, in the order you'll actually use it: install, set up, train, checkpoint, resume, plot.

## Installation

```bash
pip install microflame
```

## Set up your training pieces

Microflame doesn't replace any of your usual PyTorch objects — bring a model, a loss function, an optimizer, and two `DataLoader`s.

```python
from torch import nn
from torch.optim import Adam

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# train_loader, val_loader = ... your DataLoaders
```

## Create a `Trainer`

```python
from microflame import Trainer

trainer = Trainer(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
)
```

By default the device is picked automatically (CUDA → MPS → CPU). Pass `device="cpu"` (or `"cuda"`, `"mps"`) to override it.

### Using a learning-rate scheduler

Pass a `scheduler` and it's stepped once per epoch by default. Set `step_scheduler_per_batch=True` to step it after every batch instead.

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

trainer = Trainer(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
)
```

## Train with `fit()`

```python
trainer.fit(num_epochs=10)
```

Each epoch runs a training pass and a validation pass, printing loss and accuracy for both. Losses and accuracies are also recorded on `trainer.history` for later plotting.

`fit()` also accepts:

- `save_path` — where checkpoints are written (default `"checkpoint.pth"`)
- `save_frequency` — save a checkpoint every N epochs (omit to never save)
- `start_epoch` — the epoch to start counting from (used internally by `resume()`)

```python
trainer.fit(num_epochs=10, save_path="checkpoint.pth", save_frequency=5)
```

## Checkpointing

`fit()` handles checkpointing for you via `save_frequency`, but you can also save or load a checkpoint directly:

```python
trainer.save_checkpoint("checkpoint.pth", epoch=9)

last_epoch = trainer.load_checkpoint("checkpoint.pth")
```

A checkpoint stores the model, optimizer, and scheduler state along with `trainer.history`.

## Resume training

`resume()` loads a checkpoint and continues `fit()` from the epoch right after it:

```python
trainer.resume("checkpoint.pth", num_epochs=20)
```

## Plot results

```python
trainer.plot()
```

Renders loss and accuracy curves (train vs. validation) from `trainer.history` using matplotlib.
