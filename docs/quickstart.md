---
title: Quickstart
sidebar_title: Quickstart
---

This brief walkthrough takes you through central component of this library, the
`Trainer` class. We explain the basic usage in the order you'll typically use
it: install, set up, train, checkpoint, resume, plot.

## Installation

```bash
pip install microflame
```

## Prerequisites

Microflame doesn't replace the PyTorch training setup. Define training and
validation dataloaders, a model, a loss function, and pick an optimizer.

```python
from torch import nn
from torch.optim import SGD

training_data = # define your dataset
test_data # define your test set

# pass datasets into your dataloaders
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# an example model
model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))

# define your loss function
loss_fn = nn.CrossEntropyLoss()

# select an optimization algorithm
optimizer = SGD(model.parameters(), lr=1e-3)
```

## Instantiate a Trainer

```python
from microflame import Trainer

trainer = Trainer(
    train_dataloader=train_loader,
    val_dataloader=validation_loader,
    model=model,
    loss_fn=loss_function,
    optimizer=optimizer,
)
```

By default the device is picked automatically. Pass `device="cpu"` (or `"cuda"`,
`"mps"`) to override this option.

### Using a learning-rate scheduler

Learning rate schedulers are stepped once per epoch by default.

`step_scheduler_per_batch=True` allows for more frequent steps and updates the
learning rate scheduler after training every batch.

```python
from torch.optim.lr_scheduler import StepLR

lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

trainer = Trainer(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=lr_scheduler,
)
```

## Train the model with `fit()`

```python
trainer.fit(num_epochs=10)
```

During each epoch, we run a training pass and a validation pass, printing the
loss and accuracy for both.

Losses and accuracies are also recorded in a `trainer.history` dict for
plotting.

`fit()` also accepts:

- `save_path`: where checkpoints are written (default `"checkpoint.pth"`)
- `save_frequency`: save a checkpoint every N epochs (omit to never save)
- `start_epoch`: the epoch to start counting from (used internally by
  `resume()`)

```python
trainer.fit(num_epochs=10, save_path="checkpoint.pth", save_frequency=5)
```

## Checkpointing

`fit()` handles checkpointing for you via `save_frequency`, but you can also
save or load a checkpoint directly:

```python
trainer.save_checkpoint("checkpoint.pth", epoch=9)

last_epoch = trainer.load_checkpoint("checkpoint.pth")
```

Checkpoints store the model, optimizer, and scheduler state along with
`trainer.history`.

## Resume training partially trained models

`resume()` loads a checkpoint and continues `fit()` from the epoch right after
it:

```python
trainer.resume("checkpoint.pth", num_epochs=20)
```

## Plot results

```python
trainer.plot()
```

Renders loss and accuracy curves (train vs. validation) from `trainer.history`
using matplotlib.
