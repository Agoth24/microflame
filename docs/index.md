---
title: Microflame
sidebar_title: Home
external_links:
  "Github": https://github.com/agoth24/microflame
---

A PyTorch training loop, written once and no more. 

Microflame handles the training and validation loop, device placement, checkpointing, and plotting so you can focus on writing the model and not the boilerplate

!!! info ""
    Microflame is a very thin wrapper around PyTorch, and is in no way a replacement. If you are new to Pytorch, check out the tutorials [here](https://pytorch.org/tutorials/) first.

## Installation

```bash
pip install microflame
```


## Why Microflame?

PyTorch projects commonly end up with the same ~100 lines code in training & test loops.

Microflame encapsulates that code so you can focus on the model and data.

### What it offers
- Simplified training and validation with one class method
- Live training metrics, logged to the console
- Model checkpointing & portable training
- Automatic device hardware selection (CUDA → MPS → CPU)

## Example Usage

```python
from microflame import Trainer

trainer = Trainer(
    train_dataloader = train_loader,
    val_dataloader = val_loader,
    model = model,
    loss_fn = loss_fn,
    optimizer = optimizer,
)

trainer.fit(num_epochs=10, save_path="checkpoint.pth")
trainer.plot()
```

See [Quickstart](quickstart.md) for a full walkthrough.
