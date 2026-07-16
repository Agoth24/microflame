# microflame

[![PyPI](https://img.shields.io/pypi/v/microflame.svg)](https://pypi.org/project/microflame/)
[![Python](https://img.shields.io/pypi/pyversions/microflame.svg)](https://pypi.org/project/microflame/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal PyTorch training loop wrapper for reducing boilerplate.

## Install

```bash
pip install microflame
```

## Quickstart

```python
from torch import nn
from torch.optim import Adam
from microflame import Trainer

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))

trainer = Trainer(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=Adam(model.parameters(), lr=1e-3),
)

trainer.fit(num_epochs=10, save_path="checkpoint.pth", save_frequency=5)
trainer.plot()
```

Resume from a checkpoint:

```python
trainer.resume("checkpoint.pth", num_epochs=20)
```

## Features

- Automatic device selection (CUDA → MPS → CPU)
- Training + validation loop with live `tqdm` progress and per-epoch metrics
- Loss/accuracy history tracking
- Checkpoint save/load (model, optimizer, scheduler, history)
- Resume training from a checkpoint
- One-line loss/accuracy plots via matplotlib

## License

MIT © Agoth Arop — see [LICENSE](LICENSE).
