"""Data loading, training and evaluation helpers for CIFAR-10.

The whole dataset (~600 MB as float32) is kept on the GPU and augmentation
(random crop + horizontal flip) is done there too, so training is limited by
the model, not by the DataLoader.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _to_tensor(ds, device):
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255)  # (N, 3, 32, 32)
    x = (x - MEAN) / STD
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


def load_cifar10(root='data', val_size=5000, seed=0, device='cuda'):
    """Returns (x, y) tensors for train / validation / test.

    The validation set is held out from the official training set and is used
    for model selection; the test set is only used for the final report.
    """
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=False)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=False)
    x, y = _to_tensor(train, device)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)).to(device)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    return {
        'train': (x[train_idx], y[train_idx]),
        'val': (x[val_idx], y[val_idx]),
        'test': _to_tensor(test, device),
    }


def augment(x, pad=4):
    """Random crop (zero padding) + random horizontal flip, per sample, on GPU."""
    n, c, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(h, device=x.device)).view(n, 1, h, 1)
    cols = (j.view(n, 1) + torch.arange(w, device=x.device)).view(n, 1, 1, w)
    return xp[torch.arange(n, device=x.device).view(n, 1, 1, 1),
              torch.arange(c, device=x.device).view(1, c, 1, 1), rows, cols]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, x, batch_size=1000):
    model.eval()
    return torch.cat([model(x[k:k + batch_size]).argmax(1)
                      for k in range(0, len(x), batch_size)])


def accuracy(model, x, y):
    return (predict(model, x) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    """Returns (mean cross-entropy loss, accuracy) without augmentation/dropout."""
    model.eval()
    loss, correct = 0.0, 0
    for k in range(0, len(x), batch_size):
        out = model(x[k:k + batch_size])
        yb = y[k:k + batch_size]
        loss += F.cross_entropy(out, yb, reduction='sum').item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss / len(y), correct / len(y)


def train(model, data, epochs=30, batch_size=128, optimizer='sgd', lr=0.05,
          weight_decay=5e-4, use_augment=True, seed=0, verbose=True):
    """Trains with a one-cycle LR schedule. Returns a history dict."""
    torch.manual_seed(seed)
    x_tr, y_tr = data['train']
    x_val, y_val = data['val']
    if optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              nesterov=True, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'unknown optimizer {optimizer!r}')
    steps = (len(y_tr) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss()

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=x_tr.device)
        tot_loss, correct = 0.0, 0
        for k in range(0, len(perm), batch_size):
            idx = perm[k:k + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            if use_augment:
                xb = augment(xb)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        hist['train_loss'].append(tot_loss / len(y_tr))
        hist['train_acc'].append(correct / len(y_tr))
        val_loss, val_acc = evaluate(model, x_val, y_val)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f'  ep {ep + 1:3d}  loss {hist["train_loss"][-1]:.3f}  '
                  f'train {hist["train_acc"][-1]:.3f}  val_loss {val_loss:.3f}  val {val_acc:.3f}', flush=True)
    hist['train_time_s'] = time.time() - t0
    return hist


def mcnemar(pred_a, pred_b, y):
    """Exact McNemar test on two classifiers' predictions over the same samples."""
    from scipy.stats import binomtest
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int(np.sum(a_ok & ~b_ok))   # only A right
    c = int(np.sum(~a_ok & b_ok))   # only B right
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {'only_a_correct': b, 'only_b_correct': c, 'p_value': p}


def bootstrap_diff(pred_a, pred_b, y, n=2000, seed=0):
    """95% bootstrap CI for acc(A) - acc(B) on the same test samples."""
    rng = np.random.default_rng(seed)
    d = (pred_a == y).astype(float) - (pred_b == y).astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
