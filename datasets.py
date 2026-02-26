import torch
from torchvision import datasets, transforms
from typing import Tuple
from torch.utils.data import DataLoader
from torch import nn


def gen_mask_indices(
    d: int, k: int, rng: torch.Generator, device: torch.device | None = None
) -> torch.Tensor:
    idx = torch.randperm(d, generator=rng)[:k]
    return idx.to(device) if device is not None else idx

def partition_data_non_IID_dirichlet_equal_exact(
    dataset,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    remainder: str = "drop",  # 'drop' | 'oversample' | 'error'
):
    """
    Exact-equal client sizes with Dirichlet class skew.
    - remainder='drop': drop N % K items (stratified) so each client has N//K.
    - remainder='oversample': duplicate a few items so each client has ceil(N/K).
    - remainder='error': raise if N not divisible by K.
    """
    import math
    import numpy as np
    from torch.utils.data import Subset

    # robust labels
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets, dtype=np.int64)
    elif hasattr(dataset, "train_labels"):
        labels = np.asarray(dataset.train_labels, dtype=np.int64)
    else:
        raise AttributeError("Dataset has no targets/train_labels")

    N = len(labels)
    K = num_clients
    C = int(labels.max()) + 1
    rng = np.random.RandomState(seed)

    # per-class pools (shuffled)
    class_to_indices = [np.where(labels == c)[0].tolist() for c in range(C)]
    for c in range(C):
        rng.shuffle(class_to_indices[c])

    # handle remainder to make total exactly target*K
    if remainder == "error":
        if N % K != 0:
            raise ValueError("N not divisible by num_clients; use remainder='drop' or 'oversample'.")
        target = N // K
        total = target * K
    elif remainder == "drop":
        target = N // K
        total = target * K
        r = N - total  # items to drop
        if r > 0:
            class_sizes = np.array([len(ix) for ix in class_to_indices], dtype=int)
            if class_sizes.sum() != N:
                raise RuntimeError("Internal size mismatch.")
            probs = np.where(class_sizes > 0, class_sizes / class_sizes.sum(), 0.0)
            drops = rng.multinomial(r, probs)
            # Guard: can't drop more than a class has
            drops = np.minimum(drops, class_sizes)
            dropped = int(drops.sum())
            # If still need to drop (due to min), drop uniformly from remaining pool
            if dropped < r:
                need = r - dropped
                pool = np.repeat(np.arange(C), class_sizes - drops)
                extra = rng.choice(pool, size=need, replace=False)
                for cc in extra:
                    drops[cc] += 1
            # apply drops
            for c in range(C):
                d = int(drops[c])
                if d:
                    class_to_indices[c] = class_to_indices[c][:-d]
    elif remainder == "oversample":
        target = math.ceil(N / K)
        total = target * K
        r = total - N  # items to add (duplicates)
        if r > 0:
            class_sizes = np.array([len(ix) for ix in class_to_indices], dtype=int)
            probs = np.where(class_sizes > 0, class_sizes / class_sizes.sum(), 0.0)
            adds = rng.multinomial(r, probs)
            for c in range(C):
                a = int(adds[c])
                if a > 0 and class_to_indices[c]:
                    extra = rng.choice(class_to_indices[c], size=a, replace=True)
                    class_to_indices[c].extend(extra.tolist())
    else:
        raise ValueError("remainder must be 'drop', 'oversample', or 'error'")

    # allocate with exact per-client capacity
    caps = np.full(K, target, dtype=int)
    client_indices = [[] for _ in range(K)]

    for c in range(C):
        idxs = class_to_indices[c]
        left = len(idxs)
        start = 0
        while left > 0:
            open_mask = caps > 0
            if not np.any(open_mask):
                raise RuntimeError(f"No capacity left while allocating class {c}.")
            open_ids = np.where(open_mask)[0]
            # Dirichlet proportions among clients that still have room
            props = rng.dirichlet([alpha] * len(open_ids))
            counts = rng.multinomial(left, props)
            counts = np.minimum(counts, caps[open_mask])
            assigned = int(counts.sum())

            if assigned == 0:
                # ensure progress: give as many as possible to the most-open client
                cid = open_ids[np.argmax(caps[open_mask])]
                give = min(left, caps[cid])
                client_indices[cid].extend(idxs[start:start + give])
                caps[cid] -= give
                start += give
                left -= give
            else:
                for cnt, cid in zip(counts, open_ids):
                    cnt = int(cnt)
                    if cnt:
                        client_indices[cid].extend(idxs[start:start + cnt])
                        caps[cid] -= cnt
                        start += cnt
                left -= assigned

    # final checks + shuffle within clients
    if not np.all(caps == 0):
        raise RuntimeError(f"Allocation error: leftover capacity {int(caps.sum())}.")
    for k in range(K):
        rng.shuffle(client_indices[k])
        assert len(client_indices[k]) == target, "Client size mismatch."

    return [Subset(dataset, client_indices[k]) for k in range(K)]

def get_mnist_datasets(data_dir: str = "./data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Downloads MNIST and returns (train_set, test_set) with standard transforms.
    Output shape: 1x28x28, normalized with dataset stats.
    """
    mean, std = (0.1307,), (0.3081,)

    train_transform = transforms.Compose([
        #transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.MNIST(root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    return train_set, test_set



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def make_plain_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

def _reset_bn(m: nn.Module):
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean.zero_()
        m.running_var.fill_(1)
        m.num_batches_tracked.zero_()

@torch.no_grad()
def recompute_bn_stats(model: nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       num_batches: int = 200) -> None:
    """
    Re-estimate BatchNorm running stats by streaming a few batches
    """
    model.apply(_reset_bn)
    was_training = model.training
    model.train()  # BN uses batch stats
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        seen += 1
        if seen >= num_batches:
            break
    model.train(was_training)


def get_cifar10_datasets(data_dir: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Downloads CIFAR-10 and returns (train_set, test_set) with standard transforms.
    """
    # CIFAR-10 channel-wise mean/std in [0,1]
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
    
     
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_set, test_set