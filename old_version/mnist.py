## we start with rosdhb on MNIST

import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
from typing import List, Tuple
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path

DATASET="MNIST" 
THR = 0.95 # accuracy threshold as a fraction e.g. 0.80
R_LIMIT = 250  # maximum number of rounds
PATH_NAME="." # directory where you want to save the pkl files 


def make_model(dataset: str) -> nn.Module:
    if dataset == "MNIST":
        return LightNet2()
    else:
        raise ValueError(f"Unknown DATASET={dataset}")


### function for better reproducibility
import os, random, numpy as np, torch
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True) 

# save function
def save(obj, filename):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(p) + ".pkl")




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




def iid_stratified_splits(dataset, num_clients: int = 10, seed: int = 42) -> List[Subset]:
    """
    Create IID (stratified) training splits so each client gets the same label distribution.
    Returns a list of torch.utils.data.Subset, one per client.
    """
    rng = random.Random(seed)

    targets = dataset.targets if hasattr(dataset, "targets") else dataset.train_labels
    num_classes = len(set(targets))

    # Build per-class index lists
    class_to_indices = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(targets):
        class_to_indices[int(y)].append(idx)

    # Shuffle each class's indices and split evenly into `num_clients` chunks
    per_client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = class_to_indices[c]
        rng.shuffle(idxs)
        base = len(idxs) // num_clients
        rem = len(idxs) % num_clients
        start = 0
        for client_id in range(num_clients):
            take = base + (1 if client_id < rem else 0)
            per_client_indices[client_id].extend(idxs[start:start+take])
            start += take

    # Wrap into Subset objects
    client_subsets = [Subset(dataset, sorted(ixs)) for ixs in per_client_indices]
    return client_subsets


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



### CNN for the MNIST dataset

class LightNet2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.fc1 = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (N, 6, 14, 14)
        x = torch.flatten(x, 1)               # (N, 1176)
        x = self.fc1(x)                       # logits (N, 10)
        return x


# ---- Flatten helpers ----
def params_to_vec(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def vector_to_params_(vec: torch.Tensor, model: nn.Module) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].view_as(p))
        offset += n


# ---- Mask & compression helpers ----
def gen_mask_indices(
    d: int, k: int, rng: torch.Generator, device: torch.device | None = None
) -> torch.Tensor:
    idx = torch.randperm(d, generator=rng)[:k]
    return idx.to(device) if device is not None else idx


def compress_with_mask(vec: torch.Tensor, idx: torch.Tensor) -> dict:
    idx = idx.to(device=vec.device, dtype=torch.long)
    vals = torch.index_select(vec, 0, idx).clone()
    return {"idx": idx, "values": vals}

def decompress_and_scale(packet: dict, d: int, k: int, device=None) -> torch.Tensor:
    """
    Reconstructs \tilde{g} = (d/k) * (g ⊙ mask) from C_k(g). 
    """
    idx = packet["idx"]
    vals = packet["values"]
    device = device or vals.device
    out = torch.zeros(d, device=device, dtype=vals.dtype)
    out[idx] = vals
    return (d / float(k)) * out

def reconstruct_tilde_from_packets(packets: List[dict], d: int, k: int, device: torch.device) -> List[torch.Tensor]:
    return [decompress_and_scale(pkt, d=d, k=k, device=device) for pkt in packets]




# ---- Honest client ----
class LocalClient:
    def __init__(self, train_subset: Subset, device: torch.device, batch_size: int = 128,
                 model_fn: Callable[[], nn.Module] = None):
        self.device = device
        self.model = (model_fn)().to(device)
        self.criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        self.criterion_train  = nn.CrossEntropyLoss(reduction="mean", label_smoothing= 0.0)
        self.criterion_metric = nn.CrossEntropyLoss(reduction="sum",  label_smoothing=0.0)
        num_workers = 2
        self.loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0)
        )

    def load_global_state(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    
    # local training
    def compute_local_update(
        self,
        E: int = 1,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        global_params_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float, int]:
        """
        Runs E local epochs of SGD and returns:
          Δθ_i (vector), mean loss, mean acc, dataset size n_i.
        """
        #momentum=0.0 # try with no momentum for local training
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)


        # snapshot of θ to measure Δθ at the end
        theta_start = params_to_vec(self.model).detach().clone()
        N = len(self.loader.dataset)
        crit = nn.CrossEntropyLoss(reduction="sum",label_smoothing=0.1)
        loss_sum = 0.0
        correct = 0
        seen = 0

        if DATASET=="MNIST":
            for _ in range(E):
                for x, y in self.loader:
                    opt.zero_grad(set_to_none=True)
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    #batch_loss_sum = crit(logits, y)  # sum over batch
                    #batch_mean_loss = batch_loss_sum / x.size(0)
                    batch_mean_loss = self.criterion_train(logits, y)  # no manual / batch_size needed

                    batch_mean_loss.backward()
                    opt.step()

                    #loss_sum += batch_loss_sum.item()
                    loss_sum += self.criterion_metric(logits, y).item()
                    correct  += (logits.argmax(1) == y).sum().item()
                    seen     += x.size(0)
                    break
                break # stops also if I put more than 1 epoch


        mean_loss = loss_sum / max(1, seen)
        mean_acc  = correct  / max(1, seen)

        theta_end = params_to_vec(self.model).detach()
        delta = theta_end - theta_start  # Δθ_i

        # (Optional) restore to θ_start to keep object stateless across rounds
        vector_to_params_(theta_start, self.model)

        return delta, mean_loss, mean_acc, N


# ============ DASHA PAGE CLIENT ===========

class DashaPageClient:
    def __init__(self, train_subset: Subset, device: torch.device,
                 batch_size: int = 128,
                 model_fn: Callable[[], nn.Module] = None):
        self.device = device
        model_ctor = model_fn 
        self.model = model_ctor().to(device)
        self.prev_model = model_ctor().to(device)  # to evaluate g(x_{t-1}; B)

        self.criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        self.loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=(device.type == "cuda")
        )
        self.reset_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=False
        )
        self.v_prev: torch.Tensor | None = None
        self.h: torch.Tensor | None = None
        self.g_i: torch.Tensor | None = None

    
    def dasha_pack_with_memory( ###Not used as well
        self,
        delta_vec: torch.Tensor,      # v_i^t := Δθ_i (your local update)
        mask_idx: torch.Tensor,
        d: int, k: int,
        alpha: float = 1.0,
    ) -> tuple[dict, torch.Tensor]:
        
        assert self.h is not None and self.h.numel() == d, "h_i not initialized or wrong size."
        u = delta_vec - self.h
        pkt = compress_with_mask(u, mask_idx)
        tilde_u = decompress_and_scale(pkt, d=d, k=k, device=delta_vec.device)
        self.h = self.h + alpha * tilde_u
        return pkt, tilde_u

    def set_prev_theta(self, theta_prev_vec: torch.Tensor) -> None:
        vector_to_params_(theta_prev_vec, self.prev_model)

    def _full_local_gradient_epoch(self, loader: DataLoader | None = None):
        loader = loader or self.loader
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        grad_sum = None
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            for p in self.model.parameters():
                if p.grad is not None: 
                    p.grad.zero_()
            logits = self.model(x)
            loss = self.criterion_sum(logits, y)
            loss.backward()
            flat = torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
            grad_sum = flat.clone() if grad_sum is None else grad_sum + flat
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        g_mean = grad_sum / float(total)
        return g_mean.detach(), total_loss / total, total_correct / total, total


    def _grad_on_batch(self, model_for_grad: nn.Module, x, y) -> torch.Tensor:
        model_for_grad.train()
        for p in model_for_grad.parameters():
            if p.grad is not None: p.grad.zero_()
        logits = model_for_grad(x)
        loss = self.criterion_sum(logits, y) / float(y.numel())
        loss.backward()
        return torch.cat([p.grad.detach().reshape(-1) for p in model_for_grad.parameters()]).clone()
        
    def local_train_and_delta_PAGE(
    self,
    theta_prev_vec: torch.Tensor,
    E: int,
    lr: float,
    p_reset: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    reset_batches: int = 0,   # <--- NEW (0 = full-epoch; >0 = average over R mini-batches)

    ):
        """
        Run E local epochs, but each update uses a PAGE estimator v.
        Returns (Δθ_i, mean_loss, mean_acc, N).
        """
        # start exactly from the broadcasted model, and cache prev global for PAGE
        vector_to_params_(theta_prev_vec, self.model)
        self.set_prev_theta(theta_prev_vec)
        self.v_prev = None # new addition 


        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        steps_per_epoch = max(1, len(self.loader))

        crit_sum = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1)
        self.model.train()

        # metrics across all seen samples
        N = len(self.loader.dataset)
        total_seen = 0
        loss_sum = 0.0
        correct = 0

        theta_start = params_to_vec(self.model).detach().clone()

        for _ in range(E): #####Need to break later 
            for xB, yB in self.loader:
                xB, yB = xB.to(self.device), yB.to(self.device)

                # ---- Flip the PAGE coin *per batch* ----
                do_reset = (self.v_prev is None) or (np.random.rand() < p_reset)
                if do_reset:
                    if reset_batches <= 0:
                        v = self._full_local_gradient_epoch(loader=self.reset_loader)[0]
                    else:
                        v = self._approx_full_grad(reset_batches)
                else:
                    # PAGE correction on this batch (same batch for both terms)
                    g_cur  = self._grad_on_batch(self.model,      xB, yB)
                    g_prev = self._grad_on_batch(self.prev_model, xB, yB)
                    v = self.v_prev + (g_cur - g_prev) / max(1e-8, (1.0 - p_reset))

                # Keep PAGE's "previous iterate" equal to *current* x_t for the NEXT batch
                #self.prev_model.load_state_dict(self.model.state_dict())
                # faster way:
                with torch.no_grad():
                    for p_prev, p in zip(self.prev_model.parameters(), self.model.parameters()):
                        p_prev.copy_(p) 


                # apply v as the gradient for this step
                opt.zero_grad(set_to_none=True)
                assign_flat_grad_(self.model, v)
                opt.step()

                self.v_prev = v.clone()

                # metrics (sum over samples to produce means later)
                with torch.no_grad():
                    logits = self.model(xB)
                    batch_loss_sum = crit_sum(logits, yB)  
                    loss_sum   += batch_loss_sum.item()
                    correct    += (logits.argmax(1) == yB).sum().item()
                    total_seen += yB.size(0)


        with torch.no_grad():
            theta_end = params_to_vec(self.model).detach()
            delta = theta_end - theta_start

        # restore to start to keep client “stateless” w.r.t. params
        vector_to_params_(theta_start, self.model)

        mean_loss = loss_sum / max(1, total_seen)
        mean_acc  = correct  / max(1, total_seen)
        return delta, mean_loss, mean_acc, total_seen
    
    def _approx_full_grad(self, R: int) -> torch.Tensor:
        # average gradients over R mini-batches
        seen = 0
        acc = None
        it = iter(self.reset_loader)
        for _ in range(R):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(self.reset_loader)
                x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)
            for p in self.model.parameters():
                if p.grad is not None: p.grad.zero_()
            logits = self.model(x)
            loss = self.criterion_sum(logits, y) / float(y.numel())
            loss.backward()
            g = torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
            acc = g.clone() if acc is None else acc + g
            seen += 1
        return acc / float(max(1, seen))
    
    # one of the last method added for Dasha

    def load_global_state(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
        if self.h is None:
            d = params_to_vec(self.model).numel()
            self.h = torch.zeros(d, device=self.device)
        if self.g_i is None:                       # <--- init g_i^0 = 0
            d = params_to_vec(self.model).numel()
            self.g_i = torch.zeros(d, device=self.device)



    def byz_dasha_page_message(
    self,
    theta_prev_vec: torch.Tensor,   # x^t
    g_global_vec: torch.Tensor,     # now interpreted as Δ^t (aggregated delta), not a gradient
    gamma: float,                   # server mixing for the delta field
    p_reset: float,                 # PAGE reset prob (used inside local training)
    a: float,                       # relaxation/momentum-style factor for memory
    mask_idx: torch.Tensor,
    d: int, k: int,
    *,                               # ---- local training hyperparams ----
    E: int = 1,
    lr: float = 0.1,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    reset_batches: int = 0,
    ) -> tuple[dict, torch.Tensor]:
        """
        Same Byz-DASHA-PAGE logic, but 'h_next' is now the client's local *delta* (θ_end - θ_start),
        computed with PAGE-driven local updates.
        Returns (compressed m_i^{t+1}, decompressed \tilde m_i^{t+1}).
        """
        # x^{t+1} = x^t + γ Δ^t
        # Alg.2 L6 (client applies broadcast): x^{t+1} = x^t + γ·Δ^t

        x_next = theta_prev_vec + gamma * g_global_vec
        vector_to_params_(x_next, self.model)
        #self.set_prev_theta(theta_prev_vec)  # keep x^t for PAGE correction in local training

        # ---- NEW: do local training with PAGE to get a delta (θ_end - x_next) ----
        # Byz Dasha Page L7–L11: run local PAGE to form the target signal
        delta, _, _, _ = self.local_train_and_delta_PAGE(
            theta_prev_vec=x_next,
            E=E,
            lr=lr,
            p_reset=p_reset,
            momentum=momentum,
            weight_decay=weight_decay,
            reset_batches=reset_batches,
        )

        # Interpret the locally-trained delta as "h_next" (the target signal to track)
        # set h_i^{t+1} := local delta produced by PAGE
        h_next = delta.detach()

        # DASHA-style message on deltas (unchanged math, just different signal):
        # m_i^{t+1} = Q( h_next - h_i^t - a (g_i^t - h_i^t) )
        # Byz Dasha Page L12: build message and compress with unbiased compressor Q_i
        diff = h_next - self.h - a * (self.g_i - self.h)
        pkt = compress_with_mask(diff, mask_idx)
        tilde_m = decompress_and_scale(pkt, d=d, k=k, device=self.device)

        # Keep client-side variables in sync with server:
        # g_i^{t+1} = g_i^t + m_i^{t+1};   h_i^{t+1} = h_next
        # Byz Dasha Page L13: update client’s running estimator (g_i) and memory (h_i)
        self.g_i = self.g_i + tilde_m
        self.h   = h_next


        return pkt, tilde_m

# ================== BYZANTINE CRAFTING  ==================

# ---- stats over honest vectors ----
def _honest_stats(H_vectors: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (avg_vector, var_vector) across honest gradients (shape [d]).
    """
    H = torch.stack(H_vectors, dim=0)  # [H, d]
    avg = H.mean(dim=0)
    var = H.var(dim=0, unbiased=False) + 1e-12  # eps for numerical stability
    return avg, var

# ---- simple robust aggregators to evaluate attack impact ----
def _aggregate(all_vectors: List[torch.Tensor], defense: str = "trimmed", trim_k: int = 0) -> torch.Tensor:
    """
    defense in {"mean", "median", "trimmed"}.
    For "trimmed", trim_k is the number of vectors to trim from each tail (per coordinate).
    """
    X = torch.stack(all_vectors, dim=0)  # [n, d]
    if defense == "mean":
        return X.mean(dim=0)
    elif defense == "median":
        return X.median(dim=0).values
    elif defense == "trimmed":
        # coordinate-wise sort then trim
        n = X.shape[0]
        k = max(0, min(trim_k, (n - 1) // 2))
        X_sorted, _ = torch.sort(X, dim=0)
        return X_sorted[k:n - k, :].mean(dim=0)
    else:
        raise ValueError(f"Unknown defense: {defense}")

# ---- choose coordinates to attack  ----
def compute_BMasks(
    H_vectors: List[torch.Tensor],
    k_percent: float,
    select_k_attack: str = "var",  # {"var","absmean","random"}
    BW_Num: int = 1,
    rng: Optional[torch.Generator] = None,
    restrict_to_mask_idx: Optional[torch.Tensor] = None,  # NEW
) -> List[torch.Tensor]:
    d = H_vectors[0].numel()
    device = H_vectors[0].device
    avg, var = _honest_stats(H_vectors)

    # Allowed pool of coordinates (default: all)
    allowed = torch.zeros(d, dtype=torch.bool, device=device)
    if restrict_to_mask_idx is None:
        allowed[:] = True
    else:
        allowed[restrict_to_mask_idx.to(device)] = True

    if k_percent >= 1.0:
        return [allowed.clone() for _ in range(BW_Num)]

    k_allowed = max(1, int(allowed.sum().item() * k_percent))
    scores = torch.full((d,), float("-inf"), device=device)
    if select_k_attack.lower() in ("var", "variance"):
        scores[allowed] = var[allowed]
    elif select_k_attack.lower() in ("absmean", "meanabs"):
        scores[allowed] = avg.abs()[allowed]
    elif select_k_attack.lower() == "random":
        if rng is None:
            rng = torch.Generator()          
            rng.manual_seed(0)
        count = int(allowed.sum().item())     
        r_cpu = torch.rand(count, generator=rng)    
        scores[allowed] = r_cpu.to(device)          
    else:
        raise ValueError("select_k_attack must be one of {'var','absmean','random'}")

    topk = torch.topk(scores, k=k_allowed, largest=True).indices
    mask = torch.zeros(d, dtype=torch.bool, device=device)
    mask[topk] = True
    return [mask.clone() for _ in range(BW_Num)]


# ---- instantiate Byzantine vectors for a given eta ----
def get_BVectors(
    H_vectors: List[torch.Tensor],
    BW_Type: str,
    BMasks: List[torch.Tensor],
    eta: float,
    avg_vector: torch.Tensor,
    var_vector: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Produces one malicious vector per mask.
    BW_Type in {"foe","random","alie","zero"}.
    The returned vectors are length-d tensors (same dtype/device as avg_vector).
    """
    B_vectors: List[torch.Tensor] = []
    d = avg_vector.numel()
    device = avg_vector.device
    for mask in BMasks:
        if BW_Type.lower() in ("foe", "scaled_neg_mean"):
            core = -eta * avg_vector
        elif BW_Type.lower() == "random":
            core = eta * torch.randn(d, device=device, dtype=avg_vector.dtype)
        elif BW_Type.lower() == "alie":
            core = avg_vector + eta * torch.sqrt(var_vector) * torch.randn(d, device=device, dtype=avg_vector.dtype)
        elif BW_Type.lower() == "zero":
            core = torch.zeros(d, device=device, dtype=avg_vector.dtype)
        else:
            raise ValueError("BW_Type must be one of {'foe','random','alie','zero'}")

        b = avg_vector.clone()
        b[mask] = core[mask]
        B_vectors.append(b)
    return B_vectors


def Compute_best_b_vector(
    H_vectors: List[torch.Tensor],
    avg_vector: torch.Tensor,
    var_vector: torch.Tensor,
    BW_Type: str,
    BW_Num: int,
    k_percent: float,
    select_k_attack: str,
    eta_range: List[float],
    defense: str = "trimmed",   
    NNM: Optional[int] = None,
    trim_k: int = 0,
    mask_idx: Optional[torch.Tensor] = None,   
    d: Optional[int] = None,                   
    k: Optional[int] = None,
    algo = "rosdhb"                   
) -> torch.Tensor:
    """
    Returns the crafted Byzantine vector for one attacker.
    The impact metric is computed on reconstructed masked vectors (\tilde g).
    """
    BW_Num = 1 

    if mask_idx is None or d is None or k is None:
        raise ValueError("For Rand-k, pass mask_idx, d and k.")
    
    if algo=="rosdhb":
        # Build a single mask covering the transmitted coords
        allowed_mask = torch.zeros(d, dtype=torch.bool, device=avg_vector.device)
        allowed_mask[mask_idx.to(allowed_mask.device).long()] = True
        BMasks = [allowed_mask] 

    elif algo=="byz_dasha_page":
        BMasks = compute_BMasks(
        H_vectors, k_percent, select_k_attack, BW_Num, restrict_to_mask_idx=mask_idx
    )


    # honest, after Rand-k masking + scaling
    honest_masked = []
    for h in H_vectors:
        pkt = compress_with_mask(h, mask_idx)
        honest_masked.append(decompress_and_scale(pkt, d=d, k=k, device=h.device))
    honest_agg = _aggregate(honest_masked, defense=defense, trim_k=trim_k)

    if len(eta_range) == 1:
        B_vectors = get_BVectors(H_vectors, BW_Type, BMasks, eta_range[0], avg_vector, var_vector)
        return B_vectors[0]

    # sweep eta: maximize deviation from the honest aggregated direction (post Rand-k)
    best_vec = None
    max_distance = -float("inf")
    for eta in eta_range:
        B_vectors = get_BVectors(H_vectors, BW_Type, BMasks, eta, avg_vector, var_vector)
        # reconstruct masked Byzantine vector too
        B_masked = []
        for b in B_vectors:
            pkt_b = compress_with_mask(b, mask_idx)
            B_masked.append(decompress_and_scale(pkt_b, d=d, k=k, device=b.device))
        agg_with_byz = _aggregate(honest_masked + B_masked, defense=defense, trim_k=trim_k)
        distance = torch.norm(honest_agg - agg_with_byz, p=2)
        if distance > max_distance:
            max_distance = distance
            best_vec = B_vectors[0]
    return best_vec

# ---- helper: craft N Byzantine packets ----

def craft_byzantine_packets(
    H_vectors: List[torch.Tensor],
    mask_idx: torch.Tensor,
    num_byzantine: int,
    BW_Type: str = "foe",
    eta_range: List[float] = (10.0,),   
    k_percent: float = 1.0,
    select_k_attack: str = "var",
    defense: str = "trimmed",   
    trim_k: int = 0,
) -> tuple[List[dict], List[torch.Tensor]]:
    avg_vector, var_vector = _honest_stats(H_vectors)
    d = avg_vector.numel()
    k = mask_idx.numel()

    byz_packets: List[dict] = []
    byz_full: List[torch.Tensor] = []
    eta_list = list(eta_range) if isinstance(eta_range, (list, tuple)) else [float(eta_range)]

    for _ in range(num_byzantine):
        b_vec = Compute_best_b_vector(
            H_vectors=H_vectors,
            avg_vector=avg_vector,
            var_vector=var_vector,
            BW_Type=BW_Type,
            BW_Num=1,
            k_percent=k_percent,
            select_k_attack=select_k_attack,
            eta_range=eta_list,
            defense=defense,
            NNM=None,
            trim_k=trim_k,
            mask_idx=mask_idx,  
            d=d, k=k,
            algo="rosdhb",
        )
        byz_full.append(b_vec)
        byz_packets.append(compress_with_mask(b_vec, mask_idx))
    return byz_packets, byz_full



# ========= MOMENTUM =========

class MomentumBank:
    def __init__(self, d: int, device: torch.device, beta: float = 0.9, dtype: torch.dtype = torch.float32):
        self.d = d
        self.device = device
        self.beta = beta
        self.dtype = dtype
        self._state: dict[int, torch.Tensor] = {}  

    def update_one(self, client_id: int, tilde_g: torch.Tensor) -> torch.Tensor:
        m_prev = self._state.get(
            client_id,
            torch.zeros(self.d, device=self.device, dtype=tilde_g.dtype if tilde_g is not None else self.dtype),
        )
        m_new = self.beta * m_prev + (1.0 - self.beta) * tilde_g
        self._state[client_id] = m_new
        return m_new

    def update_batch(self, client_ids: List[int], tilde_list: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(client_ids) == len(tilde_list)
        return [self.update_one(cid, g) for cid, g in zip(client_ids, tilde_list)]



# Another bank used for the serve side for Dasha:
class GradBank:
    def __init__(self, num_clients: int, d: int, device: torch.device, dtype=torch.float32):
        self._g = [torch.zeros(d, device=device, dtype=dtype) for _ in range(num_clients)]

    def add_message(self, client_id: int, tilde_m: torch.Tensor) -> torch.Tensor:
        self._g[client_id] = self._g[client_id] + tilde_m
        return self._g[client_id]

    def all_g(self) -> List[torch.Tensor]:
        return self._g
    
# The following is used for Dasha algorithm         
def assign_flat_grad_(model: nn.Module, grad_vec: torch.Tensor) -> None:
    """Write a flat gradient vector into model.parameters().grad in-place."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        g = grad_vec[offset:offset + n].view_as(p)
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(g)
        offset += n


# Evalutation function.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    crit_sum = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    loss_sum = 0.0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit_sum(logits, y).item()
        correct  += (logits.argmax(1) == y).sum().item()
        total   += x.size(0)
    return loss_sum / float(total), correct / float(total)

# --- aggregation with trimmed mean ---
def aggregate_trimmed_mean(vectors: List[torch.Tensor], trim_k: int) -> torch.Tensor:
    n = len(vectors)
    if n == 0:
        raise ValueError("No vectors to aggregate.")
    
    trim_k = min(trim_k, (n - 1) // 2)
    return _aggregate(vectors, defense="trimmed", trim_k=trim_k)  


# ---- MAIN CLASS ----

@dataclass
class Config:
    rounds: int = 3
    num_clients: int = 10          
    num_byzantine: int = 0         
    keep_ratio: float = 0.02       
    server_lr: float = 0.2         
    weight_decay: float = 0
    beta: float = 0.9              # momentum coefficient
    batch_size: int = 128
    seed: int = 123
    byz_type: str = "foe"
    byz_eta_range: tuple = (5.0, 10.0, 20.0)
    byz_k_percent: float = 1.0
    byz_select: str = "var"
    label_smoothing: float = 0.0
    eval_every: int = 1
    local_epochs: int = 1 
    client_lr: float = 0.5
    client_momentum: float = 0.0
    client_frac: float = 1.0  
    glob: bool = True
    # specific for dasha
    page_p: float = 0.1         # reset probability p in PAGE
    a_momentum: float = 0.9        # 'a' of Dasha
    page_reset_batches: int = 0   # 0 => exact full-epoch; >0 => average over this many mini-batches




# ---- MAIN ----    

def main():
    parser = argparse.ArgumentParser(description="Runnign Byzantine codes")
    parser.add_argument(
        "--kps",
        type=float,
        required=True,
        help="compression ratio k/d",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="learning rate",
    )
    parser.add_argument(
        "--byz",
        type=int,
        required=True,
        help="Byzantine numbers",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help="rosdhb or byz_dasha_page",
    )
    kps=[parser.parse_args().kps]
    lrs = [parser.parse_args().lr] 
    ALGO = parser.parse_args().algo
    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            print("mps")
            return torch.device("mps")   
        elif torch.cuda.is_available():
            return torch.device("cuda")  
        else:
            return torch.device("cpu")

    print(DATASET)

    device=get_device() 

    num_clients = 10
    split_seed = 777    


    if DATASET=='MNIST':

        train_set, test_set = get_mnist_datasets("./data")
        
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=(device.type=="cuda"))

        train_eval_set = datasets.MNIST("./data", train=True, download=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
        train_eval_loader = DataLoader(train_eval_set, batch_size=256, shuffle=False,
                                    num_workers=2, pin_memory=(device.type=="cuda"))
        
        client_train_subsets_fixed = partition_data_non_IID_dirichlet_equal_exact(
        train_set, num_clients=num_clients, alpha=5, seed=split_seed)
    
    seeds = [42, 123, 202, 256, 777] 
    num_byz= parser.parse_args().byz
    rounds_totrsh = {kp: {lr: [] for lr in lrs} for kp in kps}  




    for kp in kps:
        for lrr in lrs:
            print("This is the lr",lrr)
            total_test_acc={s: [] for s in seeds}
            total_train_acc={s: [] for s in seeds}
            for run_seed in seeds:
                
                
                client_momentum=0
                weight_decay=0
                local_epochs=1


                if ALGO=="rosdhb" :
            
                    cfg = Config(
                        rounds=R_LIMIT,
                        num_clients=10,
                        num_byzantine=num_byz,   
                        keep_ratio=kp, 
                        server_lr=0.8,
                        beta=0.9,
                        batch_size=6000,
                        seed=run_seed,
                        byz_type="foe",
                        byz_eta_range=[0.1*(i+1) for i in range(50)],
                        byz_k_percent=1.0,
                        byz_select="var",
                        eval_every=1,
                        client_lr=lrr,
                        client_momentum=client_momentum,
                        glob=True, # Global sparsification
                        weight_decay=weight_decay,
                        local_epochs=local_epochs
                    )

                    # ---- Repro ----
                    seed_everything(cfg.seed)
                    client_train_subsets = client_train_subsets_fixed  
                    clients = [
                        LocalClient(
                            subset,
                            device=device,
                            batch_size=cfg.batch_size,
                            model_fn=lambda: make_model(DATASET)  
                        )
                        for subset in client_train_subsets]
                    assert len(clients) == cfg.num_clients, "Built wrong number of clients"

                    

                    
                    # ---- Server init ----
                    global_model = make_model(DATASET).to(device)
                    theta = params_to_vec(global_model).detach()
                    d = theta.numel() # dimension of the vector in my model 

                    momentum_bank = MomentumBank(d=d, device=device, beta=cfg.beta)

                    print(f"[Init] d={d:,} params | clients: {cfg.num_clients} honest + {cfg.num_byzantine} byz | keep_ratio={cfg.keep_ratio}")

                    # ---- Training rounds ----
                    hit_round = None 
                    best_acc = 0.0

                    for t in range(1, cfg.rounds + 1):
                        # Step 1/2: sample Rand-k mask & broadcast θ
                        k = max(1, int(d * cfg.keep_ratio))
                        rng = torch.Generator()  
                        rng.manual_seed(cfg.seed + t)
                        mask_idx = gen_mask_indices(d, k, rng, device=device) # creation of global mask 
                        state_to_broadcast = global_model.state_dict()
                        theta_prev = params_to_vec(global_model).detach()


                        # Step 3: honest clients do E local epochs and send masked \Delta\theta_i
                        client_packets: List[dict] = []
                        honest_updates: List[torch.Tensor] = []
                        client_metrics: List[tuple[int, float, float]] = []
                        client_sizes: List[int] = []

                        for cid, client in enumerate(clients):
                            # broadcast latest global weights to the persistent client
                            client.load_global_state(state_to_broadcast)

                            delta_i, loss_mean, acc_mean, n_i = client.compute_local_update(
                                E=getattr(cfg, "local_epochs", 1),
                                lr=getattr(cfg, "client_lr", 0.1),
                                momentum=getattr(cfg, "client_momentum", 0.9),
                                weight_decay=getattr(cfg, "weight_decay", 0.0),
                                global_params_vec=theta_prev,
                            )

                            packet = compress_with_mask(delta_i, mask_idx)
                            honest_updates.append(delta_i)
                            client_packets.append(packet)
                            client_metrics.append((cid, loss_mean, acc_mean))
                            client_sizes.append(n_i)


                        avg_loss = sum(m[1] for m in client_metrics) / len(client_metrics)
                        avg_acc  = sum(m[2] for m in client_metrics) / len(client_metrics)
                        print(f"[Round {t}] Honest packets: {len(client_packets)} | mean loss={avg_loss:.4f} | mean acc={avg_acc:.4f}")

                        # Byzantine clients
                        byz_packets: List[dict] = []
                        if cfg.num_byzantine > 0:
                            byz_packets, _ = craft_byzantine_packets(
                                H_vectors=honest_updates,
                                mask_idx=mask_idx,
                                num_byzantine=cfg.num_byzantine,
                                BW_Type=cfg.byz_type,
                                eta_range=list(cfg.byz_eta_range),
                                k_percent=cfg.byz_k_percent,
                                select_k_attack=cfg.byz_select,
                                defense="trimmed",
                                trim_k=cfg.num_byzantine,
                            )
                            print(f"[Round {t}] Byzantine packets added: {len(byz_packets)}")

                        # Step 4: reconstruct \tilde{g} (masked + scaled) for all
                        all_packets = client_packets + byz_packets
                        tilde_all = reconstruct_tilde_from_packets(all_packets, d=d, k=k, device=device)

                        # Build client ids: honest [0..N-1], then byz [N..N+B-1]
                        all_ids = list(range(cfg.num_clients)) + list(range(cfg.num_clients, cfg.num_clients + len(byz_packets)))

                        # Step 5: per-client momentum update
                        m_list = momentum_bank.update_batch(all_ids, tilde_all)

                        # Step 6: aggregate momenta 
                        trim_k = min(cfg.num_byzantine, (len(m_list) - 1) // 2)
                        R_t = aggregate_trimmed_mean(m_list, trim_k=trim_k)

                        # Step 7: Server update (aggregate \Delta\theta)
                        lr_t=cfg.server_lr
                        theta_new = theta_prev + lr_t * R_t   # add aggregated \Delta\theta
                        vector_to_params_(theta_new, global_model)

                        # Evaluation
                        if (t % cfg.eval_every) == 0:
                            train_loss, train_acc = evaluate(global_model, train_eval_loader, device)
                            test_loss, test_acc = evaluate(global_model, test_loader, device)
                            print(f"[Eval t={t:3d}] train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%")

                            total_train_acc[run_seed].append(train_acc)
                            total_test_acc[run_seed].append(test_acc)

                            best_acc = max(best_acc, test_acc)
                            if hit_round is None and test_acc >= THR:
                                hit_round = t
                                break  # stop training early for this (kp, lr, seed)


                elif ALGO == "byz_dasha_page":
                    cfg = Config(
                        rounds=R_LIMIT,
                        num_clients=10,
                        num_byzantine=num_byz,   
                        keep_ratio=kp, 
                        server_lr=0.8, 
                        a_momentum=1/(2*(1/kps[0]) - 1), # optimzed value in the thoery of byz dasha page
                        batch_size=6000,
                        seed=run_seed,
                        byz_type="foe",
                        byz_eta_range=[0.1*(i+1) for i in range(50)],
                        byz_k_percent=1.0,
                        byz_select="var",
                        eval_every=1,
                        client_lr=lrr,
                        client_momentum=0,
                        # >>>> THE DASHA-SPECIFIC BITS:
                        page_p=1,       
                        page_reset_batches=0       
                    )
                    client_train_subsets=client_train_subsets_fixed
                    page_clients = [
                        DashaPageClient(subset, device=device, batch_size=cfg.batch_size,
                                        model_fn=lambda: make_model(DATASET))  
                        for subset in client_train_subsets
                    ]
                    global_model = make_model(DATASET).to(device)
                    theta = params_to_vec(global_model).detach()
                    d = theta.numel()

                    grad_bank = GradBank(num_clients=cfg.num_clients + cfg.num_byzantine, d=d, device=device)
                    g_global = torch.zeros(d, device=device)
                    hit_round = None 
                    best_acc = 0.0

                    for t in range(1, cfg.rounds + 1):
                        # Byz Dasha Page L2: server enters round t
                        theta_prev = params_to_vec(global_model).detach()  # x^t
                        theta_next = theta_prev + cfg.server_lr * g_global
                        # Byz Dasha Page L6: apply server step x^{t+1}=x^{t}−γ g^{t}
                        vector_to_params_(theta_next, global_model)

                        # Rand-k budget per client (k can be same for all; masks are different)
                        k = max(1, int(cfg.keep_ratio * d))

                        honest_msgs = []  # store \tilde m_i for optional Byzantine crafting

                        # ---- HONEST CLIENTS: local Rand-k mask per client ----
                        for cid, client in enumerate(page_clients):
                            client.load_global_state(global_model.state_dict())

                            # Byz Dasha Page L4: broadcast (x^{t}, g^{t}) to clients

                            # deterministic per-(t,cid) RNG for reproducibility
                            rng_i = torch.Generator()
                            rng_i.manual_seed(cfg.seed * 9973 + t * 101 + cid)
                            mask_idx_i = gen_mask_indices(d, k, rng_i, device=device)


                            pkt_i, tilde_m_i = client.byz_dasha_page_message(
                                theta_prev_vec=theta_prev,
                                g_global_vec=g_global,
                                gamma=cfg.server_lr,
                                p_reset=cfg.page_p,
                                a=cfg.a_momentum,
                                mask_idx=mask_idx_i, d=d, k=k,
                                E=cfg.local_epochs,
                                lr=cfg.client_lr,
                                momentum=cfg.client_momentum,
                                weight_decay=cfg.weight_decay,
                                reset_batches=cfg.page_reset_batches,
                            )
                            grad_bank.add_message(cid, tilde_m_i)
                            honest_msgs.append(tilde_m_i)

                        # ---- (Optional) BYZANTINE CLIENTS: their own masks + attacked coords ----
                        if cfg.num_byzantine > 0:
                            avg_vec, var_vec = _honest_stats(honest_msgs)
                            for b in range(cfg.num_byzantine):
                                rng_b = torch.Generator()
                                rng_b.manual_seed(cfg.seed * 123457 + t * 991 + 10000 + b)
                                mask_idx_b = gen_mask_indices(d, k, rng_b, device=device)

                                # craft best attack constrained to this byzantine's mask
                                b_vec = Compute_best_b_vector(
                                    H_vectors=honest_msgs,
                                    avg_vector=avg_vec,
                                    var_vector=var_vec,
                                    BW_Type=cfg.byz_type,              # e.g., "foe"
                                    BW_Num=1,
                                    k_percent=cfg.byz_k_percent,        # fraction within allowed coords
                                    select_k_attack="var",
                                    eta_range=list(cfg.byz_eta_range),
                                    defense="trimmed",
                                    trim_k=cfg.num_byzantine,
                                    mask_idx=mask_idx_b, d=d, k=k,
                                    algo="byz_dasha_page",
                                )
                                # emulate compressed channel and server-side reconstruction
                                byz_pkt = compress_with_mask(b_vec, mask_idx_b)
                                tilde_b = decompress_and_scale(byz_pkt, d=d, k=k, device=device)
                                grad_bank.add_message(cfg.num_clients + b, tilde_b)
                                # Byz Dasha Page L14: client sends m_i^{t+1} to server

                        # === Aggregate and update global delta ===
                        all_g = grad_bank.all_g()
                        trim_k = min(cfg.num_byzantine, (len(all_g) - 1) // 2)
                        g_global = aggregate_trimmed_mean(all_g, trim_k=trim_k)
                        # Byz Dasha Page L16: robust aggregation  to form g^{t+1}

                        # --- Evaluate
                        if (t % cfg.eval_every) == 0:
                            train_loss, train_acc = evaluate(global_model, train_eval_loader, device)
                            test_loss,  test_acc  = evaluate(global_model, test_loader, device)
                            print(f"[Eval t={t:3d}] train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%")
                            
                            total_train_acc[run_seed].append(train_acc)
                            total_test_acc[run_seed].append(test_acc)

                            best_acc = max(best_acc, test_acc)
                            if hit_round is None and test_acc >= THR:
                                hit_round = t
                                break

                # store best acc for this (kp, lrr, seed)
                """
                results[kp][lrr].append(best_acc)
                print(f"[DONE] keep_ratio={kp} | lr={lrr} | seed={run_seed} -> best_acc={best_acc:.4f} | rounds_needed={rounds_needed}")
                """
                if hit_round is None:
                    # didn’t reach 0.75 within cfg.rounds; keep info but mark as "no hit"
                    print(f"[DONE] keep_ratio={kp} | lr={lrr} | seed={run_seed-121} -> no hit in {cfg.rounds} rounds (best_acc={best_acc:.4f})"
                          f"(best_acc={best_acc*100:.2f}%)")
                    # store a sentinel > cfg.rounds so we can compute both "mean over hits" and "capped mean"
                    rounds_totrsh[kp][lrr].append(cfg.rounds + 1)
                else:
                    print(f"[DONE] keep_ratio={kp} | lr={lrr} | seed={run_seed-121} -> hit in {hit_round} rounds (best_acc={best_acc:.4f})"
                          f"(best_acc={best_acc*100:.2f}%)")
                    rounds_totrsh[kp][lrr].append(hit_round)


    

            fname = os.path.join(
                PATH_NAME, "train",
                f"train_{ALGO}_MNIST_{num_byz}_cr{kp}_bz{cfg.batch_size}_non_iid"
            )
            save(total_train_acc, fname)

            fname = os.path.join(
                PATH_NAME, "test",
                f"test_{ALGO}_MNIST_{num_byz}_cr{kp}_bz{cfg.batch_size}_non_iid"
            )
            save(total_test_acc, fname)

            lr_stats = []
            for lrr in lrs:
                vals = rounds_totrsh[kp][lrr]
                hits_only = [r for r in vals if r <= R_LIMIT]
                if hits_only:
                    lr_stats.append((lrr, float(np.mean(hits_only))))
            if lr_stats:
                best_lr, best_mean = min(lr_stats, key=lambda x: x[1])
                print(f"  -> best lr by mean rounds: {best_lr:.3f} (mean={best_mean:.1f})")

        
    


    # === FINAL SUMMARY ===
    print("\n=== ROUNDS-TO-THRESHOLD SUMMARY (THR = {:.2f}) ===".format(THR))
    for kp in kps:
        print(f"\nkeep_ratio={kp:.2f}")
        for lr in lrs:
            vals = rounds_totrsh[kp][lr]                
            total = len(vals)
            hits_only = [r for r in vals if r <= R_LIMIT]
            hits = len(hits_only)

            if hits > 0:
                mean_rounds_hits = float(np.mean(hits_only))
                capped_mean = float(np.mean([min(r, R_LIMIT) for r in vals]))
                print(f"  lr={lr:.3f}: mean_rounds_to_{int(THR*100)}% = {mean_rounds_hits:.1f} "
                    f"(hits {hits}/{total}), capped_mean={capped_mean:.1f}")
            else:
                print(f"  lr={lr:.3f}: no seed reached {int(THR*100)}% within {R_LIMIT} rounds (0/{total})")


if __name__ == "__main__":
    main()






