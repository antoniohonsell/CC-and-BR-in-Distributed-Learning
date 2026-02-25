
import torch
import torch.nn as nn
from pathlib import Path
from .architectures import make_resnet18_cifar10, LightNet2
from torch.utils.data import DataLoader
from typing import List


def make_model(dataset: str) -> nn.Module:
    if dataset == "CIFAR-10":
        return make_resnet18_cifar10()
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


# ---- Flatten helpers ----
def params_to_vec(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def vector_to_params_(vec: torch.Tensor, model: nn.Module) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].view_as(p))
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


def reconstruct_tilde_from_packets(packets: List[dict], d: int, k: int, device: torch.device) -> List[torch.Tensor]:
    """
    Applies step (4) to a list of packets and returns a list of \tilde{g}_i tensors.
    """
    return [decompress_and_scale(pkt, d=d, k=k, device=device) for pkt in packets]

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