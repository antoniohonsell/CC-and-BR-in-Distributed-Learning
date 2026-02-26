import torch
from typing import List

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
    

def aggregate_trimmed_mean(vectors: List[torch.Tensor], trim_k: int) -> torch.Tensor:
    n = len(vectors)
    if n == 0:
        raise ValueError("No vectors to aggregate.")
    # safety: ensure feasibility of trimmed mean
    trim_k = min(trim_k, (n - 1) // 2)
    return _aggregate(vectors, defense="trimmed", trim_k=trim_k)  # uses your existing _aggregate