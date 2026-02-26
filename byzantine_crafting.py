import torch
from typing import List, Optional
from compressor import compress_with_mask, decompress_and_scale
from aggregator import _aggregate

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

