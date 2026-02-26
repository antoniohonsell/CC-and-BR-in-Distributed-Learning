import torch
from typing import List

def reconstruct_tilde_from_packets(packets: List[dict], d: int, k: int, device: torch.device) -> List[torch.Tensor]:
    """
    Applies step (4) to a list of packets and returns a list of \tilde{g}_i tensors.
    """
    return [decompress_and_scale(pkt, d=d, k=k, device=device) for pkt in packets]

def compress_with_mask(vec: torch.Tensor, idx: torch.Tensor) -> dict:
    # Ensure indices are on the same device as vec before using them.
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