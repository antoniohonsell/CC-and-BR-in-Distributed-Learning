import torch
from typing import List
import torch.nn as nn


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