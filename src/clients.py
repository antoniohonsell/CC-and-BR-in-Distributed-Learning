import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable
from utils import assign_flat_grad_, vector_to_params_, params_to_vec
from compressor import compress_with_mask, decompress_and_scale
from architectures import make_resnet18_cifar10

# ---- Honest client (gradient-only; no local updates) ----
class LocalClient:
    def __init__(self, train_subset: Subset, device: torch.device, batch_size: int = 128,
                 model_fn: Callable[[], nn.Module] = None):
        self.device = device
        self.model = (model_fn or make_resnet18_cifar10)().to(device)  # default CIFAR
        # Use sum reduction so we can compute a true dataset mean
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


    def make_masked_message_from_gradient(self, grad_vec: torch.Tensor, mask_idx: torch.Tensor) -> dict:
        return compress_with_mask(grad_vec, mask_idx)
    
    # new function added for local training
    def compute_local_update(
        self,
        E: int = 1,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        global_params_vec: torch.Tensor | None = None,
        DATASET: str = "CIFAR-10",
    ) -> tuple[torch.Tensor, float, float, int]:
        """
        Runs E local epochs of SGD and returns:
          Δθ_i (vector), mean loss, mean acc, dataset size n_i.
        """
        #momentum=0.0 # try with no momentum for local training
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        
        # addition for users scheduler lr
        steps_per_epoch = max(1, len(self.loader))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=E * steps_per_epoch, eta_min=lr * 0.1
        )


        # snapshot of θ to measure Δθ at the end
        theta_start = params_to_vec(self.model).detach().clone()
        N = len(self.loader.dataset)
        crit = nn.CrossEntropyLoss(reduction="sum",label_smoothing=0.1)
        loss_sum = 0.0
        correct = 0
        seen = 0
        if DATASET=="CIFAR-10":
            for _ in range(E):
                for x, y in self.loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    #batch_loss_sum = crit(logits, y)  # sum over batch
                    #batch_mean_loss = batch_loss_sum / x.size(0)
                    batch_mean_loss = self.criterion_train(logits, y)  # no manual / batch_size needed

                    opt.zero_grad(set_to_none=True)
                    batch_mean_loss.backward()
                    opt.step()
                    #scheduler.step()

                    #loss_sum += batch_loss_sum.item()
                    loss_sum += self.criterion_metric(logits, y).item()
                    correct  += (logits.argmax(1) == y).sum().item()
                    seen     += x.size(0)
                    break
                break


        mean_loss = loss_sum / max(1, seen)
        mean_acc  = correct  / max(1, seen)

        theta_end = params_to_vec(self.model).detach()
        delta = theta_end - theta_start  # Δθ_i

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