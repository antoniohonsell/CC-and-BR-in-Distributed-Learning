import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable, Optional
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
    def __init__(
        self,
        train_subset: Subset,
        device: torch.device,
        batch_size: int = 128,
        model_fn: Callable[[], nn.Module] = None,
    ):
        self.device = device
        model_ctor = model_fn
        if model_ctor is None:
            raise ValueError("DashaPageClient requires model_fn")

        self.model = model_ctor().to(device)
        self.prev_model = model_ctor().to(device)  # snapshot for g(x_old; B)

        self.criterion_sum = nn.CrossEntropyLoss(reduction="sum")
        self.loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        self.reset_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
        )

        # PAGE / DASHA state
        self.v_prev: Optional[torch.Tensor] = None
        self.h: Optional[torch.Tensor] = None
        self.g_i: Optional[torch.Tensor] = None
        self.h_grad: Optional[torch.Tensor] = None  # PAGE memory in gradient-domain (strict 1-batch path)

    def set_prev_theta(self, theta_prev_vec: torch.Tensor) -> None:
        vector_to_params_(theta_prev_vec, self.prev_model)

    def _full_local_gradient_epoch(self, loader: Optional[DataLoader] = None):
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
            if p.grad is not None:
                p.grad.zero_()

        logits = model_for_grad(x)
        loss = self.criterion_sum(logits, y) / float(y.numel())
        loss.backward()
        return torch.cat([p.grad.detach().reshape(-1) for p in model_for_grad.parameters()]).clone()

    def _approx_full_grad(self, R: int) -> torch.Tensor:
        """
        Approximate full gradient by averaging gradients over R mini-batches
        from reset_loader.
        """
        ns = 0
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
                if p.grad is not None:
                    p.grad.zero_()

            logits = self.model(x)
            loss = self.criterion_sum(logits, y)  # sum over batch
            loss.backward()

            g = torch.cat([p.grad.detach().reshape(-1) for p in self.model.parameters()])
            acc = g.clone() if acc is None else acc + g
            ns += y.numel()

        return acc / float(max(1, ns))

    def load_global_state(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

        d = params_to_vec(self.model).numel()
        if self.h is None:
            self.h = torch.zeros(d, device=self.device)
        if self.g_i is None:
            self.g_i = torch.zeros(d, device=self.device)
        if self.h_grad is None:
            self.h_grad = torch.zeros(d, device=self.device)

    def local_train_and_delta_PAGE(
        self,
        theta_prev_vec: torch.Tensor,
        E: int,
        lr: float,
        p_reset: Optional[float],          # may be None if using global_reset
        global_reset: Optional[bool],      # if not None: shared coin for the whole round
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reset_batches: int = 0,
        total_steps: Optional[int] = None,
    ):
        """
        Rebuttal-style PAGE local training:
        - If global_reset is provided, use it as the reset decision (shared coin).
        - Otherwise, use per-step coin with probability p_reset.
        - Returns delta = theta_end - theta_start and basic metrics.
        """
        vector_to_params_(theta_prev_vec, self.model)
        self.set_prev_theta(theta_prev_vec)
        self.v_prev = None

        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        crit_sum = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1)
        self.model.train()

        theta_start = params_to_vec(self.model).detach().clone()
        loss_sum = 0.0
        correct = 0
        total_seen = 0

        def one_step(xB, yB):
            nonlocal loss_sum, correct, total_seen

            did_reset = (self.v_prev is None) or (
                global_reset if global_reset is not None else (np.random.rand() < float(p_reset or 0.0))
            )

            # estimator at current point
            if did_reset:
                v_used = (
                    self._full_local_gradient_epoch(loader=self.reset_loader)[0]
                    if reset_batches == 0
                    else self._approx_full_grad(max(1, reset_batches))
                )
            else:
                v_used = self.v_prev

            # snapshot current params into prev_model BEFORE stepping
            with torch.no_grad():
                for p_prev, p in zip(self.prev_model.parameters(), self.model.parameters()):
                    p_prev.copy_(p)

            # x_{s+1} = x_s - lr * v_used  (implemented via opt.step on assigned grads)
            opt.zero_grad(set_to_none=True)
            assign_flat_grad_(self.model, v_used)
            opt.step()

            # PAGE correction computed on the SAME batch
            g_new = self._grad_on_batch(self.model, xB, yB)
            g_old = self._grad_on_batch(self.prev_model, xB, yB)

            if did_reset:
                self.v_prev = (
                    self._full_local_gradient_epoch(loader=self.reset_loader)[0]
                    if reset_batches == 0
                    else self._approx_full_grad(max(1, reset_batches))
                )
            else:
                self.v_prev = v_used + (g_new - g_old)

            # metrics
            with torch.no_grad():
                logits = self.model(xB)
                loss_sum += crit_sum(logits, yB).item()
                correct += (logits.argmax(1) == yB).sum().item()
                total_seen += yB.size(0)

        # Drive steps
        if total_steps is None:
            for _ in range(E):
                for xB, yB in self.loader:
                    xB, yB = xB.to(self.device), yB.to(self.device)
                    one_step(xB, yB)
        else:
            it = iter(self.loader)
            for _ in range(int(total_steps)):
                try:
                    xB, yB = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    xB, yB = next(it)
                xB, yB = xB.to(self.device), yB.to(self.device)
                one_step(xB, yB)

        theta_end = params_to_vec(self.model).detach()
        delta = theta_end - theta_start
        vector_to_params_(theta_start, self.model)  # keep stateless

        mean_loss = loss_sum / max(1, total_seen)
        mean_acc = correct / max(1, total_seen)
        return delta, mean_loss, mean_acc, total_seen

    def one_batch_PAGE_to_delta(
        self,
        theta_prev_vec: torch.Tensor,   # x^t
        theta_next_vec: torch.Tensor,   # x^{t+1}
        global_reset: bool,
        reset_batches: int = 0,
        lr_for_delta: float = 0.1,
    ) -> tuple[torch.Tensor, int]:
        """
        Strict 1-batch PAGE update in gradient-domain, then map gradient-estimator -> parameter delta.
        """
        vector_to_params_(theta_next_vec, self.model)
        self.set_prev_theta(theta_prev_vec)

        if global_reset:
            h_next_grad = (
                self._full_local_gradient_epoch(loader=self.reset_loader)[0]
                if reset_batches == 0
                else self._approx_full_grad(max(1, reset_batches))
            )
            seen = 0
        else:
            it = iter(self.loader)
            try:
                xB, yB = next(it)
            except StopIteration:
                it = iter(self.loader)
                xB, yB = next(it)

            xB, yB = xB.to(self.device), yB.to(self.device)
            g_new = self._grad_on_batch(self.model, xB, yB)       # ∇f_B(x^{t+1})
            g_old = self._grad_on_batch(self.prev_model, xB, yB)  # ∇f_B(x^{t})

            if self.h_grad is None:
                self.h_grad = torch.zeros_like(g_new)

            h_next_grad = self.h_grad + (g_new - g_old)
            seen = yB.size(0)

        self.h_grad = h_next_grad.detach()
        delta = -lr_for_delta * h_next_grad.detach()
        return delta, seen

    def byz_dasha_page_message(
        self,
        theta_prev_vec: torch.Tensor,   # x^t
        g_global_vec: torch.Tensor,     # interpreted as Δ^t (aggregated delta), not a gradient
        gamma: float,
        p_reset: Optional[float],       # may be None if using global_reset
        global_reset: Optional[bool],   # shared round coin if provided
        a: float,                       # DASHA relaxation factor
        mask_idx: torch.Tensor,
        d: int,
        k: int,
        *,
        E: int = 1,
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reset_batches: int = 0,
        total_steps: Optional[int] = None,
    ) -> tuple[dict, torch.Tensor]:
        """
        Rebuttal-style Byz-DASHA-PAGE client:
        - Supports shared reset coin (global_reset) and strict 1-batch mode (total_steps == 1),
          matching old_version/cifar10.py.
        """
        if self.h is None or self.g_i is None:
            raise RuntimeError("Call load_global_state(...) before byz_dasha_page_message(...)")

        # x^{t+1} = x^t + gamma * Δ^t
        x_next = theta_prev_vec + gamma * g_global_vec

        # Decide reset if caller didn't provide it (keeps MNIST path working)
        if global_reset is None:
            global_reset_effective = (self.h_grad is None) or (np.random.rand() < float(p_reset or 0.0))
        else:
            global_reset_effective = bool(global_reset)

        # Strict 1-batch PAGE (no local SGD), or multi-step local PAGE
        if total_steps == 1:
            delta, _ = self.one_batch_PAGE_to_delta(
                theta_prev_vec=theta_prev_vec,
                theta_next_vec=x_next,
                global_reset=global_reset_effective,
                reset_batches=reset_batches,
                lr_for_delta=lr,
            )
            h_next = delta.detach()
        else:
            delta, _, _, _ = self.local_train_and_delta_PAGE(
                theta_prev_vec=x_next,
                E=E,
                lr=lr,
                p_reset=p_reset,
                global_reset=global_reset_effective,
                momentum=momentum,
                weight_decay=weight_decay,
                reset_batches=reset_batches,
                total_steps=total_steps,
            )
            h_next = delta.detach()

        # DASHA message on deltas
        diff = h_next - self.h - a * (self.g_i - self.h)
        pkt = compress_with_mask(diff, mask_idx)
        tilde_m = decompress_and_scale(pkt, d=d, k=k, device=self.device)

        # Update client memory
        self.g_i = self.g_i + tilde_m
        self.h = h_next

        return pkt, tilde_m