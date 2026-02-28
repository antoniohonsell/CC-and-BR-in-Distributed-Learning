from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from compressor import reconstruct_tilde_from_packets, compress_with_mask, decompress_and_scale
from utils import make_model, seed_everything, save, params_to_vec, vector_to_params_, evaluate
from byzantine_crafting import _honest_stats, Compute_best_b_vector, craft_byzantine_packets
from aggregator import aggregate_trimmed_mean
from momentum_bank import MomentumBank, GradBank
from clients_code import LocalClient, DashaPageClient
from datasets import get_cifar10_datasets, make_plain_transform, partition_data_non_IID_dirichlet_equal_exact, recompute_bn_stats, gen_mask_indices
from typing import List
from dataclasses import dataclass


DATASET="CIFAR-10" 
THR = 0.95 # accuracy threshold as a fraction e.g. 0.80
R_LIMIT = 250  # maximum number of rounds
PATH_NAME="." # directory where you want to save the pkl files 

@dataclass
class Config:
    rounds: int = 3
    num_clients: int = 10          # honest
    num_byzantine: int = 0         # set >0 to enable attackers
    keep_ratio: float = 0.02       # Rand-k fraction
    server_lr: float = 0.2         
    weight_decay: float = 0
    beta: float = 0.9              # momentum coefficient for rosdhb
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
    glob: bool = True
    # new for dasha
    a_momentum: float = 0.9        # 'a' of Dasha
    algo: str = "rosdhb"        # {"rosdhb", "byz_dasha_page"}
    page_p: float = 0.1         # reset probability p in PAGE
    page_reset_batches: int = 0   # 0 => exact full-epoch; >0 => average over this many mini-batches
    local_total_steps: int | None = None   # None = full epochs, 1 = single batch


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

    if DATASET=='CIFAR-10':
        train_set, test_set = get_cifar10_datasets("./data")
        bn_calib_set = datasets.CIFAR10("./data", train=True, download=False, transform=make_plain_transform())
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=(device.type=="cuda"))
        
        train_eval_set = datasets.CIFAR10("./data", train=True, download=False, transform=make_plain_transform())
        train_eval_loader = DataLoader(train_eval_set, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=(device.type=="cuda"))
        
        bn_calib_loader = DataLoader(bn_calib_set, batch_size=512, shuffle=False,
                                            num_workers=2, pin_memory=(device.type == "cuda"))
    


    split_seed = 777     # fixed partition across all runs
    num_clients = 10
    

        
        
    if DATASET == 'CIFAR-10':
        client_train_subsets_fixed = partition_data_non_IID_dirichlet_equal_exact(
            train_set, num_clients=num_clients, alpha=5, seed=split_seed)

    kps=[parser.parse_args().kps] 
    lrs = [parser.parse_args().lr] 
    seeds = [42, 123, 202, 256, 777] 
    num_byz= parser.parse_args().byz
    rounds_totrsh = {kp: {lr: [] for lr in lrs} for kp in kps}  

    print(ALGO)

    for kp in kps:
        for lrr in lrs:
            print("This is the lr",lrr)
            total_test_acc={s: [] for s in seeds}
            total_train_acc={s: [] for s in seeds}
            for run_seed in seeds:
                

                client_momentum=0.0
                weight_decay=0.0
                local_epochs=1
                
                if ALGO=="rosdhb":

                    cfg = Config(
                        rounds=R_LIMIT,
                        num_clients=10,
                        num_byzantine=num_byz,   
                        keep_ratio=kp, 
                        server_lr=0.8,
                        beta=0.9,
                        batch_size=128,
                        seed=run_seed,
                        byz_type="foe",
                        byz_eta_range=[0.1*(i+1) for i in range(50)],
                        byz_k_percent=1.0,
                        byz_select="var",
                        eval_every=2,
                        client_lr=lrr,
                        client_momentum=client_momentum,
                        glob=True, # Global sparsification
                        weight_decay=weight_decay,
                        local_epochs=local_epochs
                    )

                    # ---- Repro ----
                    seed_everything(cfg.seed)
                    client_train_subsets = client_train_subsets_fixed  # fixed partition
                    # Build persistent clients ONCE per (kp, lrr, run_seed) run
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
                        rng = torch.Generator()  # cpu generator is fine for indices
                        rng.manual_seed(cfg.seed + t)
                        mask_idx = gen_mask_indices(d, k, rng, device=device) # creation of global mask 
                        state_to_broadcast = global_model.state_dict()
                        theta_prev = params_to_vec(global_model).detach() # new addition


                        # Step 3: honest clients do E local epochs and send masked Δθ_i
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

                        # Step 6: aggregate momenta (trimmed mean). Typically trim_k ≈ num_byzantine.
                        trim_k = min(cfg.num_byzantine, (len(m_list) - 1) // 2)
                        R_t = aggregate_trimmed_mean(m_list, trim_k=trim_k)

                        # Step 7: Server update (aggregate Δθ, not gradients)
                        # Let R_t be the aggregated parameter delta Δθ (e.g., avg(local_theta - theta_prev)).
                        # Then: θ^t = θ^{t-1} + γ · R_t
                        lr_t=cfg.server_lr
                        theta_new = theta_prev + lr_t * R_t   # add aggregated Δθ
                        vector_to_params_(theta_new, global_model)

                        # Evaluation
                        if (t % cfg.eval_every) == 0:
                            # Recompute BN stats from a few batches of the (non-augmented) training data if on CIFAR10
                            recompute_bn_stats(global_model, bn_calib_loader, device, num_batches=200) 
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
                    server_lr=lrr, 
                    a_momentum=1/(2*(1/kps[0]) - 1),
                    batch_size=128,
                    seed=run_seed,
                    byz_type="foe",
                    byz_eta_range=[0.1*(i+1) for i in range(50)],
                    byz_k_percent=1.0,
                    byz_select="var",
                    eval_every=2,
                    client_lr=1.0,
                    client_momentum=0,
                    # >>>> THE DASHA-SPECIFIC BITS:
                    algo="byz_dasha_page",
                    page_p=128/5000, # optmized value found in the theory         
                    page_reset_batches=0,      
                    local_total_steps=1, 
                )

                # ---- Repro ----
                seed_everything(cfg.seed)
                client_train_subsets = client_train_subsets_fixed  # <-- fixed partition
                
                # ---- Server init ----
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                global_model = make_model(DATASET).to(device)   
                theta = params_to_vec(global_model).detach()
                d = theta.numel()       

                print(f"[Init] d={d:,} params | clients: {cfg.num_clients} honest + {cfg.num_byzantine} byz | keep_ratio={cfg.keep_ratio}")

                # ---- Training rounds ----
                hit_round = None  # will hold the first round where test_acc >= THR
                best_acc = 0.0

                if cfg.algo == "byz_dasha_page":
                    page_clients = [
                        DashaPageClient(subset, device=device, batch_size=cfg.batch_size,
                                        model_fn=lambda: make_model(DATASET))  # <--- important
                        for subset in client_train_subsets
                    ]

                    theta = params_to_vec(global_model).detach()
                    d = theta.numel()

                    grad_bank = GradBank(num_clients=cfg.num_clients + cfg.num_byzantine, d=d, device=device)
                    g_global = torch.zeros(d, device=device)




                    for t in range(1, cfg.rounds + 1):
                        # Byz Dasha Page L2: for t = 0, …, T−1
                        theta_prev = params_to_vec(global_model).detach()  # x^t
                        # Byz Dasha Page L6 x^{t+1} = x^t − γ·g^t (actually x^{t+1} = x^t + γ·Δ^t, beacuse we work with Δs)
                        theta_next = theta_prev + cfg.server_lr * g_global
                        vector_to_params_(theta_next, global_model)

                        # Rand-k budget per client (k can be same for all; masks are different)
                        k = max(1, int(cfg.keep_ratio * d))

                        honest_msgs = []  # store \tilde m_i for optional Byzantine crafting

                        # ---- HONEST CLIENTS: local Rand-k mask per client ----
                        global_reset = (t == 1) or (np.random.rand() < cfg.page_p)
                        p_reset=None


                        for cid, client in enumerate(page_clients):

                            # Byz Dasha Page L4: Broadcast g^t (and θ) to all nodes
                            client.load_global_state(global_model.state_dict())

                            # deterministic per-(t,cid) RNG for reproducibility
                            rng_i = torch.Generator()
                            rng_i.manual_seed(cfg.seed * 9973 + t * 101 + cid)
                            mask_idx_i = gen_mask_indices(d, k, rng_i, device=device)

                            pkt_i, tilde_m_i = client.byz_dasha_page_message(
                                theta_prev_vec=theta_prev,
                                g_global_vec=g_global,
                                gamma=cfg.server_lr,
                                p_reset=None,       # we could remove this 
                                global_reset=global_reset,
                                a=cfg.a_momentum,
                                mask_idx=mask_idx_i, d=d, k=k,
                                E=cfg.local_epochs,
                                lr=cfg.client_lr,
                                momentum=cfg.client_momentum,
                                weight_decay=cfg.weight_decay,
                                reset_batches=cfg.page_reset_batches,
                                total_steps=cfg.local_total_steps, # NEW
                            )
                            # Byz Dasha Page L14: Send m_i^{t+1} to the server
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
                                    k_percent=1.0,                     # fraction within allowed coords
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

                        # === Aggregate and update global delta ===
                        all_g = grad_bank.all_g()
                        trim_k = min(cfg.num_byzantine, (len(all_g) - 1) // 2)
                        # Byz Dasha Page L16: g^{t+1} = ARAgg(g_1^{t+1}, …, g_n^{t+1})
                        g_global = aggregate_trimmed_mean(all_g, trim_k=trim_k)

                        # --- Evaluate
                        if (t % cfg.eval_every) == 0:
                            recompute_bn_stats(global_model, bn_calib_loader, device, num_batches=200) # in the end this was setto 200
                            train_loss, train_acc = evaluate(global_model, train_eval_loader, device)
                            test_loss,  test_acc  = evaluate(global_model, test_loader, device)
                            print(f"[Eval t={t:3d}] train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}%")
                            
                            total_train_acc[run_seed].append(train_acc)
                            total_test_acc[run_seed].append(test_acc)

                            best_acc = max(best_acc, test_acc)
                            if hit_round is None and test_acc >= THR:
                                hit_round = t
                                break


                if hit_round is None:
                    # didn’t reach Treshold within cfg.rounds; keep info but mark as "no hit"
                    print(f"[DONE] keep_ratio={kp} | lr={lrr} | seed={run_seed-121} -> no hit in {cfg.rounds} rounds (best_acc={best_acc:.4f})"
                        f"(best_acc={best_acc*100:.2f}%)")
                    rounds_totrsh[kp][lrr].append(cfg.rounds + 1)
                else:
                    print(f"[DONE] keep_ratio={kp} | lr={lrr} | seed={run_seed-121} -> hit in {hit_round} rounds (best_acc={best_acc:.4f})"
                        f"(best_acc={best_acc*100:.2f}%)")
                    rounds_totrsh[kp][lrr].append(hit_round)


            fname = os.path.join(
                        PATH_NAME, "train",
                        f"train_{ALGO}_CIFAR-10_{num_byz}_cr{kp}_bz{cfg.batch_size}_non_iid"
                    )
            save(total_train_acc, fname)

            fname = os.path.join(
                        PATH_NAME, "test",
                        f"test_{ALGO}_CIFAR-10_{num_byz}_cr{kp}_bz{cfg.batch_size}_non_iid"
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





