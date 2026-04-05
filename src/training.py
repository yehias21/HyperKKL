"""
Unified training module for Phase 1 (autonomous) and Phase 2 (non-autonomous).

Phase 2 methods:
    - augmented:  frozen T/T*, learn phi injection into observer ODE
    - full:       ResidualHyperNetwork generates full weight residuals for T/T*
    - lora:       PerLayerLoRAHyperNetwork generates per-layer LoRA deltas

Encoder type (lstm/gru) is a config parameter, not part of the method name.

Standalone test:
    python -m src.training --system duffing --method augmented --epochs 2
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import ExperimentConfig, SystemConfig, load_config
from src.systems import create_system
from src.dataset import Phase1Dataset, generate_phase2_data
from src.models import (
    Normalizer, KKLNetwork, build_kkl_network,
    RecurrentEncoder, InputInjectionNet,
    ResidualHyperNetwork, PerLayerLoRAHyperNetwork,
    apply_weight_modulation, apply_weight_modulation_skip_bias,
    count_parameters, get_layer_sizes, pde_loss,
)
from src.logger import ExperimentLogger, NullLogger


# ---------------------------------------------------------------------------
# Phase 1: Autonomous KKL (encoder T + decoder T*)
# ---------------------------------------------------------------------------

def train_phase1(system, sys_config: SystemConfig, cfg: ExperimentConfig,
                 device: torch.device, logger: ExperimentLogger = None):
    """Train autonomous KKL encoder T and decoder T*.

    Returns: T_net, T_inv_net, loss_history dict
    """
    if logger is None:
        logger = NullLogger()

    p1 = cfg.phase1
    M, K = sys_config.M_np, sys_config.K_np

    print(f"\n{'=' * 60}")
    print(f"Phase 1: Training Autonomous KKL for {sys_config.name.upper()}")
    print(f"{'=' * 60}")

    # --- Encoder T ---
    trainset = Phase1Dataset(system, sys_config, p1.num_ic, seed=cfg.seed, pinn_mode="split_traj")
    norm_T = Normalizer.from_dataset(trainset)
    T_net = build_kkl_network(sys_config, norm_T, role="encoder").to(device)
    T_net.mode = "normal"

    optimizer = torch.optim.Adam(T_net.parameters(), lr=p1.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loader = DataLoader(trainset, batch_size=p1.batch_size, shuffle=True)

    encoder_losses = []
    print(f"\nTraining T (encoder) for {p1.epochs} epochs...")
    for epoch in range(p1.epochs):
        loss_sum = 0
        for idx, (x, z, y, x_ph, y_ph) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)):
            x, z, y = x.to(device), z.to(device), y.to(device)
            x_ph, y_ph = x_ph.to(device), y_ph.to(device)

            z_hat = T_net(x)
            loss_mse = F.mse_loss(z_hat, z)

            if p1.use_pde:
                loss_pde_val = pde_loss(T_net, x_ph, y_ph, T_net(x_ph), system, M, K, device)
                lam = p1.lambda_pde * min(1.0, epoch / 5)
                loss = loss_mse + lam * loss_pde_val
            else:
                loss = loss_mse

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T_net.parameters(), 1.0)
            optimizer.step()

        avg = loss_sum / (idx + 1)
        encoder_losses.append(avg)
        scheduler.step(avg)
        logger.log_scalar("phase1/encoder_loss", avg, epoch)
        print(f"  Epoch {epoch + 1}: Loss = {avg:.6f}")

    # --- Decoder T* ---
    inv_trainset = Phase1Dataset(system, sys_config, p1.num_ic, seed=cfg.seed,
                                 pretrained_T=T_net, pinn_mode="no_physics")
    norm_Tinv = Normalizer.from_dataset(inv_trainset)
    T_inv_net = build_kkl_network(sys_config, norm_Tinv, role="decoder").to(device)
    T_inv_net.mode = "normal"

    optimizer = torch.optim.Adam(T_inv_net.parameters(), lr=p1.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loader = DataLoader(inv_trainset, batch_size=p1.batch_size, shuffle=True)

    decoder_losses = []
    print(f"\nTraining T* (decoder) for {p1.epochs} epochs...")
    for epoch in range(p1.epochs):
        loss_sum = 0
        for idx, (x, z, y, _, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)):
            x, z = x.to(device), z.to(device)
            x_hat = T_inv_net(z)
            loss = F.mse_loss(x_hat, x)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T_inv_net.parameters(), 1.0)
            optimizer.step()

        avg = loss_sum / (idx + 1)
        decoder_losses.append(avg)
        scheduler.step(avg)
        logger.log_scalar("phase1/decoder_loss", avg, epoch)
        print(f"  Epoch {epoch + 1}: Loss = {avg:.6f}")

    return T_net, T_inv_net, {"encoder": encoder_losses, "decoder": decoder_losses}


# ---------------------------------------------------------------------------
# Phase 2: Augmented Observer (frozen T/T*, learn encoder + phi)
# ---------------------------------------------------------------------------

def train_augmented(T_base, T_inv_base, sys_config: SystemConfig, train_data: dict,
                    cfg: ExperimentConfig, device: torch.device,
                    logger: ExperimentLogger = None):
    """Train Augmented Observer: learn input encoder + phi injection into observer ODE."""
    if logger is None:
        logger = NullLogger()
    p2 = cfg.phase2
    encoder_type = p2.encoder_type

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Training Augmented Observer ({encoder_type}) for {sys_config.name.upper()}")
    print(f"{'=' * 60}")

    n_z = sys_config.z_size
    input_encoder = RecurrentEncoder(1, p2.rnn_hidden, p2.latent_dim, cell_type=encoder_type).to(device)

    phi_net = InputInjectionNet(n_z, p2.latent_dim, 1).to(device)

    for p in T_base.parameters():
        p.requires_grad = False
    for p in T_inv_base.parameters():
        p.requires_grad = False
    T_base, T_inv_base = T_base.to(device), T_inv_base.to(device)

    A = torch.tensor(sys_config.M_np, dtype=torch.float32, device=device)
    B = torch.tensor(sys_config.K_np, dtype=torch.float32, device=device)

    params = list(input_encoder.parameters()) + list(phi_net.parameters())
    optimizer = torch.optim.Adam(params, lr=p2.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2.epochs)

    dataset = TensorDataset(train_data["x"], train_data["y"], train_data["u_window"],
                            train_data["u_current"], train_data["dxdt"])
    loader = DataLoader(dataset, batch_size=p2.batch_size, shuffle=True)

    loss_history = []
    for epoch in range(p2.epochs):
        input_encoder.train()
        phi_net.train()
        loss_sum = 0

        for x, y, u_win, u_cur, dxdt in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            x, y, u_win, u_cur, dxdt = [t.to(device) for t in [x, y, u_win, u_cur, dxdt]]

            with torch.no_grad():
                z = T_base(x)

            latent = input_encoder(u_win)
            phi = phi_net(z, latent)

            _, dz_true = jvp(T_base, x, dxdt)

            Az = torch.matmul(z, A.T)
            By = y * B.squeeze(-1).unsqueeze(0)
            dz_observer = Az + By + phi.squeeze(-1)

            loss = F.mse_loss(dz_observer, dz_true)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()
        avg = loss_sum / len(loader)
        loss_history.append(avg)
        tag = "phase2/augmented_loss"
        logger.log_scalar(tag, avg, epoch)
        print(f"  Epoch {epoch + 1}: Loss = {avg:.6f}")

    return input_encoder, phi_net, loss_history


# ---------------------------------------------------------------------------
# Phase 2: Curriculum Learning
# ---------------------------------------------------------------------------

def train_curriculum(T_base, T_inv_base, system, sys_config: SystemConfig,
                     cfg: ExperimentConfig, device: torch.device,
                     logger: ExperimentLogger = None):
    """Curriculum learning: staged data complexity, fine-tunes T and T*.

    Stages expose the autoencoder to increasingly complex input signals:
        1. Zero input (autonomous) — with PDE regularizer
        2. Constant input
        3. Sinusoidal input
        4. All natural inputs mixed

    Loss: ||T*(T(x)) - x||^2 + lambda_pde * ||dT/dx*f - Mz - Ky||^2 (stage 1 only)
    """
    if logger is None:
        logger = NullLogger()
    cc = cfg.curriculum

    print(f"\n{'=' * 60}")
    print(f"Curriculum Learning for {sys_config.name.upper()}")
    print(f"{'=' * 60}")

    lambda_pde = cfg.phase1.lambda_pde

    stages = [
        {"name": "Stage 1: Zero", "inputs": ["zero"], "epochs": cc.stage1_epochs, "use_pde": True},
        {"name": "Stage 2: Constant", "inputs": ["constant"], "epochs": cc.stage2_epochs, "use_pde": False},
        {"name": "Stage 3: Sinusoid", "inputs": ["sinusoid"], "epochs": cc.stage3_epochs, "use_pde": False},
        {"name": "Stage 4: All", "inputs": sys_config.natural_inputs, "epochs": cc.stage4_epochs, "use_pde": False},
    ]

    for p in T_base.parameters():
        p.requires_grad = True
    for p in T_inv_base.parameters():
        p.requires_grad = True
    T_base, T_inv_base = T_base.to(device), T_inv_base.to(device)

    A = torch.tensor(sys_config.M_np, dtype=torch.float32, device=device)
    B = torch.tensor(sys_config.K_np, dtype=torch.float32, device=device)

    params = list(T_base.parameters()) + list(T_inv_base.parameters())
    total_epochs = cc.stage1_epochs + cc.stage2_epochs + cc.stage3_epochs + cc.stage4_epochs
    optimizer = torch.optim.AdamW(params, lr=cc.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    loss_history, stage_boundaries, stage_names = [], [], []
    global_epoch = 0

    for stage_idx, stage in enumerate(stages):
        print(f"\n--- {stage['name']} ({stage['epochs']} epochs, "
              f"pde={'on' if stage['use_pde'] else 'off'}) ---")
        stage_boundaries.append(len(loss_history))
        stage_names.append(stage["name"])

        # Different seed per stage so each stage gets fresh initial conditions
        stage_seed = cfg.seed + stage_idx * 1000
        train_data = generate_phase2_data(
            sys_config, stage["inputs"], cc.n_traj_per_stage,
            cc.window_size, stage_seed)

        dataset = TensorDataset(train_data["x"], train_data["y"], train_data["u_window"],
                                train_data["u_current"], train_data["dxdt"])
        loader = DataLoader(dataset, batch_size=cc.batch_size, shuffle=True)

        for epoch in range(stage["epochs"]):
            T_base.train()
            T_inv_base.train()
            loss_sum = 0

            for x, y, u_win, u_cur, dxdt in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
                x, y, u_win, u_cur, dxdt = [t.to(device) for t in [x, y, u_win, u_cur, dxdt]]

                z = T_base(x)
                x_recon = T_inv_base(z)
                loss_recon = F.mse_loss(x_recon, x)

                loss = loss_recon

                if stage["use_pde"]:
                    _, dz_true = jvp(T_base, x, dxdt)
                    Az = torch.matmul(z, A.T)
                    By = y * B.squeeze(-1).unsqueeze(0)
                    loss_pde = F.mse_loss(Az + By, dz_true)
                    loss = loss + lambda_pde * loss_pde

                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            scheduler.step()
            avg = loss_sum / len(loader)
            loss_history.append(avg)
            logger.log_scalar("phase2/curriculum_loss", avg, global_epoch)
            global_epoch += 1
            print(f"    Epoch {epoch + 1}: Loss = {avg:.6f} (lr={scheduler.get_last_lr()[0]:.2e})")

    return T_base, T_inv_base, {
        "losses": loss_history,
        "stage_boundaries": stage_boundaries,
        "stage_names": stage_names,
    }


# ---------------------------------------------------------------------------
# Phase 2: Dynamic Training (full weight modulation or LoRA)
# ---------------------------------------------------------------------------

def train_dynamic(T_base, T_inv_base, sys_config: SystemConfig, train_data: dict,
                  cfg: ExperimentConfig, device: torch.device,
                  method: str = "full",
                  logger: ExperimentLogger = None):
    """Unified dynamic HyperKKL training.

    Args:
        method: "full" (ResidualHyperNetwork) or "lora" (PerLayerLoRAHyperNetwork)
            Encoder type (lstm/gru) is read from cfg.phase2.encoder_type.
    """
    if logger is None:
        logger = NullLogger()
    p2 = cfg.phase2
    cell_type = p2.encoder_type

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Training {method.upper()} ({cell_type}) for {sys_config.name.upper()}")
    print(f"{'=' * 60}")

    is_lora = method == "lora"
    skip_bias = is_lora
    modulate_fn = apply_weight_modulation_skip_bias if skip_bias else apply_weight_modulation

    # Build hypernetwork
    theta_enc = count_parameters(T_base)
    theta_dec = count_parameters(T_inv_base)

    if is_lora:
        enc_sizes = get_layer_sizes(T_base)
        dec_sizes = get_layer_sizes(T_inv_base)
        hypernet = PerLayerLoRAHyperNetwork(
            n_u=1, lstm_hidden_dim=p2.rnn_hidden,
            enc_layer_sizes=enc_sizes, dec_layer_sizes=dec_sizes,
            rank=p2.lora_rank, mlp_hidden_dim=p2.hypernet_hidden,
            scale_init=0.01, cell_type=cell_type,
        ).to(device)
    else:
        hypernet = ResidualHyperNetwork(
            n_u=1, hidden_dim=p2.rnn_hidden,
            encoder_theta_size=theta_enc, decoder_theta_size=theta_dec,
            mlp_hidden_dim=p2.hypernet_hidden,
            scale_init=0.01, cell_type=cell_type,
        ).to(device)

    print(f"  Hypernet params: {sum(p.numel() for p in hypernet.parameters())}")

    for p in T_base.parameters():
        p.requires_grad = False
    for p in T_inv_base.parameters():
        p.requires_grad = False
    T_base, T_inv_base = T_base.to(device), T_inv_base.to(device)

    A = torch.tensor(sys_config.M_np, dtype=torch.float32, device=device)
    B = torch.tensor(sys_config.K_np, dtype=torch.float32, device=device)
    dt_sim = sys_config.dt

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=p2.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2.epochs)

    dataset = TensorDataset(train_data["x"], train_data["y"], train_data["u_window"],
                            train_data["u_current"], train_data["dxdt"],
                            train_data["u_window_prev"])
    loader = DataLoader(dataset, batch_size=p2.batch_size, shuffle=True)

    loss_history = []
    for epoch in range(p2.epochs):
        hypernet.train()
        loss_sum = 0

        for x, y, u_win, u_cur, dxdt, u_win_prev in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            x, y, u_win, u_cur, dxdt, u_win_prev = [t.to(device) for t in [x, y, u_win, u_cur, dxdt, u_win_prev]]

            delta_enc, delta_dec = hypernet(u_win)

            def T_modulated(inp):
                return torch.func.functional_call(T_base, modulate_fn(T_base, delta_enc), inp)

            def Tstar_modulated(inp):
                return torch.func.functional_call(T_inv_base, modulate_fn(T_inv_base, delta_dec), inp)

            T_x = T_modulated(x)

            # dT/dt via finite difference
            delta_enc_prev, _ = hypernet(u_win_prev)

            def T_modulated_prev(inp):
                return torch.func.functional_call(T_base, modulate_fn(T_base, delta_enc_prev), inp)

            dTdt = (T_x - T_modulated_prev(x)) / dt_sim

            # dT/dx * f(x, u)
            _, dTdx_f = jvp(T_modulated, x, dxdt)

            # PDE residual: dT/dt + dT/dx*f - Mz - Ky
            A_T_x = torch.matmul(T_x, A.T)
            B_y = y * B.squeeze(-1).unsqueeze(0)
            pde_residual = (dTdt + dTdx_f) - A_T_x - B_y

            # PDE loss (LoRA uses normalized residual, others use Huber-like)
            if is_lora:
                dxdt_norm = torch.sqrt(torch.sum(dxdt ** 2, dim=-1, keepdim=True) + 1e-8)
                loss_pde = torch.mean(torch.sum((pde_residual / dxdt_norm) ** 2, dim=-1))
            else:
                loss_pde = torch.mean(torch.sum(torch.sqrt(pde_residual ** 2 + 1e-8), dim=-1))

            # Reconstruction loss
            loss_recon = F.mse_loss(Tstar_modulated(T_x), x)

            loss = loss_pde + loss_recon
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        avg = loss_sum / len(loader)
        loss_history.append(avg)
        logger.log_scalar(f"phase2/{method}_loss", avg, epoch)
        print(f"  Epoch {epoch + 1}: Loss = {avg:.6f}")

    return hypernet, T_base, T_inv_base, loss_history


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training test")
    parser.add_argument("--system", default="duffing")
    parser.add_argument("--method", default="augmented",
                        choices=["autonomous", "augmented", "full", "lora"])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.system, overrides={
        "phase1": {"epochs": args.epochs, "num_ic": 30},
        "phase2": {"epochs": args.epochs, "n_train_traj": 30},
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = create_system(cfg.system)

    print("Training Phase 1...")
    T, T_inv, p1_loss = train_phase1(system, cfg.system, cfg, device)
    print(f"Phase 1 done. Encoder final loss: {p1_loss['encoder'][-1]:.6f}")

    if args.method != "autonomous":
        print(f"\nTraining {args.method}...")
        train_data = generate_phase2_data(
            cfg.system, cfg.system.natural_inputs,
            cfg.phase2.n_train_traj, cfg.phase2.window_size, cfg.seed)

        T_copy = copy.deepcopy(T)
        T_inv_copy = copy.deepcopy(T_inv)

        if args.method == "augmented":
            enc, phi, losses = train_augmented(T_copy, T_inv_copy, cfg.system, train_data, cfg, device)
            print(f"Augmented final loss: {losses[-1]:.6f}")
        elif args.method in ("full", "lora"):
            hypernet, _, _, losses = train_dynamic(T_copy, T_inv_copy, cfg.system, train_data,
                                                   cfg, device, args.method)
            print(f"{args.method} final loss: {losses[-1]:.6f}")
