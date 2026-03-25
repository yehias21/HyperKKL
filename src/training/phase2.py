"""
Phase 2: Non-autonomous HyperKKL training methods.

- train_hyperkkl:          Static configuration (frozen T/T*, learn phi)
- train_curriculum:        Curriculum learning (staged data, same architecture as static)
- train_dynamic_hyperkkl:  Dynamic configuration (hypernetwork modulates T/T* weights)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd.functional import jvp
from tqdm import tqdm

from hyperkkl.src.models.hypernetworks import (
    WindowEncoder, LSTMEncoder, InputInjectionNet,
    DualHyperNetwork, ResidualHyperNetwork,
    apply_weight_modulation, count_parameters,
    PerLayerLoRAHyperNetwork, get_layer_sizes, count_weight_parameters,
    apply_weight_modulation_skip_bias,
)
from hyperkkl.src.data.data_gen import generate_hyperkkl_data_parallel


# ---------------------------------------------------------------------------
# Static HyperKKL
# ---------------------------------------------------------------------------

def train_hyperkkl(T_base, T_inv_base, sys_config: dict, train_data: dict,
                   hyper_cfg: dict, device: torch.device, encoder_type: str = 'window'):
    """Train Static HyperKKL (Phase 2) using JVP."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Training Static HyperKKL ({encoder_type}) for {sys_config['name'].upper()}")
    print(f"{'='*60}")

    n_z = sys_config['z_size']
    latent_dim = hyper_cfg['latent_dim']
    window_size = hyper_cfg['window_size']

    if encoder_type == 'window':
        input_encoder = WindowEncoder(window_size, 1, latent_dim).to(device)
    else:
        input_encoder = LSTMEncoder(1, hyper_cfg.get('lstm_hidden', 64), latent_dim).to(device)

    # Use InputInjectionNet instead of PhiNetwork for consistency with full pipeline
    phi_net = InputInjectionNet(n_z, latent_dim, 1).to(device)

    for p in T_base.parameters():
        p.requires_grad = False
    for p in T_inv_base.parameters():
        p.requires_grad = False
    T_base = T_base.to(device)
    T_inv_base = T_inv_base.to(device)

    A = torch.tensor(sys_config['M'], dtype=torch.float32).to(device)
    B = torch.tensor(sys_config['K'], dtype=torch.float32).to(device)

    params = list(input_encoder.parameters()) + list(phi_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=hyper_cfg['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyper_cfg['epochs'])

    dataset = TensorDataset(train_data['x'], train_data['y'], train_data['u_window'],
                            train_data['u_current'], train_data['dxdt'])
    train_loader = DataLoader(dataset, batch_size=hyper_cfg['batch_size'], shuffle=True)

    loss_history = []
    for epoch in range(hyper_cfg['epochs']):
        input_encoder.train()
        phi_net.train()
        loss_sum = 0

        for x, y, u_window, u_current, dxdt in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            x, y, u_window, u_current, dxdt = [t.to(device) for t in [x, y, u_window, u_current, dxdt]]

            with torch.no_grad():
                z = T_base(x)

            latent = input_encoder(u_window)
            phi = phi_net(z, latent)

            # JVP for efficient dz/dt = dT/dx * dx/dt
            _, dz_true = jvp(T_base, x, dxdt)

            Az = torch.matmul(z, A.T)
            By = y * B.squeeze(-1).unsqueeze(0)
            phi_val = phi.squeeze(-1)  # (batch, z_size) -- not multiplied by u
            dz_observer = Az + By + phi_val

            loss_dynamics = F.mse_loss(dz_observer, dz_true)

            loss = loss_dynamics
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()
        avg_loss = loss_sum / len(train_loader)
        loss_history.append(avg_loss)
        print(f'  Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    return input_encoder, phi_net, loss_history


# ---------------------------------------------------------------------------
# Curriculum Learning
# ---------------------------------------------------------------------------

def train_curriculum(T_base, T_inv_base, system, sys_config: dict,
                     curr_cfg: dict, device: torch.device):
    """Train T and T* with curriculum learning (staged data complexity).

    Extends the Phase-1 autoencoder (T, T*) to non-autonomous regimes by
    gradually exposing the networks to data generated under increasingly
    complex input signals.  No hypernetwork or phi_net is involved.

    Loss:
        Primary — autoencoder reconstruction:  ||T*(T(x)) - x||^2
        The PDE loss (dT/dx*f = Mz + Ky) only holds for autonomous data
        (u=0), so it is applied only in Stage 1 as a warm-up regulariser.
        In later stages (u != 0) the observer PDE is incorrect without
        a phi term, so we train with autoencoder loss only.
    """
    print(f"\n{'='*60}")
    print(f"Curriculum Learning for {sys_config['name'].upper()}")
    print(f"{'='*60}")

    stages = [
        {'name': 'Stage 1: Zero',     'inputs': ['zero'],     'epochs': curr_cfg.get('stage1_epochs', 30), 'use_pde': True},
        {'name': 'Stage 2: Constant', 'inputs': ['constant'], 'epochs': curr_cfg.get('stage2_epochs', 20), 'use_pde': False},
        {'name': 'Stage 3: Sinusoid', 'inputs': ['sinusoid'], 'epochs': curr_cfg.get('stage3_epochs', 20), 'use_pde': False},
        {'name': 'Stage 4: All',      'inputs': sys_config['natural_inputs'], 'epochs': curr_cfg.get('stage4_epochs', 30), 'use_pde': False},
    ]

    window_size = curr_cfg['window_size']

    # Unfreeze T and T* -- curriculum trains the autonomous maps
    for p in T_base.parameters():
        p.requires_grad = True
    for p in T_inv_base.parameters():
        p.requires_grad = True
    T_base = T_base.to(device)
    T_inv_base = T_inv_base.to(device)

    A = torch.tensor(sys_config['M'], dtype=torch.float32).to(device)
    B = torch.tensor(sys_config['K'], dtype=torch.float32).to(device)

    params = list(T_base.parameters()) + list(T_inv_base.parameters())
    optimizer = torch.optim.AdamW(params, lr=curr_cfg['lr'])

    loss_history = []
    stage_boundaries = []
    stage_names = []

    for stage in stages:
        print(f"\n--- {stage['name']} ({stage['epochs']} epochs, "
              f"pde={'on' if stage['use_pde'] else 'off'}) ---")
        stage_boundaries.append(len(loss_history))
        stage_names.append(stage['name'])

        train_data = generate_hyperkkl_data_parallel(
            sys_config['name'], sys_config, stage['inputs'],
            curr_cfg['n_traj_per_stage'], window_size, curr_cfg['seed']
        )

        dataset = TensorDataset(train_data['x'], train_data['y'], train_data['u_window'],
                                train_data['u_current'], train_data['dxdt'])
        train_loader = DataLoader(dataset, batch_size=curr_cfg['batch_size'], shuffle=True)

        use_pde = stage['use_pde']

        for epoch in range(stage['epochs']):
            T_base.train()
            T_inv_base.train()
            loss_sum = 0

            for x, y, u_win, u_cur, dxdt in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
                x, y, u_win, u_cur, dxdt = [t.to(device) for t in [x, y, u_win, u_cur, dxdt]]

                z = T_base(x)

                # Autoencoder reconstruction loss (primary): x ≈ T*(T(x))
                x_recon = T_inv_base(z)
                loss_recon = F.mse_loss(x_recon, x)

                loss = loss_recon

                # PDE regulariser only for autonomous data (Stage 1)
                if use_pde:
                    _, dz_true = jvp(T_base, x, dxdt)
                    Az = torch.matmul(z, A.T)
                    By = y * B.squeeze(-1).unsqueeze(0)
                    dz_observer = Az + By
                    loss_pde = F.mse_loss(dz_observer, dz_true)
                    loss = loss + loss_pde

                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            avg_loss = loss_sum / len(train_loader)
            loss_history.append(avg_loss)
            print(f'    Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    return T_base, T_inv_base, {
        'losses': loss_history,
        'stage_boundaries': stage_boundaries,
        'stage_names': stage_names,
    }


# ---------------------------------------------------------------------------
# Dynamic HyperKKL
# ---------------------------------------------------------------------------

def train_dynamic_hyperkkl(T_base, T_inv_base, sys_config: dict, train_data: dict,
                            hyper_cfg: dict, device: torch.device, encoder_type: str = 'window'):
    """Train Dynamic HyperKKL - modulates both T and T*.

    When encoder_type='window': Uses DualHyperNetwork with low-rank weight modulation.
    When encoder_type='lstm' or 'gru': Uses ResidualHyperNetwork with recurrent
        encoder (LSTM or GRU) + MLP delta generators.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Training Dynamic HyperKKL ({encoder_type}) for {sys_config['name'].upper()}")
    print(f"{'='*60}")

    n_z = sys_config['z_size']
    latent_dim = hyper_cfg['latent_dim']
    window_size = hyper_cfg['window_size']

    theta_enc_size = count_parameters(T_base)
    theta_dec_size = count_parameters(T_inv_base)
    print(f"  Encoder params: {theta_enc_size}, Decoder params: {theta_dec_size}")

    if encoder_type == 'window':
        rank = hyper_cfg.get('rank', 32)
        input_encoder = WindowEncoder(window_size, 1, latent_dim).to(device)
        hypernet = DualHyperNetwork(
            input_encoder=input_encoder, latent_dim=latent_dim,
            encoder_theta_size=theta_enc_size, decoder_theta_size=theta_dec_size,
            rank=rank, shared_hidden_dim=hyper_cfg.get('hypernet_hidden', 128)
        ).to(device)
    else:
        # Hypernetwork-based Residual Weight architecture:
        # Recurrent cell processes input history -> MLP heads generate delta weights
        cell_type = 'gru' if encoder_type == 'gru' else 'lstm'
        rnn_hidden = hyper_cfg.get('lstm_hidden', 64)
        hypernet = ResidualHyperNetwork(
            n_u=1,
            hidden_dim=rnn_hidden,
            encoder_theta_size=theta_enc_size,
            decoder_theta_size=theta_dec_size,
            mlp_hidden_dim=hyper_cfg.get('hypernet_hidden', 128),
            scale_init=0.01,
            cell_type=cell_type,
        ).to(device)

    print(f"  Hypernet params: {sum(p.numel() for p in hypernet.parameters())}")

    for p in T_base.parameters():
        p.requires_grad = False
    for p in T_inv_base.parameters():
        p.requires_grad = False
    T_base = T_base.to(device)
    T_inv_base = T_inv_base.to(device)

    A = torch.tensor(sys_config['M'], dtype=torch.float32).to(device)
    B = torch.tensor(sys_config['K'], dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=hyper_cfg['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyper_cfg['epochs'])

    # Compute dt for the dT/dt finite difference
    dt_sim = (sys_config['b'] - sys_config['a']) / sys_config['N']

    dataset = TensorDataset(train_data['x'], train_data['y'], train_data['u_window'],
                            train_data['u_current'], train_data['dxdt'],
                            train_data['u_window_prev'])
    train_loader = DataLoader(dataset, batch_size=hyper_cfg['batch_size'], shuffle=True)

    loss_history = []
    for epoch in range(hyper_cfg['epochs']):
        hypernet.train()
        loss_sum = 0

        for x, y, u_win, u_cur, dxdt, u_win_prev in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            x, y, u_win, u_cur, dxdt, u_win_prev = [t.to(device) for t in [x, y, u_win, u_cur, dxdt, u_win_prev]]

            # Generate time-varying weight residuals from input window
            delta_enc, delta_dec = hypernet(u_win)

            # Time-Varying Encoder: T_t(x) = T_base(x; theta*_enc + alpha * delta_enc)
            def T_modulated(inp):
                mod_params = apply_weight_modulation(T_base, delta_enc)
                return torch.func.functional_call(T_base, mod_params, inp)

            # Time-Varying Decoder: T*_t(z) = T*_base(z; theta*_dec + alpha * delta_dec)
            def Tstar_modulated(inp):
                mod_params = apply_weight_modulation(T_inv_base, delta_dec)
                return torch.func.functional_call(T_inv_base, mod_params, inp)

            T_x = T_modulated(x)

            # Compute dT/dt via finite difference: (T_t(x) - T_{t-1}(x)) / dt
            # T_{t-1} uses the previous input window to produce different weights
            delta_enc_prev, _ = hypernet(u_win_prev)
            def T_modulated_prev(inp):
                mod_params = apply_weight_modulation(T_base, delta_enc_prev)
                return torch.func.functional_call(T_base, mod_params, inp)
            T_x_prev = T_modulated_prev(x)
            dTdt = (T_x - T_x_prev) / dt_sim

            # Spatial derivative: (dT/dx) * f(x,u)
            _, dTdx_f = jvp(T_modulated, x, dxdt)

            # Full Luenberger PDE: dT/dt + (dT/dx)*f = M*z + K*y
            A_mul_T = torch.matmul(T_x, A.T)
            B_mul_h = y * B.squeeze(-1).unsqueeze(0)

            pde_residual = (dTdt + dTdx_f) - A_mul_T - B_mul_h

            loss_pde = torch.mean(torch.sum(torch.sqrt(pde_residual**2 + 1e-8), dim=-1))

            # Reconstruction Loss: ||T*_t(T_t(x)) - x||^2
            x_reconstructed = Tstar_modulated(T_x)
            loss_recon = F.mse_loss(x_reconstructed, x)

            loss = loss_pde + loss_recon
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        avg_loss = loss_sum / len(train_loader)
        loss_history.append(avg_loss)
        print(f'  Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    return hypernet, T_base, T_inv_base, loss_history


# ---------------------------------------------------------------------------
# Dynamic LoRA
# ---------------------------------------------------------------------------

def train_dynamic_lora(T_base, T_inv_base, sys_config: dict, train_data: dict,
                       hyper_cfg: dict, device: torch.device,
                       cell_type: str = 'lstm'):
    """Train Dynamic LoRA - per-layer low-rank weight modulation for T and T*.

    Uses PerLayerLoRAHyperNetwork with recurrent encoder (LSTM or GRU).
    Modulates only weight parameters (biases are kept fixed from Phase-1).
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Training Dynamic LoRA ({cell_type}) for {sys_config['name'].upper()}")
    print(f"{'='*60}")

    enc_sizes = get_layer_sizes(T_base)
    dec_sizes = get_layer_sizes(T_inv_base)
    print(f"  Encoder layers: {enc_sizes}, Decoder layers: {dec_sizes}")

    hypernet = PerLayerLoRAHyperNetwork(
        n_u=1,
        lstm_hidden_dim=hyper_cfg.get('lstm_hidden', 64),
        enc_layer_sizes=enc_sizes,
        dec_layer_sizes=dec_sizes,
        rank=hyper_cfg.get('lora_rank', 4),
        mlp_hidden_dim=hyper_cfg.get('hypernet_hidden', 128),
        scale_init=0.01,
        cell_type=cell_type,
        use_input_gate=hyper_cfg.get('lora_input_gate', True),
    ).to(device)

    print(f"  LoRA Hypernet params: {sum(p.numel() for p in hypernet.parameters())}")

    for p in T_base.parameters():
        p.requires_grad = False
    for p in T_inv_base.parameters():
        p.requires_grad = False
    T_base = T_base.to(device)
    T_inv_base = T_inv_base.to(device)

    A = torch.tensor(sys_config['M'], dtype=torch.float32).to(device)
    B = torch.tensor(sys_config['K'], dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=hyper_cfg['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyper_cfg['epochs'])

    dt_sim = (sys_config['b'] - sys_config['a']) / sys_config['N']

    dataset = TensorDataset(train_data['x'], train_data['y'], train_data['u_window'],
                            train_data['u_current'], train_data['dxdt'],
                            train_data['u_window_prev'])
    train_loader = DataLoader(dataset, batch_size=hyper_cfg['batch_size'], shuffle=True)

    loss_history = []
    for epoch in range(hyper_cfg['epochs']):
        hypernet.train()
        loss_sum = 0

        for x, y, u_win, u_cur, dxdt, u_win_prev in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            x, y, u_win, u_cur, dxdt, u_win_prev = [t.to(device) for t in [x, y, u_win, u_cur, dxdt, u_win_prev]]

            delta_enc, delta_dec = hypernet(u_win)

            def T_modulated(inp):
                mod_params = apply_weight_modulation_skip_bias(T_base, delta_enc)
                return torch.func.functional_call(T_base, mod_params, inp)

            def Tstar_modulated(inp):
                mod_params = apply_weight_modulation_skip_bias(T_inv_base, delta_dec)
                return torch.func.functional_call(T_inv_base, mod_params, inp)

            T_x = T_modulated(x)

            delta_enc_prev, _ = hypernet(u_win_prev)
            def T_modulated_prev(inp):
                mod_params = apply_weight_modulation_skip_bias(T_base, delta_enc_prev)
                return torch.func.functional_call(T_base, mod_params, inp)
            T_x_prev = T_modulated_prev(x)
            dTdt = (T_x - T_x_prev) / dt_sim

            _, dTdx_f = jvp(T_modulated, x, dxdt)

            A_mul_T = torch.matmul(T_x, A.T)
            B_mul_h = y * B.squeeze(-1).unsqueeze(0)

            pde_residual = (dTdt + dTdx_f) - A_mul_T - B_mul_h

            if hyper_cfg.get('physics_norm', True):
                dxdt_norm_sq = torch.sum(dxdt**2, dim=-1, keepdim=True) + 1e-8
                pde_residual_normalized = pde_residual / dxdt_norm_sq.sqrt()
                loss_pde = torch.mean(torch.sum(pde_residual_normalized**2, dim=-1))
            else:
                loss_pde = torch.mean(torch.sum(pde_residual**2, dim=-1))

            x_reconstructed = Tstar_modulated(T_x)
            loss_recon = F.mse_loss(x_reconstructed, x)

            loss = loss_pde + loss_recon
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        avg_loss = loss_sum / len(train_loader)
        loss_history.append(avg_loss)
        print(f'  Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    return hypernet, T_base, T_inv_base, loss_history
