"""
Neural network models for HyperKKL.

Contains:
    - Normalizer: input/output normalization
    - KKLNetwork: feedforward network with normalizer (encoder T / decoder T*)
    - RecurrentEncoder: LSTM/GRU-based input sequence encoder
    - InputInjectionNet: phi network for augmented observer
    - ResidualHyperNetwork: recurrent full weight modulation (dynamic full)
    - PerLayerLoRAHyperNetwork: per-layer LoRA weight modulation (dynamic LoRA)
    - Weight modulation utilities

Standalone test:
    python -m src.models
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Dataset statistics normalizer. Registers buffers for GPU transfer."""

    def __init__(self, x_size: int, z_size: int,
                 mean_x=None, std_x=None, mean_z=None, std_z=None,
                 mean_x_ph=None, std_x_ph=None, mean_z_ph=None, std_z_ph=None):
        super().__init__()
        self.x_size = x_size
        self.z_size = z_size

        self.register_buffer("mean_x", mean_x if mean_x is not None else torch.zeros(x_size))
        self.register_buffer("std_x", std_x if std_x is not None else torch.ones(x_size))
        self.register_buffer("mean_z", mean_z if mean_z is not None else torch.zeros(z_size))
        self.register_buffer("std_z", std_z if std_z is not None else torch.ones(z_size))
        self.register_buffer("mean_x_ph", mean_x_ph if mean_x_ph is not None else torch.zeros(x_size))
        self.register_buffer("std_x_ph", std_x_ph if std_x_ph is not None else torch.ones(x_size))
        self.register_buffer("mean_z_ph", mean_z_ph if mean_z_ph is not None else torch.zeros(z_size))
        self.register_buffer("std_z_ph", std_z_ph if std_z_ph is not None else torch.ones(z_size))

    @classmethod
    def from_dataset(cls, dataset):
        """Create normalizer from Phase1Dataset statistics."""
        return cls(
            x_size=dataset.x_data.shape[1],
            z_size=dataset.z_data.shape[1],
            mean_x=dataset.mean_x, std_x=dataset.std_x,
            mean_z=dataset.mean_z, std_z=dataset.std_z,
            mean_x_ph=dataset.mean_x_ph, std_x_ph=dataset.std_x_ph,
            mean_z_ph=dataset.mean_z_ph, std_z_ph=dataset.std_z_ph,
        )

    @classmethod
    def dummy(cls, x_size: int, z_size: int):
        """Create a dummy normalizer (buffers overwritten on checkpoint load)."""
        return cls(x_size, z_size)

    def _select_stats(self, tensor, mode):
        dim = tensor.shape[1]
        if dim == self.x_size:
            if mode == "physics":
                return self.mean_x_ph, self.std_x
            return self.mean_x, self.std_x
        elif dim == self.z_size:
            if mode == "physics":
                return self.mean_z_ph, self.std_z_ph
            return self.mean_z, self.std_z
        raise RuntimeError(f"Tensor dim {dim} doesn't match x_size={self.x_size} or z_size={self.z_size}")

    def normalize(self, tensor, mode="normal"):
        mean, std = self._select_stats(tensor, mode)
        return (tensor - mean) / std

    def denormalize(self, tensor, mode="normal"):
        mean, std = self._select_stats(tensor, mode)
        return tensor * std + mean

    # Legacy API compatibility
    def Normalize(self, tensor, mode="normal"):
        return self.normalize(tensor, mode)

    def Denormalize(self, tensor, mode="normal"):
        return self.denormalize(tensor, mode)

    def check_sys(self, tensor, mode):
        return self._select_stats(tensor, mode)


# ---------------------------------------------------------------------------
# Base KKL Network (encoder T / decoder T*)
# ---------------------------------------------------------------------------

class KKLNetwork(nn.Module):
    """Feedforward network with input normalization and output denormalization.

    Used as both encoder T: x -> z and decoder T*: z -> x.
    """

    def __init__(self, num_hidden: int, hidden_size: int,
                 in_size: int, out_size: int,
                 activation=F.relu, normalizer: Normalizer = None):
        super().__init__()
        self.normalizer = normalizer
        self.mode = "normal"

        layers = []
        current_dim = in_size
        for _ in range(num_hidden):
            layers.append(nn.Linear(current_dim, hidden_size))
            current_dim = hidden_size
        layers.append(nn.Linear(current_dim, out_size))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer.normalize(x, self.mode).float()

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        if self.normalizer is not None:
            x = self.normalizer.denormalize(x, self.mode).float()
        return x


def build_kkl_network(sys_config, normalizer, role="encoder"):
    """Build a KKL network for encoder or decoder role."""
    if role == "encoder":
        return KKLNetwork(
            sys_config.num_hidden, sys_config.hidden_size,
            sys_config.x_size, sys_config.z_size,
            normalizer=normalizer)
    else:
        return KKLNetwork(
            sys_config.num_hidden, sys_config.hidden_size,
            sys_config.z_size, sys_config.x_size,
            normalizer=normalizer)


# ---------------------------------------------------------------------------
# Input Encoder (LSTM or GRU — selected via config)
# ---------------------------------------------------------------------------

class RecurrentEncoder(nn.Module):
    """Recurrent input sequence encoder. Supports LSTM and GRU."""

    def __init__(self, n_u: int, hidden_dim: int, latent_dim: int,
                 cell_type: str = "lstm"):
        super().__init__()
        self.cell_type = cell_type
        if cell_type == "gru":
            self.rnn = nn.GRU(n_u, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_u, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim, bias=False)

    def forward(self, u_sequence):
        if self.cell_type == "gru":
            _, h_n = self.rnn(u_sequence)
        else:
            _, (h_n, _) = self.rnn(u_sequence)
        return self.fc(h_n.squeeze(0))


# ---------------------------------------------------------------------------
# Augmented Observer: Input Injection
# ---------------------------------------------------------------------------

class InputInjectionNet(nn.Module):
    """Phi network for augmented observer: maps (z, latent) -> phi correction."""

    def __init__(self, n_z: int, latent_dim: int, n_u: int = 1,
                 hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.n_z = n_z
        self.n_u = n_u

        layers = []
        in_dim = n_z + latent_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(in_dim, hd), nn.Tanh()])
            in_dim = hd
        layers.append(nn.Linear(in_dim, n_z * n_u, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, z, latent):
        x = torch.cat([z, latent], dim=-1)
        return self.net(x).view(z.shape[0], self.n_z, self.n_u)


# ---------------------------------------------------------------------------
# Dynamic Full: ResidualHyperNetwork (recurrent, full weight deltas)
# ---------------------------------------------------------------------------

class ResidualHyperNetwork(nn.Module):
    """Recurrent hypernetwork generating full weight residuals for T and T*.

    Architecture: RNN(u_history) -> h_t -> MLP_enc(h_t) -> delta_enc
                                        -> MLP_dec(h_t) -> delta_dec
    """

    def __init__(self, n_u: int, hidden_dim: int,
                 encoder_theta_size: int, decoder_theta_size: int,
                 mlp_hidden_dim: int = 128, scale_init: float = 0.01,
                 cell_type: str = "lstm"):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim

        if cell_type == "gru":
            self.rnn = nn.GRU(n_u, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_u, hidden_dim, batch_first=True)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim), nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.Tanh(),
            nn.Linear(mlp_hidden_dim, encoder_theta_size, bias=False),
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim), nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.Tanh(),
            nn.Linear(mlp_hidden_dim, decoder_theta_size, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def _extract_hidden(self, rnn_out):
        if self.cell_type == "gru":
            _, h_n = rnn_out
        else:
            _, (h_n, _) = rnn_out
        return h_n

    def _get_state(self, rnn_out):
        _, state = rnn_out
        return state

    def forward(self, u_window, state=None):
        rnn_out = self.rnn(u_window, state)
        h = self._extract_hidden(rnn_out).squeeze(0)
        return self.scale * self.encoder_mlp(h), self.scale * self.decoder_mlp(h)

    def step(self, u_t, state=None):
        """Single-step for sequential evaluation."""
        rnn_out = self.rnn(u_t, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        return self.scale * self.encoder_mlp(h), self.scale * self.decoder_mlp(h), new_state

    def forward_with_state(self, u_window, state=None):
        rnn_out = self.rnn(u_window, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        return self.scale * self.encoder_mlp(h), self.scale * self.decoder_mlp(h), new_state


# ---------------------------------------------------------------------------
# Dynamic LoRA: PerLayerLoRAHyperNetwork
# ---------------------------------------------------------------------------

class PerLayerLoRAHyperNetwork(nn.Module):
    """Per-layer LoRA hypernetwork with RNN encoder and input energy gating.

    Generates low-rank A,B factors per layer -> delta_W = scale * A @ B.
    Energy gate ensures delta_W -> 0 when u -> 0.
    """

    def __init__(self, n_u: int, lstm_hidden_dim: int,
                 enc_layer_sizes: list, dec_layer_sizes: list,
                 rank: int = 4, mlp_hidden_dim: int = 128,
                 layer_emb_dim: int = 16, scale_init: float = 0.01,
                 cell_type: str = "lstm"):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = lstm_hidden_dim

        if cell_type == "gru":
            self.rnn = nn.GRU(n_u, lstm_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_u, lstm_hidden_dim, batch_first=True)

        self.n_enc_layers = len(enc_layer_sizes)
        self.n_dec_layers = len(dec_layer_sizes)
        self.layer_sizes = enc_layer_sizes + dec_layer_sizes
        self.rank = rank

        n_total = self.n_enc_layers + self.n_dec_layers
        self.layer_embs = nn.Embedding(n_total, layer_emb_dim)

        self.backbone = nn.Sequential(
            nn.Linear(lstm_hidden_dim + layer_emb_dim, mlp_hidden_dim), nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.Tanh(),
        )

        self.heads = nn.ModuleList()
        for d_in, d_out in self.layer_sizes:
            self.heads.append(nn.Linear(mlp_hidden_dim, rank * (d_in + d_out), bias=False))

        self.scale = nn.Parameter(torch.tensor(scale_init))

    def _extract_hidden(self, rnn_out):
        if self.cell_type == "gru":
            _, h_n = rnn_out
        else:
            _, (h_n, _) = rnn_out
        return h_n

    def _get_state(self, rnn_out):
        _, state = rnn_out
        return state

    def _generate_deltas(self, h):
        batch = h.shape[0]
        device = h.device
        all_deltas = []

        for l_idx in range(len(self.heads)):
            e_l = self.layer_embs(torch.tensor(l_idx, device=device))
            inp = torch.cat([h, e_l.unsqueeze(0).expand(batch, -1)], dim=-1)
            feat = self.backbone(inp)
            raw = self.heads[l_idx](feat)

            d_in, d_out = self.layer_sizes[l_idx]
            A = raw[:, : self.rank * d_in].view(batch, d_in, self.rank)
            B = raw[:, self.rank * d_in :].view(batch, self.rank, d_out)
            delta_W = self.scale * torch.bmm(A, B)
            all_deltas.append(delta_W.view(batch, -1))

        delta_enc = torch.cat(all_deltas[: self.n_enc_layers], dim=-1)
        delta_dec = torch.cat(all_deltas[self.n_enc_layers :], dim=-1)
        return delta_enc, delta_dec

    def _energy_gate(self, u, delta_enc, delta_dec):
        u_rms = torch.sqrt(torch.mean(u ** 2, dim=(1, 2), keepdim=True)).squeeze(-1)
        return delta_enc * u_rms, delta_dec * u_rms

    def forward(self, u_window, state=None):
        rnn_out = self.rnn(u_window, state)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        return self._energy_gate(u_window, delta_enc, delta_dec)

    def step(self, u_t, state=None):
        rnn_out = self.rnn(u_t, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        u_rms = torch.sqrt(torch.mean(u_t ** 2) + 1e-12)
        return delta_enc * u_rms, delta_dec * u_rms, new_state

    def forward_with_state(self, u_window, state=None):
        rnn_out = self.rnn(u_window, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        delta_enc, delta_dec = self._energy_gate(u_window, delta_enc, delta_dec)
        return delta_enc, delta_dec, new_state


# ---------------------------------------------------------------------------
# Weight modulation utilities
# ---------------------------------------------------------------------------

def apply_weight_modulation(base_model: nn.Module, delta_theta: torch.Tensor):
    """Apply delta_theta to all parameters of base_model."""
    new_params = {}
    offset = 0
    for name, param in base_model.named_parameters():
        size = param.numel()
        delta = delta_theta[0, offset : offset + size].view_as(param)
        new_params[name] = param + delta
        offset += size
    return new_params


def apply_weight_modulation_skip_bias(base_model: nn.Module, delta_theta: torch.Tensor):
    """Apply delta_theta only to weight parameters (skip biases)."""
    new_params = {}
    offset = 0
    for name, param in base_model.named_parameters():
        if "bias" in name:
            new_params[name] = param
            continue
        size = param.numel()
        delta = delta_theta[0, offset : offset + size].view_as(param)
        new_params[name] = param + delta
        offset += size
    return new_params


def count_parameters(model: nn.Module) -> int:
    """Count all parameters (including frozen ones)."""
    return sum(p.numel() for p in model.parameters())


def count_weight_parameters(model: nn.Module) -> int:
    """Count weight parameters only (excludes biases)."""
    return sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)


def get_layer_sizes(model: nn.Module) -> list:
    """Get (in_features, out_features) for each Linear layer."""
    return [(m.in_features, m.out_features)
            for m in model.modules() if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------------------
# PDE loss
# ---------------------------------------------------------------------------

def pde_loss(T_net, x, y, z_hat, system, M, K, device, reduction="mean"):
    """Physics-informed PDE loss: ||dT/dx * f(x) - Mz - Ky||^2 / ||f||^2."""
    M_t = torch.from_numpy(M).to(device).float()
    K_t = torch.from_numpy(K).to(device).float()
    x.requires_grad_()

    # Jacobian dT/dx via per-output-dim backward passes
    z = T_net(x)
    n_z = z.shape[1]
    jac_rows = []
    for i in range(n_z):
        grad_out = torch.zeros_like(z)
        grad_out[:, i] = 1.0
        grads = torch.autograd.grad(z, x, grad_out, create_graph=True, retain_graph=True)[0]
        jac_rows.append(grads.unsqueeze(1))
    dTdx = torch.cat(jac_rows, dim=1)  # (batch, n_z, n_x)

    # f(x) for autonomous case (u=0)
    f_val = system.function(0, 0.0, x).to(device).float().unsqueeze(2)

    # Lie derivative: dT/dx * f(x)
    dTdx_f = torch.bmm(dTdx, f_val)  # (batch, n_z, 1)

    # Observer dynamics: Mz + Ky
    z_hat_3d = z_hat.unsqueeze(2)
    Mz = torch.matmul(M_t, z_hat_3d)

    y_t = y.float()
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(1)
    if y_t.ndim == 2 and y_t.shape[1] == 1:
        y_t = y_t.unsqueeze(2)
    Ky = torch.matmul(K_t, y_t)

    pde = dTdx_f - Mz - Ky
    pde_norm = torch.linalg.norm(pde.squeeze(2), dim=1) ** 2

    # Normalize by ||f||^2 for scale invariance
    f_norm_sq = torch.sum(f_val.squeeze(2) ** 2, dim=1) + 1e-8
    loss_batch = pde_norm / f_norm_sq

    if reduction == "mean":
        return torch.mean(loss_batch)
    return loss_batch


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    norm = Normalizer.dummy(2, 5)
    T = KKLNetwork(3, 150, 2, 5, normalizer=norm)
    T_inv = KKLNetwork(3, 150, 5, 2, normalizer=norm)
    x = torch.randn(16, 2)
    z = T(x)
    x_hat = T_inv(z)
    print(f"T: {x.shape} -> {z.shape}, T*: {z.shape} -> {x_hat.shape}")
    print(f"T params: {count_parameters(T)}")

    u = torch.randn(8, 100, 1)
    for cell in ["lstm", "gru"]:
        enc = RecurrentEncoder(1, 64, 32, cell_type=cell)
        print(f"RecurrentEncoder({cell}): {u.shape} -> {enc(u).shape}")

    theta_enc = count_parameters(T)
    theta_dec = count_parameters(T_inv)

    rhn = ResidualHyperNetwork(1, 64, theta_enc, theta_dec)
    d_enc, d_dec = rhn(u)
    print(f"ResidualHyperNetwork (full): d_enc={d_enc.shape}, d_dec={d_dec.shape}")

    enc_sizes = get_layer_sizes(T)
    dec_sizes = get_layer_sizes(T_inv)
    lora = PerLayerLoRAHyperNetwork(1, 64, enc_sizes, dec_sizes, rank=4)
    d_enc, d_dec = lora(u)
    print(f"PerLayerLoRA: d_enc={d_enc.shape}, d_dec={d_dec.shape}")
    print("All model tests passed.")
