"""Neural network modules for HyperKKL: encoders, injection net, hypernetworks."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class WindowEncoder(nn.Module):
    """CNN-based input window encoder.

    Uses 1-D convolutions followed by adaptive average pooling and a bias-free
    linear projection to produce a latent representation of an input window.
    """

    def __init__(
        self,
        window_size: int,
        n_u: int,
        latent_dim: int,
        hidden_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.window_size = window_size
        layers: List[nn.Module] = []
        in_channels = n_u
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.Tanh(),
            ])
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels[-1], latent_dim, bias=False)  # NO BIAS for zero-input guarantee

    def forward(self, u_window: torch.Tensor) -> torch.Tensor:
        """Encode an input window.

        Args:
            u_window: Input tensor of shape ``(batch, window_size, n_u)``.

        Returns:
            Latent representation of shape ``(batch, latent_dim)``.
        """
        # u_window: (batch, window_size, n_u) -> (batch, n_u, window_size)
        x = u_window.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LSTMEncoder(nn.Module):
    """LSTM-based input sequence encoder.

    Processes a variable-length input sequence with an LSTM and projects the
    final hidden state to a latent space via a bias-free linear layer.
    """

    def __init__(self, n_u: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_u, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim, bias=False)  # NO BIAS

    def forward(self, u_sequence: torch.Tensor) -> torch.Tensor:
        """Encode an input sequence.

        Args:
            u_sequence: Input tensor of shape ``(batch, seq_len, n_u)``.

        Returns:
            Latent representation of shape ``(batch, latent_dim)``.
        """
        _, (h_n, _) = self.lstm(u_sequence)
        return self.fc(h_n.squeeze(0))


class InputInjectionNet(nn.Module):
    """Input injection network for static configuration.

    Concatenates the observer state *z* with a latent code and maps the result
    through an MLP to produce a state-dependent injection matrix.
    """

    def __init__(
        self,
        n_z: int,
        latent_dim: int,
        n_u: int = 1,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.n_z = n_z
        self.n_u = n_u

        layers: List[nn.Module] = []
        in_dim = n_z + latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, n_z * n_u, bias=False))  # NO BIAS
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Compute the injection matrix for a given state and latent code.

        Args:
            z: Observer state of shape ``(batch, n_z)``.
            latent: Latent code of shape ``(batch, latent_dim)``.

        Returns:
            Injection matrix of shape ``(batch, n_z, n_u)``.
        """
        x = torch.cat([z, latent], dim=-1)
        phi_flat = self.net(x)
        batch_size = z.shape[0]
        return phi_flat.view(batch_size, self.n_z, self.n_u)


class DualHyperNetwork(nn.Module):
    """Dual hypernetwork for dynamic configuration -- modulates both T and T*.

    Uses a shared trunk to produce low-rank weight residuals for both the
    encoder and decoder networks.
    """

    def __init__(
        self,
        input_encoder: nn.Module,
        latent_dim: int,
        encoder_theta_size: int,
        decoder_theta_size: int,
        rank: int = 32,
        shared_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = input_encoder

        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim, shared_hidden_dim), nn.Tanh(),
            nn.Linear(shared_hidden_dim, shared_hidden_dim), nn.Tanh()
        )

        self.encoder_U = nn.Parameter(torch.zeros(encoder_theta_size, rank))
        self.encoder_v = nn.Linear(shared_hidden_dim, rank, bias=False)
        self.decoder_U = nn.Parameter(torch.zeros(decoder_theta_size, rank))
        self.decoder_v = nn.Linear(shared_hidden_dim, rank, bias=False)

        self.encoder_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.decoder_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(
        self, u_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce encoder and decoder weight residuals.

        Args:
            u_window: Input window tensor.

        Returns:
            Tuple of ``(delta_enc, delta_dec)`` weight residuals.
        """
        latent = self.encoder(u_window)
        shared = self.shared_net(latent)

        v_enc = self.encoder_v(shared)
        delta_enc = self.encoder_scale * torch.einsum('br,pr->bp', v_enc, self.encoder_U)

        v_dec = self.decoder_v(shared)
        delta_dec = self.decoder_scale * torch.einsum('br,pr->bp', v_dec, self.decoder_U)

        return delta_enc, delta_dec


class ResidualHyperNetwork(nn.Module):
    """Hypernetwork-based Residual Weight architecture for non-autonomous KKL.

    Extends a pre-trained static KKL (trained on autonomous dynamics) to a
    non-autonomous setting by making the transformation parameters
    time-varying and input-dependent.

    Architecture:
        A. Recurrent encoder (LSTM or GRU) processes input signal history -> hidden state h_t
        B. Two MLP heads project h_t to parameter residuals:
           - delta_enc = alpha * MLP_enc(h_t)  (residuals for encoder T)
           - delta_dec = alpha * MLP_dec(h_t)  (residuals for decoder T*)
        C. Effective time-varying weights:
           - theta_enc(t) = theta*_enc + alpha * delta_enc(h_t)
           - theta_dec(t) = theta*_dec + alpha * delta_dec(h_t)

    The input energy gate (u_rms scaling) ensures delta weights vanish when
    u=0, recovering the pre-trained static solution architecturally.
    """

    def __init__(
        self,
        n_u: int,
        hidden_dim: int,
        encoder_theta_size: int,
        decoder_theta_size: int,
        mlp_hidden_dim: int = 128,
        scale_init: float = 0.01,
        cell_type: str = 'lstm',
    ) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim

        if cell_type == 'gru':
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.GRU(n_u, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_u, hidden_dim, batch_first=True)

        # MLP head for encoder delta weights: h_t -> delta_theta_enc
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, encoder_theta_size, bias=False),
        )

        # MLP head for decoder delta weights: h_t -> delta_theta_dec
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, decoder_theta_size, bias=False),
        )

        # Scaling factor alpha, initialized small to start near stable static solution
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def _extract_hidden(self, rnn_out: tuple) -> torch.Tensor:
        """Extract h_n from RNN output (handles both LSTM and GRU)."""
        if self.cell_type == 'gru':
            _, h_n = rnn_out
            return h_n  # state is just h_n for GRU
        else:
            _, (h_n, c_n) = rnn_out
            return h_n  # return h_n, state is (h_n, c_n)

    def _get_state(self, rnn_out: tuple) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract full state from RNN output for passing to next call."""
        if self.cell_type == 'gru':
            _, state = rnn_out
            return state  # h_n only
        else:
            _, state = rnn_out
            return state  # (h_n, c_n) tuple

    def forward(
        self,
        u_window: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input window, return delta weights for encoder and decoder.

        Args:
            u_window: Input tensor of shape ``(batch, window_size, n_u)``.
            state: Recurrent state (LSTM: ``(h, c)`` tuple; GRU: ``h`` tensor).

        Returns:
            Tuple of ``(delta_enc, delta_dec)`` weight residuals.
        """
        rnn_out = self.rnn(u_window, state)
        h = self._extract_hidden(rnn_out).squeeze(0)  # (batch, hidden)
        delta_enc = self.scale * self.encoder_mlp(h)
        delta_dec = self.scale * self.decoder_mlp(h)
        return delta_enc, delta_dec

    def forward_with_hidden(
        self,
        u_window: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Process input window, return delta weights and recurrent state.

        Returns:
            Tuple of ``(delta_enc, delta_dec, state)``.
        """
        rnn_out = self.rnn(u_window, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc = self.scale * self.encoder_mlp(h)
        delta_dec = self.scale * self.decoder_mlp(h)
        return delta_enc, delta_dec, new_state

    def step(
        self,
        u_t: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Process single time step, return delta weights and new recurrent state.

        Used during evaluation for sequential input processing.

        Args:
            u_t: Single input step of shape ``(batch, 1, n_u)``.
            state: Recurrent state (LSTM: ``(h, c)`` tuple; GRU: ``h`` tensor).

        Returns:
            Tuple of ``(delta_enc, delta_dec, state)``.
        """
        rnn_out = self.rnn(u_t, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc = self.scale * self.encoder_mlp(h)
        delta_dec = self.scale * self.decoder_mlp(h)
        return delta_enc, delta_dec, new_state


def apply_weight_modulation(
    base_model: nn.Module, delta_theta: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Apply weight modulation to base model parameters.

    Args:
        base_model: The base network whose parameters are modulated.
        delta_theta: Flat tensor of shape ``(1, total_params)`` containing
            additive weight residuals for every parameter.

    Returns:
        Dictionary mapping parameter names to modulated parameter tensors.
    """
    new_params: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in base_model.named_parameters():
        size = param.numel()
        delta = delta_theta[0, offset:offset + size].view_as(param)
        new_params[name] = param + delta
        offset += size
    return new_params


def apply_weight_modulation_skip_bias(
    base_model: nn.Module, delta_theta: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Apply weight modulation to base model parameters, skipping biases.

    Args:
        base_model: The base network whose parameters are modulated.
        delta_theta: Flat tensor of shape ``(1, total_weight_params)`` containing
            additive weight residuals for non-bias parameters only.

    Returns:
        Dictionary mapping parameter names to modulated parameter tensors.
    """
    new_params: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in base_model.named_parameters():
        if 'bias' in name:
            new_params[name] = param
            continue
        size = param.numel()
        delta = delta_theta[0, offset:offset + size].view_as(param)
        new_params[name] = param + delta
        offset += size
    return new_params


def count_parameters(model: nn.Module) -> int:
    """Count ALL parameters in the model, including frozen ones.

    This counts every parameter regardless of ``requires_grad`` to prevent
    size-zero results when sequential methods freeze the model.

    Args:
        model: The network to inspect.

    Returns:
        Total number of scalar parameters.
    """
    return sum(p.numel() for p in model.parameters())


def get_layer_sizes(model: nn.Module) -> List[Tuple[int, int]]:
    """Collect ``(in_features, out_features)`` for every ``nn.Linear`` in the model.

    Args:
        model: The network to inspect.

    Returns:
        List of ``(in_features, out_features)`` tuples.
    """
    sizes: List[Tuple[int, int]] = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            sizes.append((module.in_features, module.out_features))
    return sizes


def count_weight_parameters(model: nn.Module) -> int:
    """Count parameters excluding biases.

    Args:
        model: The network to inspect.

    Returns:
        Total number of non-bias scalar parameters.
    """
    return sum(p.numel() for n, p in model.named_parameters() if 'bias' not in n)


class PerLayerLoRAHyperNetwork(nn.Module):
    """Per-layer LoRA hypernetwork for dynamic KKL observer.

    Generates low-rank weight deltas for each linear layer of the encoder T
    and decoder T* using an RNN (LSTM or GRU) to process input history and
    per-layer embeddings to specialize the output for each layer.
    """

    def __init__(
        self,
        n_u: int,
        lstm_hidden_dim: int,
        enc_layer_sizes: List[Tuple[int, int]],
        dec_layer_sizes: List[Tuple[int, int]],
        rank: int = 4,
        mlp_hidden_dim: int = 128,
        layer_emb_dim: int = 16,
        scale_init: float = 0.01,
        cell_type: str = 'lstm',
        use_input_gate: bool = True,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = lstm_hidden_dim
        self.use_input_gate = use_input_gate

        if cell_type == 'gru':
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.GRU(n_u, lstm_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(n_u, lstm_hidden_dim, batch_first=True)

        self.n_enc_layers = len(enc_layer_sizes)
        self.n_dec_layers = len(dec_layer_sizes)
        self.layer_sizes: List[Tuple[int, int]] = enc_layer_sizes + dec_layer_sizes
        self.rank = rank
        self.skip_bias = True

        n_total_layers = self.n_enc_layers + self.n_dec_layers
        self.layer_embs = nn.Embedding(n_total_layers, layer_emb_dim)

        self.backbone = nn.Sequential(
            nn.Linear(lstm_hidden_dim + layer_emb_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Tanh(),
        )

        self.heads = nn.ModuleList()
        for (d_in, d_out) in self.layer_sizes:
            self.heads.append(
                nn.Linear(mlp_hidden_dim, rank * (d_in + d_out), bias=False)
            )

        self.scale = nn.Parameter(torch.tensor(scale_init))

    def _extract_hidden(self, rnn_out: tuple) -> torch.Tensor:
        """Extract h_n from RNN output (handles both LSTM and GRU)."""
        if self.cell_type == 'gru':
            _, h_n = rnn_out
            return h_n
        else:
            _, (h_n, c_n) = rnn_out
            return h_n

    def _get_state(self, rnn_out: tuple) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract full state from RNN output for passing to next call."""
        if self.cell_type == 'gru':
            _, state = rnn_out
            return state
        else:
            _, state = rnn_out
            return state

    def _generate_deltas(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate per-layer LoRA deltas from hidden state.

        Args:
            h: Hidden state of shape ``(batch, hidden_dim)``.

        Returns:
            Tuple of ``(delta_enc, delta_dec)`` flattened weight residuals.
        """
        batch = h.shape[0]
        device = h.device
        all_deltas: List[torch.Tensor] = []

        for l in range(len(self.heads)):
            e_l = self.layer_embs(torch.tensor(l, device=device))
            e_l_batch = e_l.unsqueeze(0).expand(batch, -1)
            inp = torch.cat([h, e_l_batch], dim=-1)
            feat = self.backbone(inp)
            raw = self.heads[l](feat)

            d_in, d_out = self.layer_sizes[l]
            A_flat = raw[:, :self.rank * d_in]
            B_flat = raw[:, self.rank * d_in:]
            A = A_flat.view(batch, d_in, self.rank)
            B = B_flat.view(batch, self.rank, d_out)
            delta_W = self.scale * torch.bmm(A, B)
            all_deltas.append(delta_W.view(batch, -1))

        enc_deltas = all_deltas[:self.n_enc_layers]
        dec_deltas = all_deltas[self.n_enc_layers:]

        delta_enc = torch.cat(enc_deltas, dim=-1)
        delta_dec = torch.cat(dec_deltas, dim=-1)
        return delta_enc, delta_dec

    def forward(
        self,
        u_window: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input window, return delta weights for encoder and decoder.

        Args:
            u_window: Input tensor of shape ``(batch, window_size, n_u)``.
            state: RNN state (LSTM: ``(h, c)`` tuple; GRU: ``h`` tensor).

        Returns:
            Tuple of ``(delta_enc, delta_dec)`` weight residuals.
        """
        rnn_out = self.rnn(u_window, state)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        # Input energy gate: when u_window=0, u_rms=0 -> deltas=0 exactly
        if self.use_input_gate:
            u_rms = torch.sqrt(torch.mean(u_window ** 2, dim=(1, 2), keepdim=True)).squeeze(-1)  # (batch, 1)
            delta_enc = delta_enc * u_rms
            delta_dec = delta_dec * u_rms
        return delta_enc, delta_dec

    def step(
        self,
        u_t: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Process single time step, return delta weights and new RNN state.

        Args:
            u_t: Single input step of shape ``(batch, 1, n_u)``.
            state: RNN state (LSTM: ``(h, c)`` tuple; GRU: ``h`` tensor).

        Returns:
            Tuple of ``(delta_enc, delta_dec, state)``.
        """
        rnn_out = self.rnn(u_t, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        # Input energy gate
        if self.use_input_gate:
            u_rms = torch.sqrt(torch.mean(u_t ** 2) + 1e-12)
            delta_enc = delta_enc * u_rms
            delta_dec = delta_dec * u_rms
        return delta_enc, delta_dec, new_state

    def forward_with_hidden(
        self,
        u_window: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Process input window, return delta weights and RNN state.

        Returns:
            Tuple of ``(delta_enc, delta_dec, state)``.
        """
        rnn_out = self.rnn(u_window, state)
        new_state = self._get_state(rnn_out)
        h = self._extract_hidden(rnn_out).squeeze(0)
        delta_enc, delta_dec = self._generate_deltas(h)
        # Input energy gate
        if self.use_input_gate:
            u_rms = torch.sqrt(torch.mean(u_window ** 2, dim=(1, 2), keepdim=True)).squeeze(-1)  # (batch, 1)
            delta_enc = delta_enc * u_rms
            delta_dec = delta_dec * u_rms
        return delta_enc, delta_dec, new_state
