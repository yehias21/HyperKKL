"""Phase 1: Autonomous KKL model training (forward map T and inverse map T*)."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from hyperkkl.src.data.dataset import DataSet
from hyperkkl.src.models.nn import NN
from hyperkkl.src.models.normalizer import Normalizer
from hyperkkl.src.simulators.pde_utils import pde_loss


def train_autonomous(system, sys_config: dict, train_cfg: dict, device: torch.device):
    """Train autonomous KKL model (Phase 1)."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Training Autonomous KKL for {sys_config['name'].upper()}")
    print(f"{'='*60}")

    M, K = sys_config['M'], sys_config['K']

    pinn_mode = 'split traj' if train_cfg.get('split_traj', True) else 'no physics'
    trainset = DataSet(
        system, M, K, sys_config['a'], sys_config['b'], sys_config['N'],
        train_cfg['num_ic'], sys_config['limits'], seed=train_cfg['seed'],
        PINN_sample_mode=pinn_mode, data_gen_mode='forward'
    )
    if train_cfg.get('normalize', True):
        trainset.normalize()
    print(f"Dataset: {len(trainset.x_data)} samples (split_traj={train_cfg.get('split_traj', True)}, normalize={train_cfg.get('normalize', True)})")

    num_hidden = sys_config.get('num_hidden', train_cfg['num_hidden'])
    hidden_size = sys_config.get('hidden_size', train_cfg['hidden_size'])

    normalizer_T = Normalizer(trainset)
    T_net = NN(num_hidden, hidden_size,
               sys_config['x_size'], sys_config['z_size'], F.relu, normalizer=normalizer_T).to(device)

    optimizer = torch.optim.AdamW(T_net.parameters(), lr=train_cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # gentler
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(trainset, batch_size=train_cfg['batch_size'], shuffle=True)
    T_net.mode = 'normal'
    lambda_pde_max = train_cfg['lambda_pde']

    encoder_losses = []
    print(f"\nTraining T (encoder) for {train_cfg['epochs']} epochs...")
    for epoch in range(train_cfg['epochs']):
        loss_sum = 0
        for idx, (x, z, y, x_ph, y_ph) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
            x, z, y = x.to(device), z.to(device), y.to(device)
            x_ph, y_ph = x_ph.to(device), y_ph.to(device)

            z_hat = T_net(x)
            loss_mse = loss_fn(z_hat, z)

            if train_cfg.get('use_pde', True):
                batch_time = torch.zeros(x_ph.shape[0], device=device)
                loss_pde = pde_loss(T_net, x_ph, y_ph, T_net(x_ph), batch_time,
                                    system, M, K, device, reduction='mean')
                lambda_pde = lambda_pde_max * min(1.0, epoch / 5)  # linear warmup over 20 epochs
                loss = loss_mse + lambda_pde * loss_pde
            else:
                loss = loss_mse

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T_net.parameters(), 1.0)
            optimizer.step()

        avg_loss = loss_sum / (idx + 1)
        encoder_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f'  Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    # Train decoder
    inv_trainset = DataSet(
        system, M, K, sys_config['a'], sys_config['b'], sys_config['N'],
        train_cfg['num_ic'], sys_config['limits'], seed=train_cfg['seed'],
        PINN_sample_mode='no physics', pretrained_T=T_net
    )

    normalizer_Tinv = Normalizer(inv_trainset)
    T_inv_net = NN(num_hidden, hidden_size,
                   sys_config['z_size'], sys_config['x_size'], F.relu, normalizer=normalizer_Tinv).to(device)

    optimizer = torch.optim.AdamW(T_inv_net.parameters(), lr=train_cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # gentler
    train_loader = DataLoader(inv_trainset, batch_size=train_cfg['batch_size'], shuffle=True)
    T_inv_net.mode = 'normal'

    decoder_losses = []
    print(f"\nTraining T* (decoder) for {train_cfg['epochs']} epochs...")
    for epoch in range(train_cfg['epochs']):
        loss_sum = 0
        for idx, (x, z, y, _, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
            x, z = x.to(device), z.to(device)
            x_hat = T_inv_net(z)
            loss = loss_fn(x_hat, x)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(T_inv_net.parameters(), 1.0)
            optimizer.step()

        avg_loss = loss_sum / (idx + 1)
        decoder_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f'  Epoch {epoch+1}: Loss = {avg_loss:.6f}')

    return T_net, T_inv_net, {'encoder': encoder_losses, 'decoder': decoder_losses}


# ── Loading pre-trained Phase 1 models ───────────────────────────────────────

class _DummySystem:
    """Minimal object that satisfies Normalizer.check_sys size checks."""
    def __init__(self, x_size, z_size):
        self.x_size = x_size
        self.z_size = z_size


class _DummyNormalizer(nn.Module):
    """Placeholder Normalizer whose buffers are overwritten by load_state_dict."""
    def __init__(self, x_size, z_size):
        super().__init__()
        self.sys = _DummySystem(x_size, z_size)
        self.register_buffer('mean_x', torch.zeros(x_size))
        self.register_buffer('std_x', torch.ones(x_size))
        self.register_buffer('mean_z', torch.zeros(z_size))
        self.register_buffer('std_z', torch.ones(z_size))
        self.register_buffer('mean_x_ph', torch.zeros(x_size))
        self.register_buffer('std_x_ph', torch.ones(x_size))
        self.register_buffer('mean_z_ph', torch.zeros(z_size))
        self.register_buffer('std_z_ph', torch.ones(z_size))

    def check_sys(self, tensor, mode):
        if tensor.size()[1] == self.sys.x_size:
            if mode == 'physics':
                return self.mean_x_ph, self.std_x
            return self.mean_x, self.std_x
        elif tensor.size()[1] == self.sys.z_size:
            if mode == 'physics':
                return self.mean_z_ph, self.std_z_ph
            return self.mean_z, self.std_z
        raise RuntimeError('Size of tensor unmatched with any system.')

    def Normalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)
        return (tensor - mean) / std

    def Denormalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)
        return tensor * std + mean


def _load_state_dict_flexible(model: nn.Module, state_dict: dict):
    """Load state_dict, skipping keys whose shapes don't match."""
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def load_autonomous(phase1_dir: str, sys_config: dict, device: torch.device):
    """Load pre-trained Phase 1 T (encoder) and T* (decoder) from a directory.

    Args:
        phase1_dir: Path to directory containing T_encoder.pt and T_inv_decoder.pt.
        sys_config: System configuration dict (needs x_size, z_size, num_hidden, hidden_size).
        device: torch device to load onto.

    Returns:
        (T_net, T_inv_net, loss_history) — loss_history is None when loading.
    """
    phase1_dir = Path(phase1_dir)
    encoder_path = phase1_dir / 'T_encoder.pt'
    decoder_path = phase1_dir / 'T_inv_decoder.pt'

    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoints not found in {phase1_dir}. "
            f"Expected T_encoder.pt and T_inv_decoder.pt."
        )

    x_size = sys_config['x_size']
    z_size = sys_config['z_size']
    num_hidden = sys_config.get('num_hidden', 3)
    hidden_size = sys_config.get('hidden_size', 150)

    norm_T = _DummyNormalizer(x_size, z_size)
    T_net = NN(num_hidden, hidden_size, x_size, z_size, F.relu, normalizer=norm_T).to(device)

    norm_Tinv = _DummyNormalizer(x_size, z_size)
    T_inv_net = NN(num_hidden, hidden_size, z_size, x_size, F.relu, normalizer=norm_Tinv).to(device)

    ckpt_T = torch.load(encoder_path, map_location=device, weights_only=False)
    ckpt_Tinv = torch.load(decoder_path, map_location=device, weights_only=False)
    _load_state_dict_flexible(T_net, ckpt_T['model'])
    _load_state_dict_flexible(T_inv_net, ckpt_Tinv['model'])

    T_net.mode = 'normal'
    T_inv_net.mode = 'normal'
    T_net.eval()
    T_inv_net.eval()

    print(f"Loaded Phase 1 models from {phase1_dir}")
    return T_net, T_inv_net, None
