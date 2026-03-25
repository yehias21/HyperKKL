#!/usr/bin/env python
"""
Standalone Phase 1 Training — trains autonomous T (encoder) and T* (decoder).

Saves checkpoints to an output directory so Phase 2 can load them directly
via --phase1_dir without retraining.

Usage:
    # Train Phase 1 for duffing
    python -m hyperkkl.scripts.train_phase1 --system duffing --out_dir results/phase1/duffing

    # Train with custom settings
    python -m hyperkkl.scripts.train_phase1 --system lorenz --epochs 50 --lr 5e-4

    # Then run Phase 2 using the saved Phase 1
    python -m hyperkkl.scripts.run_pipeline --system duffing --method dynamic_lora \
        --phase1_dir results/phase1/duffing
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from hyperkkl.src.training.configs import get_system_config
from hyperkkl.src.training.phase1 import train_autonomous


def main():
    parser = argparse.ArgumentParser(
        description='Standalone Phase 1: Train autonomous T and T*')

    parser.add_argument(
        '--system', type=str, required=True,
        choices=['duffing', 'vdp', 'lorenz', 'rossler',
                 'fitzhugh_nagumo', 'highway_traffic'],
        help='Dynamical system to train on')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs for both T and T*')
    parser.add_argument('--batch_size', type=int, default=2049)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_ic', type=int, default=200,
                        help='Number of initial conditions for data generation')
    parser.add_argument('--num_hidden', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--lambda_pde', type=float, default=1.0,
                        help='PDE loss weight')
    parser.add_argument('--no_pde', action='store_true',
                        help='Disable PDE loss (train with MSE only)')
    parser.add_argument('--no_split_traj', action='store_true',
                        help='Disable split-trajectory PINN sampling')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable data normalization')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: results/phase1/{system})')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sys_config = get_system_config(args.system)
    system = sys_config['class'](**sys_config['init_args'])

    train_cfg = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_ic': args.num_ic,
        'num_hidden': args.num_hidden,
        'hidden_size': args.hidden_size,
        'use_pde': not args.no_pde,
        'lambda_pde': args.lambda_pde,
        'seed': args.seed,
        'split_traj': not args.no_split_traj,
        'normalize': not args.no_normalize,
    }

    # Train
    T_net, T_inv_net, loss_history = train_autonomous(
        system, sys_config, train_cfg, device)

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else Path(f'results/phase1/{args.system}')
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({'model': T_net.state_dict()}, out_dir / 'T_encoder.pt')
    torch.save({'model': T_inv_net.state_dict()}, out_dir / 'T_inv_decoder.pt')

    import json
    with open(out_dir / 'phase1_config.json', 'w') as f:
        json.dump({
            'system': args.system,
            'train_cfg': {k: v for k, v in train_cfg.items()},
            'sys_config': {
                k: v for k, v in sys_config.items()
                if k not in ('class', 'M', 'K', 'limits')
            },
            'encoder_losses': loss_history['encoder'],
            'decoder_losses': loss_history['decoder'],
        }, f, indent=2)

    print(f"\nPhase 1 checkpoints saved to {out_dir}/")
    print(f"  T_encoder.pt, T_inv_decoder.pt, phase1_config.json")
    print(f"\nTo use in Phase 2:")
    print(f"  python -m hyperkkl.scripts.run_pipeline --system {args.system} "
          f"--method all --phase1_dir {out_dir}")


if __name__ == '__main__':
    main()
