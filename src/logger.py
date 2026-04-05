"""
Unified experiment logger supporting TensorBoard and Weights & Biases.

Usage:
    logger = ExperimentLogger(log_dir="results/run1", use_tb=True, use_wandb=False)
    logger.log_scalar("loss/train", 0.5, step=10)
    logger.log_config(config_dict)
    logger.close()

Standalone test:
    python -m src.logger
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Logs scalars, configs, and artifacts to TensorBoard and/or W&B."""

    def __init__(
        self,
        log_dir: str,
        use_tb: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "hyperkkl",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.wandb_run = None

        # TensorBoard
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            except ImportError:
                print("[Logger] tensorboard not installed, skipping TB logging")

        # Weights & Biases
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=str(self.log_dir),
                    reinit=True,
                )
            except ImportError:
                print("[Logger] wandb not installed, skipping W&B logging")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log multiple scalar values at once."""
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_config(self, config: dict):
        """Log experiment configuration."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.wandb_run is not None:
            import wandb
            wandb.config.update(config, allow_val_change=True)

    def log_image(self, tag: str, image_path: str, step: int):
        """Log an image file."""
        if self.tb_writer is not None:
            try:
                import matplotlib.image as mpimg
                import numpy as np
                img = mpimg.imread(image_path)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                # TB expects (C, H, W)
                self.tb_writer.add_image(tag, img.transpose(2, 0, 1), step)
            except Exception:
                pass

        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Image(image_path)}, step=step)

    def log_text(self, tag: str, text: str, step: int = 0):
        """Log a text string."""
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)

    def close(self):
        """Flush and close all writers."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


# Null logger for when logging is disabled
class NullLogger(ExperimentLogger):
    """No-op logger that silently ignores all calls."""

    def __init__(self):
        self.log_dir = Path("/dev/null")
        self.tb_writer = None
        self.wandb_run = None

    def log_scalar(self, tag, value, step):
        pass

    def log_scalars(self, scalars, step):
        pass

    def log_config(self, config):
        pass

    def log_image(self, tag, image_path, step):
        pass

    def log_text(self, tag, text, step=0):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        logger = ExperimentLogger(log_dir=d, use_tb=True, use_wandb=False)
        for i in range(50):
            logger.log_scalar("test/loss", 1.0 / (i + 1), step=i)
        logger.log_config({"lr": 0.001, "epochs": 50})
        logger.close()
        print(f"[Logger] Test passed. Logs written to {d}")
