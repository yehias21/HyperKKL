import hydra, os
from omegaconf import DictConfig, OmegaConf
import torch, logging
from src.data_loader.dataset_creation import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.normalizer import Normalizer
from src.utils.loss import Criterion, PDELoss
from src.utils.logger import Logger
from src.model.model_generation import init_models
from src.trainer.trainer import Trainer

def run_experiment(cfg: DictConfig) -> None:
    # 1. Log the configuration
    OmegaConf.to_yaml(cfg)
    # logger = Logger(local_path=cfg.trainer.log_dir, env_path=cfg.trainer.env_path)

    # 2. Load train dataset
    train_loader = load_dataset(cfg.data, partition='train')
    normalizer = Normalizer(train_loader.dataset)

    # 3. Initialize model
    kkl_model = init_models(cfg.models)
    logging.info(f"The  model is {kkl_model}")

    pde_loss = None
    if cfg.trainer.method == 'supervised_pinn':
        pde_loss = PDELoss(train_loader.dataset.system.diff_eq, kkl_model.forward_mapper,
                           train_loader.dataset.observer.A,
                           train_loader.dataset.observer.B, normalizer=normalizer)

    loss = Criterion(cfg.trainer.loss, cfg.trainer.method, pde_loss, normalizer=normalizer)
    kkl_model.set_normalizer(normalizer)
    optimizer = torch.optim.Adam(kkl_model.learnable_params, lr=cfg.trainer.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, verbose=True)
    trainer = Trainer(cfg=cfg, model=kkl_model, criterion=loss, optimizer=optimizer, train_loader=train_loader,
                      lr_scheduler=scheduler, logger=None, epochs=cfg.trainer.epochs, device=cfg.trainer.device)

    # 4. Train
    trainer.train()
    os.makedirs(cfg.trainer.save_dir, exist_ok=True)
    # 5. Save model
    torch.save(kkl_model.state_dict(), f"{cfg.trainer.save_dir}/model_{cfg.experiment_name}.pth")
    # Test and Evaluate


if __name__ == "__main__":
    run_experiment()
