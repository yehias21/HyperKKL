import torch, os, time, random
import numpy as np
from glob import glob
import os


def save_dataset(dataset):
    main, sub = gen_dir_time()
    pth = os.path.join('data', main, sub)
    os.makedirs(pth, exist_ok= True)
    torch.save(dataset, os.path.join(pth, 'train_wo_inp.pth'))
    return pth


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def gen_dir_time() -> tuple[str, str]:
    main_dir = time.strftime("%Y-%m-%d")
    sub_dir = time.strftime("%H-%M-%S")
    return main_dir, sub_dir


def get_files(path, extension='.tif', str_filter=None):
    if str_filter is None:
        return glob(os.path.join(path, '**', f'*{extension}'), recursive=True)
    else:
        raise NotImplementedError


def save_model(cfg, model, optimizer, scheduler):
    main, sub = gen_dir_time()
    path = cfg.save_model.path
    pth = os.path.join(path, cfg.save_model.name, main, sub)
    os.makedirs(pth)
    torch.save(model.state_dict(), os.path.join(pth, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(pth, 'optimizer.pth'))
    torch.save(scheduler.state_dict(), os.path.join(pth, 'scheduler.pth'))
    return pth
