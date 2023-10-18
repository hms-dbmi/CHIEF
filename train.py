import argparse
import os
import shutil

import torch

from utils.utils import read_yaml, seed_torch
from utils.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = read_yaml(args.config_path)

    result_dir = os.path.join(cfg.General.work_dir, args.config_path.split('/')[-1])
    os.makedirs(result_dir, exist_ok=True)
    if not os.path.exists(os.path.join(result_dir, args.config_path.split('/')[-1])):
        shutil.copy(args.config_path, result_dir)

    trainer = Trainer(cfg, result_dir)
    for i in range(args.begin, args.end):
        seed_torch(device, cfg.General.seed + i)
        trainer.train(i)
    trainer.eval()








