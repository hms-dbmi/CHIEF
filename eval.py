import argparse
import os
import shutil

import torch

from utils.utils import read_yaml, seed_torch
from utils.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
parser.add_argument('--dataset_name', type=str, default='test_set')
args = parser.parse_args()

cfg = read_yaml(args.config_path)
result_dir = os.path.join(cfg.General.work_dir, args.config_path.split('/')[-1])
trainer = Trainer(cfg, result_dir)
trainer.eval(args.dataset_name)
