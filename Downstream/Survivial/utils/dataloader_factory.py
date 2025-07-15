import os
import random
import shutil
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


def create_dataloader(index, dataset, cfg, result_dir):
    if cfg.Data.dataset_name in ['BagDataset', 'TwoStreamBagDataset', 'SurvivalBagDataset']:
        return create_bag_dataloader(index, dataset, cfg, result_dir)
    else:
        raise NotImplementedError

def create_bag_dataloader(index, dataset_name, cfg, result_dir):
    df = pd.read_csv(os.path.join(cfg.Data.split_dir, f'split_{index}.csv'))
    from datasets import SurvivalBagDataset
    dataset = SurvivalBagDataset(df,istrain=False, **cfg.Data)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker)

    return dataloader
