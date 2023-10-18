import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, df, data_dir, return_idx=False):
        super(BagDataset, self).__init__()

        self.data_dir = data_dir
        self.df = df
        self.rt_idx = return_idx

    def __len__(self):
        return len(self.df.values)

    def __getitem__(self, idx):
        label = self.df['label'].values[idx]

        slide_id = self.df['slide_id'].values[idx]

        # load from pt files
        if not slide_id.endswith('.pt'):
            slide_id = slide_id + '.pt'
        if 'feature_path' in self.df.columns:
            full_path = self.df['feature_path'].values[idx]
        else:
            if os.path.exists(os.path.join(self.data_dir, 'patch_feature')):
                full_path = os.path.join(self.data_dir, 'patch_feature', slide_id)
            else:
                full_path = os.path.join(self.data_dir, slide_id)
        features = torch.load(full_path, map_location=torch.device('cpu'))

        res = {
            'features': features,
            'label': torch.tensor([label])
        }

        if self.rt_idx:
            res['idx'] = idx

        return res

    def get_balance_weight(self):
        # for data balance
        label = self.df['label'].values
        label_np = np.array(label)
        classes = list(set(label))
        N = len(self.df)
        num_of_classes = [(label_np==c).sum() for c in classes]
        c_weight = [N/num_of_classes[i] for i in range(len(classes))]

        weight = [0 for _ in range(N)]
        for i in range(N):
            c_index = classes.index(label[i])
            weight[i] = c_weight[c_index]

        return weight

    def get_data_df(self):
        return self.df

