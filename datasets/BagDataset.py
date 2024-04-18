import os

import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, df, data_dir,anatomic, **kwargs):
        super(BagDataset, self).__init__()

        self.data_dir = data_dir
        self.df = df
        self.anatomic = anatomic
    def __len__(self):
        return len(self.df.values)

    def __getitem__(self, idx):

        slide_id = str(self.df['case_id'].values[idx])
        full_path = os.path.join(self.data_dir, slide_id + '.pt')

        features = torch.load(full_path, map_location=torch.device('cpu'))

        res = {
            'x': features,
            'z': torch.tensor([self.anatomic]),
            'id':slide_id
        }


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

    def get_id_list(self):
        return self.df['case_id'].values