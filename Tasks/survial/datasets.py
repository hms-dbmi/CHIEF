import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

class SurvivalBagDataset(Dataset):
    def __init__(self, df, data_dir, label_field='status', extra_df=None, csv_path=None, **kwargs):
        super(SurvivalBagDataset, self).__init__()
        self.data_dir = data_dir
        self.label_field = label_field
        self.extra_df = None
        # inverse censorship 
        df.status = 1-df.status
        self.df = df

    def __len__(self):
        return len(self.df.values)

    def get_data_df(self):
        return self.df

    def get_id_list(self):
        return self.df['filename'].values
    
    def get_balance_weight(self):
        # for data balance
        label = self.df['status'].values
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

    def __getitem__(self, idx):
        if self.extra_df is None:
            label = self.df[self.label_field].values[idx]
            status = self.df['status'].values[idx]
            #patient_id = self.df['patient_id'].values[idx]
            if type(self.df['filename'])==np.float64:
                self.df['filename'] = self.df['filename'].astype(int)
            slide_id = str(self.df['filename'].values[idx])
            time = self.df['time'].values[idx]
            # load from pt files
            if 'feature_path' in self.df.columns:
                full_path = self.df['feature_path'].values[idx]
            else:
                if os.path.exists(os.path.join(self.data_dir, 'patch_feature')):
                    full_path = os.path.join(self.data_dir, 'patch_feature', slide_id if slide_id.endswith('.pt') else slide_id + '.pt')
                else:
                    full_path = os.path.join(self.data_dir, slide_id if slide_id.endswith('.pt') else slide_id + '.pt')
            features = torch.load(full_path, map_location=torch.device('cpu'))
            res = {
                'feature': features,
                'label': torch.tensor([label]),
                'time': torch.tensor(time),
                'status': torch.tensor(status)
            }
            return res
    