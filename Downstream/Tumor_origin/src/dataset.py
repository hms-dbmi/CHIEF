import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler
import os

class MultiModalDataset(Dataset):
    def __init__(self, gt_csv, split_csv, label_dict,histology_features, split_name='train', site_name=None, balance_met=False):
        self.label_dict = label_dict
        self.balance_met = balance_met
        self.split_name = split_name
        self.site_name = site_name
        self.balance_met = balance_met
        self.histology_features=histology_features

        self.x = []
        self.labels = []
        self.pro_labels = []

        gt_df = pd.read_csv(split_csv)

        for i, row in gt_df.iterrows():
            self.x.append(row['case_id'])
            self.labels.append(self.label_dict[row['label']])

            self.pro_labels.append(row['site'])



    def case_level_dataset(self, gt_df, case_list, genomic_features, histology_features):
        self.x = []
        self.labels = []
        self.site_labels = []
        self.genomic_features = {}
        self.histology_features = {}
        for i, row in gt_df.iterrows():
            if (row['case_id'] in case_list):
                if row['case_id'] not in self.x:
                    flag = True
                    if not (self.site_name is None or self.site_name != 'None'):
                        flag = False
                        if row['site'].find(self.site_name) != -1:
                            flag = True
                    if flag:
                        self.x.append(row['case_id'])
                        self.labels.append(self.label_dict[row['label']])
                        if row['site'].find('Metastatic') != -1:
                            self.site_labels.append(1)
                        else:
                            self.site_labels.append(0)
                        self.genomic_features[row['case_id']] = genomic_features[row['case_id']]
                        self.histology_features[row['case_id']] = histology_features[row['case_id']]

    def stats(self, split_name='test', site_name=None):
        row_labels = []
        class_counts = []
        for label in self.label_dict.keys():
            row_labels.append(label)
            class_counts.append(self.labels.count(self.label_dict[label]))
        if site_name is not None:
            col_name = '%s_%s' % (split_name, site_name)
        else:
            col_name = split_name
        return pd.DataFrame(class_counts, index=row_labels, columns=[col_name])

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False):
        if training:
            n = float(len(dataset))
            weight = [0] * int(n)
            if dataset.balance_met:
                weight_per_class = []
                for c in range(len(dataset.label_dict)):
                    if dataset.labels.count(c) > 0:
                        weight_per_class.append(n / dataset.labels.count(c))
                    else:
                        weight_per_class.append(0)
                for idx in range(len(dataset)):
                    label = dataset.site_labels[idx]
                    weight[idx] = weight_per_class[label]
            else:
                weight_per_class = []
                for c in range(len(dataset.label_dict)):
                    if dataset.labels.count(c) > 0:
                        weight_per_class.append(n / dataset.labels.count(c))
                    else:
                        weight_per_class.append(0)
                for idx in range(len(dataset)):
                    label = dataset.labels[idx]
                    weight[idx] = weight_per_class[label]

            weight = torch.DoubleTensor(weight)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weight, len(weight)), drop_last=True, num_workers=4)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), drop_last=False, num_workers=4)

        return loader

    def __len__(self):
        print(11,len(self.x))

        return len(self.x)

    def __getitem__(self, idx):


        case_id, label,tmp_pro = self.x[idx], self.labels[idx],self.pro_labels[idx]
        full_path = os.path.join(self.histology_features, case_id + '.pt')



        h_features = torch.load(full_path)
        new_tensor = h_features



        return new_tensor, label, case_id
