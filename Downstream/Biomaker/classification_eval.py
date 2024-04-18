import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from utils.utils import read_yaml
from datasets.dataloader_factory import create_dataloader
from training_methods.embedding_general import evaluation
from models.CHIEF import CHIEF_biomaker

def load_model(cfg):
    model = CHIEF_biomaker(n_classes=cfg.Data.n_classes, **cfg.Model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/colon.yaml')
parser.add_argument('--dataset_name', type=str, default='test_set')
parser.add_argument('--decimals', type=int, default=4)
args = parser.parse_args()
decimals = args.decimals

if __name__ == '__main__':

    cfg = read_yaml(args.config_path)
    model = load_model(cfg)
    result_dir = os.path.join(cfg.General.result_dir,
                              'evaluation',args.dataset_name)

    os.makedirs(result_dir, exist_ok=True)

    for i in range(cfg.General.fold_num):
        dataloader = create_dataloader(i, args.dataset_name, cfg, result_dir)
        model.load_state_dict(
            torch.load('./weights/IDH/s_'+str(i)+'_checkpoint.pt'))
        evaluation(i, model, dataloader, result_dir, cfg)

    result = {'auc': []}
    all_labels = []
    for i in range(cfg.General.fold_num):
        df = pd.read_csv(os.path.join(result_dir, f'preds_{i}.csv'))
        all_labels.append(df['label'].values)
        label = df['label'].values
        prob = df['prob_1'].values
        auc = np.around(roc_auc_score(label, prob), decimals=decimals)
        result['auc'].append(auc)


    result['auc'].append(np.around(np.array(result['auc']).mean(), decimals=decimals).astype(str) + '+' + np.around(np.array(result['auc']).std(), decimals=decimals).astype(str))

    df = pd.DataFrame(result)
    print(result)
    df.to_csv(os.path.join(result_dir, 'metrics.csv'), index=False, encoding='gbk')








