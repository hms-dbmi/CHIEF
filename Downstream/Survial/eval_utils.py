import os
import pandas as pd
import torch
import csv
import numpy as np
from tqdm import tqdm
import torch.nn as nn
## metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

def evaluation(index, model, loader, result_dir, cfg, gc=16):
    res_df = loader.dataset.df.copy()
    n_classes = cfg.Data.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    mode = 'cox'
    survive_time_all = []
    status_all = []
    pred_each = None
    val_loss = 0
    model.to(device)
    with torch.no_grad():
        #with tqdm(total=len(loader)) as bar:
        for idx, batch in enumerate(loader):
            x, y, time, c = batch['feature'].to(device, dtype=torch.float32), \
                            batch['label'].to(device, dtype=torch.long), \
                            batch['time'].to(device), \
                            batch['status'].to(device)
            result = model(x) # (1, n_classes)
            bag_logits = result['bag_logits']
            pred = bag_logits[0][1:]
            res_df.loc[idx, 'risk'] = pred.cpu().numpy()
            iter_ = idx % gc +1
            survive_time_all.append(np.squeeze(time.cpu().numpy()))
            status_all.append(np.squeeze(c.cpu().numpy()))
            if idx == 0:
                pred_all = pred
            if iter_ == 1:
                pred_each = pred
            else:
                pred_all = torch.cat([pred_all, pred])
                pred_each = torch.cat([pred_each, pred])
            if iter_%gc==0 or idx == len(loader)-1:
                survive_time_all = np.asarray(survive_time_all)
                status_all = np.asarray(status_all)
                #bar.update(gc)
                pred_each = None
                survive_time_all = []
                status_all = []
    val_loss /= len(loader)
    max_risk_id = res_df.groupby('patient_id')['risk'].idxmax()
    new_res_df = res_df.iloc[max_risk_id]
    cindex = concordance_index_censored((1-new_res_df.status.values).astype(bool), new_res_df.time.values, new_res_df.risk.values, tied_tol=1e-08)[0]
    # restore censorship
    new_res_df.status = 1-new_res_df.status
    return new_res_df, cindex
