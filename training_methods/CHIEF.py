import os

import torch
import numpy as np
from tqdm import tqdm
from datasets.TwoStreamBagDataset import TwoStreamDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from utils.utils import calculate_error

def train_loop(epoch, model, loader, optimizer, writer=None):
    assert isinstance(loader.dataset, TwoStreamDataset), "dataloder is not TwoStreamDataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = 0
    train_error = 0

    with tqdm(total=len(loader), desc='train epoch: {}'.format(epoch)) as bar:
        for idx, batch in enumerate(loader):
            features_1 = batch['features_1'].to(device)
            features_2 = batch['features_2'].to(device)
            coords_1 = batch['coords_1'].to(device)
            coords_2 = batch['coords_2'].to(device)
            label = batch['label'].to(device)

            res = model(features_1, features_2, coords_1, coords_2, label)

            logits_1 = res['logits_1']
            logits_2 = res['logits_2']
            logits_3 = res['logits_3']
            inst_loss = res['inst_loss']

            loss_1 = loss_fn(logits_1, label)
            loss_2 = loss_fn(logits_2, label)
            loss_3 = loss_fn(logits_3, label)

            loss = 0.25 * loss_1 + 0.25 * loss_2 + 0.25 * loss_3 + 0.25 * inst_loss
            if idx % 20 == 0:
                bar.set_postfix({'loss' : '{:.5f}'.format(loss)})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            y_prob = torch.softmax(logits_3.squeeze(), dim=0)
            y_hat = torch.argmax(y_prob, dim=0)
            train_error += calculate_error(y_hat, label)
            bar.update(1)

    train_loss /= len(loader)
    train_error /= len(loader)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', 1 - train_error, epoch)


def validation(cur, epoch, model, loader, n_classes, results_dir=None, early_stopping=None, early_stopping_type='min', writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    val_error = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        with tqdm(total=len(loader), desc='validate epoch:{}'.format(epoch)) as bar:
            for idx, batch in enumerate(loader):
                features_1 = batch['features_1'].to(device)
                features_2 = batch['features_2'].to(device)
                coords_1 = batch['coords_1'].to(device)
                coords_2 = batch['coords_2'].to(device)
                label = batch['label'].to(device)
                res = model(features_1, features_2, coords_1, coords_2)

                logits_3 = res['logits_3']
                y_prob = torch.softmax(logits_3.squeeze(), dim=0)
                y_hat = torch.argmax(y_prob, dim=0)
                loss = loss_fn(logits_3, label)
                val_loss += loss.item()
                prob[idx] = y_prob.cpu().numpy()
                labels[idx] = label.item()

                val_error += calculate_error(y_hat, label)
                bar.update(1)

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = 0
        for i in range(n_classes):
            auc += roc_auc_score((labels == i).astype(np.int), prob[:, i])
        auc /= n_classes

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', 1-val_error, epoch)

    if early_stopping:
        assert results_dir
        if early_stopping_type == 'min':
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        elif early_stopping_type == 'max':
            early_stopping(epoch, auc, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            raise NotImplementedError
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False

def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    with torch.no_grad():
        with tqdm(total=len(loader), desc='summary') as bar:
            for idx, batch in enumerate(loader):
                features_1 = batch['features_1'].to(device)
                features_2 = batch['features_2'].to(device)
                coords_1 = batch['coords_1'].to(device)
                coords_2 = batch['coords_2'].to(device)
                label = batch['label'].to(device)
                res = model(features_1, features_2, coords_1, coords_2)

                logits_3 = res['logits_3']
                y_prob = torch.softmax(logits_3.squeeze(), dim=0)
                all_probs[idx] = y_prob.cpu().numpy()
                all_labels[idx] = label.item()
                bar.update(1)

    preds = torch.argmax(torch.from_numpy(all_probs), dim=1)
    if n_classes == 2:
        f1 = f1_score(all_labels, preds)
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 1.0
        precision = precision_score(all_labels, preds)
        recall = recall_score(all_labels, preds)
        acc = accuracy_score(all_labels, preds)
        specificity = recall_score(1-all_labels, 1-preds)
    else:
        raise NotImplementedError

    return {
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'acc': acc,
        'prob': all_probs,
        'specificity': specificity
    }


