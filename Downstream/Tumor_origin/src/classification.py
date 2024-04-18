import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import MultiModalDataset
from file_utils import save_pkl

from network import CHIEF_Tumor_origin



class Tumor_origin:
    def __init__(self, args):
        self.args = args

        self.counter = 0
        self.lowest_loss = np.Inf

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def set_seed(self, seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if str(self.device.type) == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_data_loader(self, split_csv):
        self.train_dataset = MultiModalDataset(self.args.gt_csv, self.args.train_csv, self.args.label_dict, self.args.histology_feature_path, split_name='train', balance_met=self.args.balance_met)
        self.valid_dataset = MultiModalDataset(self.args.gt_csv, self.args.val_csv, self.args.label_dict, self.args.histology_feature_path, split_name='valid', site_name=self.args.site_name)
        self.test_dataset = MultiModalDataset(self.args.gt_csv, self.args.test_csv, self.args.label_dict, self.args.histology_feature_path, split_name='test')
        self.train_loader = MultiModalDataset.get_data_loader(self.train_dataset, batch_size=self.args.batch_size, training=True)
        self.valid_loader = MultiModalDataset.get_data_loader(self.valid_dataset, batch_size=self.args.batch_size, training=False)
        self.test_loader = MultiModalDataset.get_data_loader(self.test_dataset, batch_size=self.args.batch_size, training=False)

    def init_model(self, is_train=False):
        self.model=CHIEF_Tumor_origin(n_classes=self.args.n_classes)

        self.model.load_state_dict(
            torch.load(
                r'../../../model_weight/CHIEF_finetune.pth'),
            strict=False)

        if is_train:
            self.model.relocate()

    def init_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)

    def init_loss_function(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def load_model(self):
        ckpt = torch.load(self.args.ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            ckpt_clean.update({key.replace('module.', ''): ckpt[key]})
        self.model.load_state_dict(ckpt_clean, strict=True)
        self.model.relocate()

    def train_valid(self):
        self.counter = 0
        self.lowest_loss = np.Inf
        self.set_seed()
        self.init_data_loader(split_csv=self.args.split_csv)
        self.init_model(is_train=True)
        self.init_optimizer()
        self.init_loss_function()

        result_dict = {'train_loss': [], 'valid_loss': [],
                       'train_acc': [], 'valid_acc': [],
                       'train_auc': [], 'valid_auc': []}
        for epoch in range(self.args.max_epochs):
            start_time = time.time()

            train_loss, train_acc, train_auc = self.train_loop(self.train_loader)
            print('\rTrain Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, train_auc: {:.4f}                 '.format(
                epoch, train_loss, train_acc, train_auc))
            result_dict['train_loss'].append(train_loss)
            result_dict['train_acc'].append(train_acc)
            result_dict['train_auc'].append(train_auc)

            valid_loss, valid_acc, valid_auc = self.valid_loop(self.valid_loader)
            print('\rValid Epoch: {}, valid_loss: {:.4f}, valid_acc: {:.4f}, valid_auc: {:.4f}                 '.format(
                epoch, valid_loss, valid_acc, valid_auc))
            result_dict['valid_loss'].append(valid_loss)
            result_dict['valid_acc'].append(valid_acc)
            result_dict['valid_auc'].append(valid_auc)

            if self.lowest_loss > valid_loss:
                print('--------------------Saving best model--------------------')
                torch.save(self.model.state_dict(), os.path.join(self.args.results_dir, 'checkpoint.pt'))
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print('Loss is not decreased in last %d epochs' % self.counter)

            if (self.counter > self.args.patience) and (epoch >= self.args.minimum_epochs):
                break

            total_time = time.time() - start_time
            print('Time to process epoch({}): {:.4f} minutes                             \n'.format(epoch, total_time/60))
            pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(self.args.results_dir, 'training_stats.csv'))

    def eval(self, split_name='test'):
        self.set_seed()

        if self.args.split_csv is None:
            split_csv = None
        else:
            split_csv = self.args.split_csv


        dataset = MultiModalDataset(self.args.gt_csv, self.args.test_csv, self.args.label_dict,self.args.histology_feature_path,
                                    split_name=split_name)

        dataset_loader = MultiModalDataset.get_data_loader(dataset, batch_size=self.args.batch_size, training=False)

        self.init_model()
        self.load_model()

        acc, auc, stats_dict = self.test_loop(dataset_loader)
        print('\rAccuracy: {:.4f}, AUC: {:.4f}                            '.format(acc*100, auc))
        classifier_name = 'Toumor_origin'
        os.makedirs(os.path.join(self.args.results_dir, classifier_name), exist_ok=True)
        save_pkl(os.path.join(self.args.results_dir, classifier_name, '%s.pkl' % split_name), stats_dict)
        return acc, auc

    def train_loop(self, data_loader):
        total_loss = 0
        gt_labels = []
        pred_labels = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.train()
        for batch_idx, (h_features_batch, label_batch, _) in enumerate(data_loader):
            if len(gt_labels) == 0:
                gt_labels = label_batch.cpu().numpy().tolist()
            else:
                gt_labels.extend(label_batch.cpu().numpy().tolist())

            h_features_batch = h_features_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            logits, probs = self.model(h_features_batch)

            # else:
            probs = probs
            loss = self.loss_fn(logits, label_batch)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.cpu().item()

            p_labels = torch.argmax(probs, dim=1)
            p_labels = p_labels.detach().cpu().numpy().tolist()
            p_probs = probs.detach().cpu().numpy()

            if len(pred_labels) == 0:
                pred_probs = p_probs
                pred_labels = p_labels
            else:
                pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                pred_labels.extend(p_labels)

            sys.stdout.write('\rTraining Batch {}/{}, avg loss: {:.4f}'.format(
                batch_idx+1, batch_count, total_loss/(batch_idx+1)))

        acc = accuracy_score(gt_labels, pred_labels)
        auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo', labels=[class_id for class_id in range(pred_probs.shape[1])])
        return total_loss/batch_count, acc, auc

    def valid_loop(self, data_loader):
        total_loss = 0
        gt_labels = []
        pred_labels = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, ( h_features_batch, label_batch, _) in enumerate(data_loader):
                if len(gt_labels) == 0:
                    gt_labels = label_batch.cpu().numpy().tolist()
                else:
                    gt_labels.extend(label_batch.cpu().numpy().tolist())

                h_features_batch = h_features_batch.to(self.device)

                label_batch = label_batch.to(self.device)
                logits, probs = self.model(h_features_batch)

                probs = probs
                loss = self.loss_fn(logits, label_batch)

                total_loss += loss.cpu().item()

                p_labels = torch.argmax(probs, dim=1)
                p_labels = p_labels.detach().cpu().numpy().tolist()
                p_probs = probs.detach().cpu().numpy()

                if len(pred_labels) == 0:
                    pred_probs = p_probs
                    pred_labels = p_labels
                else:
                    pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                    pred_labels.extend(p_labels)

                sys.stdout.write('\rValidation Batch {}/{}, avg loss: {:.4f}'.format(
                    batch_idx + 1, batch_count, total_loss / (batch_idx + 1)))

        acc = accuracy_score(gt_labels, pred_labels)
        auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo', labels=[class_id for class_id in range(pred_probs.shape[1])])
        return total_loss / batch_count, acc, auc

    def test_loop(self, data_loader):
        gt_labels = []
        pred_labels = []
        pred_probs = None
        case_list = []
        batch_count = len(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (h_features_batch, label_batch, case_batch) in enumerate(data_loader):
                if len(gt_labels) == 0:
                    gt_labels = label_batch.numpy().tolist()
                    case_list = list(case_batch)
                else:
                    gt_labels.extend(label_batch.numpy().tolist())
                    case_list.extend(list(case_batch))


                h_features_batch = h_features_batch.to(self.device)
                logits, probs = self.model(h_features_batch)
                probs = probs

                p_labels = torch.argmax(probs, dim=1)
                p_labels = p_labels.detach().cpu().numpy().tolist()
                p_probs = probs.detach().cpu().numpy()

                if len(pred_labels) == 0:
                    pred_probs = p_probs
                    pred_labels = p_labels
                else:
                    pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                    pred_labels.extend(p_labels)

                sys.stdout.write('\rTest Batch {}/{}'.format(batch_idx + 1, batch_count))

        acc = accuracy_score(gt_labels, pred_labels)
        auc = roc_auc_score(gt_labels, pred_probs, multi_class='ovo', labels=[class_id for class_id in range(pred_probs.shape[1])])
        stats_dict = {'case_names': case_list, 'gt_labels': gt_labels, 'pred_probs': pred_probs}
        return acc, auc, stats_dict