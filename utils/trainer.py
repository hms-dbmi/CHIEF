import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.utils import EarlyStopping

class Trainer:

    def __init__(self, cfg, result_dir):
        self.cfg = cfg
        self.result_dir = result_dir

    def get_dataloader(self, index):
        cfg = self.cfg

        train_set_df = []
        val_set_df = []
        if cfg.Train.mode == 'cross_validation':
            for i in range(cfg.General.fold_num):
                if i == index:
                    val_set_df = pd.read_csv(os.path.join(cfg.Data.split_dir, f'split_{i}.csv'))
                else:
                    train_set_df.append(pd.read_csv(os.path.join(cfg.Data.split_dir, f'split_{i}.csv')))
            train_set_df = pd.concat(train_set_df, axis=0)
        elif cfg.Train.mode == 'repeat':
            train_set_df = pd.read_csv(os.path.join(cfg.Data.split_dir, 'train_set.csv'))
            val_set_df = pd.read_csv(os.path.join(cfg.Data.split_dir, 'val_set.csv'))

        if cfg.Train.dataset == 'BagDataset':
            from datasets.BagDataset import BagDataset
            train_set = BagDataset(train_set_df, cfg.Data.data_dir)
            val_set = BagDataset(val_set_df, cfg.Data.data_dir)
        elif cfg.Train.dataset == 'TwoStreamBagDataset':
            from datasets.TwoStreamBagDataset import TwoStreamDataset
            train_set = TwoStreamDataset(train_set_df, cfg.Data.data_dir_1, cfg.Data.data_dir_2)
            val_set = TwoStreamDataset(val_set_df, cfg.Data.data_dir_1, cfg.Data.data_dir_2)
        else:
            raise NotImplementedError
        train_loader = DataLoader(train_set, batch_size=None, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=None, shuffle=False, num_workers=0)

        return train_loader, val_loader

    def get_testloader(self, test_set_name='test_set', **kwargs):
        cfg = self.cfg


        if test_set_name == 'test_set':
            if cfg.Train.mode == 'cross_validation':
                raise NotImplementedError
            elif cfg.Train.mode == 'repeat':
                test_set_df = pd.read_csv(os.path.join(cfg.Data.split_dir, 'test_set.csv'))
            else:
                raise NotImplementedError
        else:
            test_set_df = pd.read_csv(cfg.Test[test_set_name].csv_path)

        if cfg.Train.dataset == 'BagDataset':
            from datasets.BagDataset import BagDataset
            if test_set_name == 'test_set':
                test_set = BagDataset(test_set_df, cfg.Data.data_dir)
            else:
                test_set = BagDataset(test_set_df, cfg.Test[test_set_name].data_dir)
        elif cfg.Train.dataset == 'TwoStreamBagDataset':
            from datasets.TwoStreamBagDataset import TwoStreamDataset
            if test_set_name == 'test_set':
                test_set = TwoStreamDataset(test_set_df, cfg.Data.data_dir_1, cfg.Data.data_dir_2)
            else:
                test_set = TwoStreamDataset(test_set_df, cfg.Test[test_set_name].data_dir_1, cfg.Test[test_set_name].data_dir_2)
        else:
            raise NotImplementedError
        test_loader = DataLoader(test_set, batch_size=None, shuffle=False, num_workers=0)

        return test_loader

    def get_model(self):

        if self.cfg.Model.network == 'CHIEF':
            from models.CHIEF import CHIEF
            model = CHIEF(features_size=self.cfg.Data.features_size, n_classes=self.cfg.Data.n_classes)
        else:
            raise NotImplementedError

        return model


    def get_train_loop(self):
        cfg = self.cfg
        if cfg.Train.train_method == 'CHIEF':
            from training_methods.CHIEF import train_loop
        else:
            raise NotImplementedError

        return train_loop

    def get_validation(self):
        cfg = self.cfg
        if cfg.Train.val_method == 'CHIEF':
            from training_methods.CHIEF import validation
        else:
            raise NotImplementedError

        return validation

    def get_summary(self):
        cfg = self.cfg
        if cfg.Train.val_method == 'CHIEF':
            from training_methods.CHIEF import summary
        else:
            raise NotImplementedError

        return summary

    def train(self, index):
        cfg = self.cfg
        train_loader, val_loader = self.get_dataloader(index)

        print(f'''
        train set num: {len(train_loader)}
        val set num: {len(val_loader)}
        ''')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'training fold {index}')

        # tensorboardX writer
        writer_dir = os.path.join(self.result_dir, 'log', str(index))
        if not os.path.isdir(writer_dir):
            os.makedirs(writer_dir)
        writer = SummaryWriter(writer_dir, flush_secs=15)

        model = self.get_model()
        model.to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Train.lr,
                                     weight_decay=cfg.Train.reg)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.Train.CosineAnnealingLR.T_max,
            eta_min=cfg.Train.CosineAnnealingLR.eta_min,
            last_epoch=-1
        )

        early_stopping = EarlyStopping(
            patience=cfg.Train.Early_stopping.patient,
            stop_epoch=cfg.Train.Early_stopping.stop_epoch,
            type=cfg.Train.Early_stopping.type
        )

        train_loop = self.get_train_loop()

        validation = self.get_validation()

        for epoch in range(cfg.Train.max_epochs):
            lr = scheduler.get_last_lr()[0]
            print('learning rate:{:.8f}'.format(lr))
            writer.add_scalar('train/lr', lr, epoch)

            train_loop(
                epoch=epoch,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                writer=writer,
            )

            stop = validation(
                cur=index,
                epoch=epoch,
                model=model,
                loader=val_loader,
                n_classes=cfg.Data.n_classes,
                results_dir=self.result_dir,
                early_stopping=early_stopping,
                early_stopping_type='max',
                writer=writer
            )

            if stop:
                break
            scheduler.step()

    def eval(self, test_set_name = 'test_set'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = self.cfg
        all_result = {
            'acc': [],
            'auc': [],
            'f1_score': [],
            'precision': [],
            'specificity': [],
            'recall': [],
            'prob': []
        }

        for i in range(cfg.General.fold_num):
            weight_path = os.path.join(self.result_dir, f's_{i}_checkpoint.pt')
            if not os.path.exists(weight_path):
                break
            model = self.get_model()
            model.load_state_dict(torch.load(weight_path))
            model.to(device)

            test_loader = self.get_testloader(test_set_name)
            summary = self.get_summary()

            result = summary(model, test_loader, cfg.Data.n_classes)
            for key, value in result.items():
                all_result[key].append(value)

        probs = all_result['prob']
        del all_result['prob']

        result_path = os.path.join(self.result_dir, 'results', test_set_name)
        os.makedirs(result_path, exist_ok=True)
        metrics_df = pd.DataFrame(all_result)
        mean = metrics_df.values.mean(axis=0)
        std = metrics_df.values.std(axis=0)
        metrics_df = pd.DataFrame(data=np.concatenate([metrics_df.values, mean[np.newaxis, :], std[np.newaxis, :]], axis=0), columns=metrics_df.columns)
        metrics_df.to_csv(os.path.join(result_path, 'metrics.csv'), index=False)

        test_set_df = test_loader.dataset.get_data_df()
        test_set_df = test_set_df[['slide_id', 'label']]
        for i, prob in enumerate(probs):
            test_set_df[f'prob_{i}'] = prob[:, 1]
        test_set_df.to_csv(os.path.join(result_path, 'probs.csv'), index=False)

        print(metrics_df)





