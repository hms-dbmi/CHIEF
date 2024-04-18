import yaml
import random
import os
import torch.nn as nn
import numpy as np
import torch
from addict import Dict



def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def seed_torch(device, seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, type='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.best_score = -99999
        self.counter = 0
        self.type = type

    def __call__(self, epoch, score, model, ckpt_name = 'checkpoint.pt'):

        if self.type == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            if self.type == 'min':
                print(f'best metric: {-self.best_score}, current metric: {-score}')
            elif self.type == 'max':
                print(f'best metric: {self.best_score}, current metric: {score}')
            else:
                raise NotImplementedError
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, ckpt_name)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.type == 'max':
            print(f'metric increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        elif self.type == 'min':
            print(f'metric decreased ({-self.best_score:.6f} --> {-score:.6f}).  Saving model ...')
        else:
            raise NotImplementedError
        if isinstance(model, dict):
            os.makedirs(ckpt_name[:-3], exist_ok=True)
            for key, value in model.items():
                torch.save(model[key].state_dict(), os.path.join(ckpt_name[:-3], key + '.pt'))
        else:
            torch.save(model.state_dict(), ckpt_name)

def load_task_state(result_dir, early_stopping, scheduler, model, index):
    task_state = read_yaml(os.path.join(result_dir, f'task_state_{index}.yaml'))
    early_stopping.counter = task_state.early_stop_count
    for _ in range(task_state.current_epoch-1):
        scheduler.step()

    state_dict_path = os.path.join(result_dir, 'last_model_state', f'checkpoint_{index}.pt')
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path))

    return task_state.current_epoch

def save_task_state(result_dir, early_stopping, current_epoch, model, index):
    task_state = {
        'current_epoch': current_epoch,
        'early_stop_count': early_stopping.counter
    }

    with open(os.path.join(result_dir, f'task_state_{index}.yaml'), "w") as f:
        yaml.dump(task_state, f)

    os.makedirs(os.path.join(result_dir, 'last_model_state'), exist_ok=True)
    state_dict_path = os.path.join(result_dir, 'last_model_state', f'checkpoint_{index}.pt')
    torch.save(model.state_dict(), state_dict_path)

