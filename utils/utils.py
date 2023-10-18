import os
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
from addict import Dict


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


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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
        elif score < self.best_score:
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
        torch.save(model.state_dict(), ckpt_name)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

class Ramper:

    def __init__(self, rampdown_length):
        self.curr = 0
        self.rampdown_length = rampdown_length
        self.ramp = 0

    def cosine_rampdown(self):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        self.curr += 1
        if 0 <= self.curr <= self.rampdown_length:
            self.ramp = 1 - float(.5 * (np.cos(np.pi * self.curr / self.rampdown_length) + 1))
        else:
            self.ramp = 1

