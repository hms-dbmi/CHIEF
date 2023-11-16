import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

class CoxSurvLoss(object):
    def __call__(self, hazards, time, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        c = torch.FloatTensor(c).to(hazards.device)
        R_mat = torch.FloatTensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        # print(loss_cox)
        # print(R_mat)
        return loss_cox

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    
    #!! here uncensored_loss means event happens(death/progression)
    # uncensored_loss = -c * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # censored_loss = - (1 - c) * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    # neg_l = censored_loss + uncensored_loss
    # loss = (1-alpha) * neg_l + alpha * uncensored_loss
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -c * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - (1 - c) * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


def weighted_multi_class_log_loss(y_hat, y, w, classes=2):
    a = torch.log(torch.clamp(y_hat, 1e-15, 1.0 - 1e-15)).cuda()
    if classes == 2:
        b = torch.tensor([torch.clamp(torch.sum(y[:,0]), min=1e-15), torch.clamp(torch.sum(y[:,1]), min=1e-15)]).cuda()
    elif classes==5:
        b = torch.tensor([torch.clamp(torch.sum(y[:,0]), min=1e-15), torch.clamp(torch.sum(y[:,1]), min=1e-15),
                          torch.clamp(torch.sum(y[:,2]), min=1e-15),torch.clamp(torch.sum(y[:,3]), min=1e-15),torch.clamp(torch.sum(y[:,4]), min=1e-15),
                          ]).cuda()
    return torch.sum(-torch.sum(w * y * a * 1/b))

class WeightedMultiClassLogLoss(torch.nn.Module):
    def __init__(self, weights=torch.tensor([1.,1.]), classes=2):
        super(WeightedMultiClassLogLoss, self).__init__()
        self.weights = weights.cuda()
        self.classes = classes

    def forward(self, inputs, targets):
        return weighted_multi_class_log_loss(inputs, targets, self.weights, classes=self.classes)