import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Att_Head(nn.Module):
    def __init__(self,FEATURE_DIM,ATT_IM_DIM):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
class CHIEF_survival(nn.Module):
    def __init__(self, gate=True, size_arg="large", dropout=True, k_sample=7, n_classes=4,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,pos_num=32,neg_num=32768,hard_neg_num=32,batch_size=8,**kwargs):
        super(CHIEF_survival, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, h,epoch=0,label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        #h=h.squeeze(0)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # A: 1 * N h: N * 512 => M: 1 * 512
        # M = torch.cat([M, embed_batch], axis=1)
        logits = self.classifiers(M)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        result = {
            'bag_logits': logits,
            'attention_raw': A_raw,
            'M': M
        }
        return result


if __name__ =='__main__':
    net = CHIEF_survival(size_arg='small', n_classes=2).cuda()
    t = torch.zeros([100,768]).cuda()
    res = net(t)
    print(res['bag_logits'].shape)