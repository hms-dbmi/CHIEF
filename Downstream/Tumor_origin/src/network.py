import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



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


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x 1, N * D


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


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

class CHIEF_Tumor_origin(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=True, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(),**kwargs):
        super(CHIEF_Tumor_origin, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}
        size = self.size_dict[size_arg]
        print(size)
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
        self.n_classes = n_classes
        initialize_weights(self)


        # self.att_head = Att_Head(size[1],size[2])
        # self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))
        #
        # self.register_buffer('organ_embedding', torch.randn(19, 768))
        # word_embedding = torch.load(r'./19_text_emdding.pth')
        # self.organ_embedding.data = word_embedding.float()
        # self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)


    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch


    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h,epoch=0,label=None, instance_eval=False, return_features=False, attention_only=False,):
        h=h.squeeze(0)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # A: 1 * N h: N * 512 => M: 1 * 512

        logits = self.classifiers(M)  # 1 * K
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob