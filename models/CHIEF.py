import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
# from utils.loss import loss_fg,loss_bg,SupConLoss
# from utils.memory import Memory
#
# from transformers import CLIPTokenizer, CLIPTextModel


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


# clip_tokenizer = CLIPTokenizer.from_pretrained()
#
# clip_tokenizer = CLIPTokenizer.from_pretrained('./')
#
# text_encoder = CLIPTextModel.from_pretrained(
#     './',
#     subfolder="text_encoder")





class CHIEF(nn.Module):
    def __init__(self, gate=True, size_arg="large", dropout=True, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(),**kwargs):
        super(CHIEF, self).__init__()
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
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        initialize_weights(self)

        self.att_head = Att_Head(size[1],size[2])
        self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

        self.register_buffer('organ_embedding', torch.randn(19, 768))
        word_embedding = torch.load(r'./model_weight/Text_emdding.pth')
        self.organ_embedding.data = word_embedding.float()
        self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h, x_anatomic):
        batch = x_anatomic
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)

        embed_batch = self.organ_embedding[batch]
        embed_batch=self.text_to_vision(embed_batch)
        WSI_feature = torch.mm(A, h)

        M = WSI_feature+embed_batch

        logits = self.classifiers(M)

        result = {
            'bag_logits': logits,
            'attention_raw': A_raw,
            'WSI_feature': WSI_feature,
            'WSI_feature_anatomical': M
        }
        return result


    def patch_probs(self, h,x_anatomic):
        batch = x_anatomic
        A, h = self.attention_net(h)
        A_raw = A
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        embed_batch = self.organ_embedding[batch]
        embed_batch=self.text_to_vision(embed_batch)
        M = torch.mm(A, h)
        M = M+embed_batch
        bag_logits = self.classifiers(M)
        bag_prob = torch.softmax(bag_logits.squeeze(), dim=0)
        patch_logits = self.classifiers(h+embed_batch)
        patch_prob = torch.sigmoid(A_raw.squeeze()) * torch.softmax(patch_logits, dim=1)[:, 1]

        return{
            'bag_prob': bag_prob,
            'patch_prob': patch_prob,
            'attention_raw': A_raw.squeeze()
        }

