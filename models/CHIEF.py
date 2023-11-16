import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from utils.loss import loss_fg,loss_bg,SupConLoss
from utils.memory import Memory

from transformers import CLIPTokenizer, CLIPTextModel


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


clip_tokenizer = CLIPTokenizer.from_pretrained()

clip_tokenizer = CLIPTokenizer.from_pretrained('./')

text_encoder = CLIPTextModel.from_pretrained(
    './',
    subfolder="text_encoder")

class CHIEF(nn.Module):
    def __init__(self, gate=True, size_arg="large", dropout=True, k_sample=4, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,pos_num=1024,neg_num=32768,hard_neg_num=1024,**kwargs):
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
        self.instance_classifiers = nn.ModuleList(instance_classifiers)  # 将多个分类器储存在一起
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


        initialize_weights(self)

        self.att_head = Att_Head(size[1],size[2])

        ######
        self.pos_mem1=Memory(size[1],pos_num)
        self.neg_mem1 = Memory(size[1], neg_num)
        self.hard_neg_mem1 = Memory(size[1], hard_neg_num)


        self.pos_mem2=Memory(size[1],pos_num)
        self.neg_mem2 = Memory(size[1], neg_num)
        self.hard_neg_mem2 = Memory(size[1], hard_neg_num)


        self.pos_mem3=Memory(size[1],pos_num)
        self.neg_mem3 = Memory(size[1], neg_num)
        self.hard_neg_mem3 = Memory(size[1], hard_neg_num)


        self.pos_mem4=Memory(size[1],pos_num)
        self.neg_mem4 = Memory(size[1], neg_num)
        self.hard_neg_mem4 = Memory(size[1], hard_neg_num)


        self.pos_mem5=Memory(size[1],pos_num)
        self.neg_mem5 = Memory(size[1], neg_num)
        self.hard_neg_mem5 = Memory(size[1], hard_neg_num)


        self.pos_mem6=Memory(size[1],pos_num)
        self.neg_mem6 = Memory(size[1], neg_num)
        self.hard_neg_mem6 = Memory(size[1], hard_neg_num)


        self.pos_mem7=Memory(size[1],pos_num)
        self.neg_mem7 = Memory(size[1], neg_num)
        self.hard_neg_mem7 = Memory(size[1], hard_neg_num)


        self.pos_mem8=Memory(size[1],pos_num)
        self.neg_mem8 = Memory(size[1], neg_num)
        self.hard_neg_mem8 = Memory(size[1], hard_neg_num)

        self.pos_mem9=Memory(size[1],pos_num)
        self.neg_mem9 = Memory(size[1], neg_num)
        self.hard_neg_mem9 = Memory(size[1], hard_neg_num)

        self.pos_mem10=Memory(size[1],pos_num)
        self.neg_mem10 = Memory(size[1], neg_num)
        self.hard_neg_mem10 = Memory(size[1], hard_neg_num)

        self.pos_mem11=Memory(size[1],pos_num)
        self.neg_mem11 = Memory(size[1], neg_num)
        self.hard_neg_mem11 = Memory(size[1], hard_neg_num)


        self.pos_mem12=Memory(size[1],pos_num)
        self.neg_mem12 = Memory(size[1], neg_num)
        self.hard_neg_mem12 = Memory(size[1], hard_neg_num)

        self.pos_mem13=Memory(size[1],pos_num)
        self.neg_mem13 = Memory(size[1], neg_num)
        self.hard_neg_mem13 = Memory(size[1], hard_neg_num)

        self.pos_mem14=Memory(size[1],pos_num)
        self.neg_mem14 = Memory(size[1], neg_num)
        self.hard_neg_mem14 = Memory(size[1], hard_neg_num)

        self.pos_mem15=Memory(size[1],pos_num)
        self.neg_mem15 = Memory(size[1], neg_num)
        self.hard_neg_mem15 = Memory(size[1], hard_neg_num)

        self.pos_mem16=Memory(size[1],pos_num)
        self.neg_mem16 = Memory(size[1], neg_num)
        self.hard_neg_mem16 = Memory(size[1], hard_neg_num)

        self.pos_mem17=Memory(size[1],pos_num)
        self.neg_mem17 = Memory(size[1], neg_num)
        self.hard_neg_mem17 = Memory(size[1], hard_neg_num)

        self.pos_mem18=Memory(size[1],pos_num)
        self.neg_mem18 = Memory(size[1], neg_num)
        self.hard_neg_mem18 = Memory(size[1], hard_neg_num)

        self.pos_mem19=Memory(size[1],pos_num)
        self.neg_mem19 = Memory(size[1], neg_num)
        self.hard_neg_mem19 = Memory(size[1], hard_neg_num)

        self.sup_loss=SupConLoss()
        self.text_to_vision=nn.Sequential(nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25))

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    def inst_eval(self, A, h,label_cpu, classifier,i,epoch):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if len(h) < 10 * self.k_sample:
            k = 1
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        contra_loss=0.0
        if epoch>1:
            if i==0:

                easy_neg = nn.functional.normalize(getattr(self, f'neg_mem{label_cpu}')._return_queue(), dim=1)
                hard_neg = nn.functional.normalize( getattr(self,f'hard_neg_mem{label_cpu}')._return_queue(), dim=1)
                pos_sample = nn.functional.normalize(getattr(self,f'pos_mem{label_cpu}')._return_queue(), dim=1)

                contra_pos_label = self.create_positive_targets(pos_sample.shape[0], device)
                contra_hard_neg_label = self.create_negative_targets(hard_neg.shape[0], device)
                contra_easy_neg_label = self.create_negative_targets(easy_neg.shape[0], device)

                contra_pos_hard_label=torch.cat([contra_pos_label,contra_hard_neg_label])
                contra_pos_hard_fea=torch.cat([pos_sample,hard_neg],dim=0).unsqueeze(dim=1)

                contra_pos_easy_label=torch.cat([contra_pos_label,contra_easy_neg_label])
                contra_pos_easy_fea=torch.cat([pos_sample,easy_neg],dim=0).unsqueeze(dim=1)
                contra_loss=contra_loss+self.sup_loss(contra_pos_hard_fea,contra_pos_hard_label)+self.sup_loss(contra_pos_easy_fea,contra_pos_easy_label)

                getattr(self, f'hard_neg_mem{label_cpu}')._dequeue_and_enqueue(top_p)

            elif i==1:

                easy_neg = nn.functional.normalize(getattr(self, f'neg_mem{label_cpu}')._return_queue(), dim=1)
                hard_neg = nn.functional.normalize( getattr(self,f'hard_neg_mem{label_cpu}')._return_queue(), dim=1)
                pos_sample = nn.functional.normalize(getattr(self,f'pos_mem{label_cpu}')._return_queue(), dim=1)

                contra_pos_label = self.create_positive_targets(pos_sample.shape[0], device)
                contra_hard_neg_label = self.create_negative_targets(hard_neg.shape[0], device)
                contra_easy_neg_label = self.create_negative_targets(easy_neg.shape[0], device)

                contra_pos_hard_label = torch.cat([contra_pos_label, contra_hard_neg_label])
                contra_pos_hard_fea = torch.cat([pos_sample, hard_neg], dim=0).unsqueeze(dim=1)

                contra_pos_easy_label = torch.cat([contra_pos_label, contra_easy_neg_label])
                contra_pos_easy_fea = torch.cat([pos_sample, easy_neg], dim=0).unsqueeze(dim=1)

                contra_loss = contra_loss+self.sup_loss(contra_pos_hard_fea, contra_pos_hard_label) + self.sup_loss(
                    contra_pos_easy_fea, contra_pos_easy_label)
                getattr(self,f'pos_mem{label_cpu}')._dequeue_and_enqueue(top_p)

        else:

            if i == 0:
                getattr(self,f'hard_neg_mem{label_cpu}')._dequeue_and_enqueue(top_p)
            elif i == 1:
                getattr(self,f'pos_mem{label_cpu}')._dequeue_and_enqueue(top_p)

        return instance_loss, all_preds, all_targets,contra_loss


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

    def forward(self, h, x_anatomic,text,epoch=0,label=None, instance_eval=False, return_features=False, attention_only=False,):
        tokenized = clip_tokenizer(text,
                                   padding="max_length",
                                   max_length=77,
                                   truncation=False,
                                   return_tensors="pt",
                                   )

        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        penultimate_state = text_encoder(captions, attention_mask)
        text_embed = penultimate_state.pooler_output
        embed_batch=self.text_to_vision(text_embed)

        label_cpu=x_anatomic.detach().cpu().numpy()[0]+1
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:

            one_hot_label= F.one_hot(label, num_classes=self.n_classes).squeeze()


            if one_hot_label[0].item()==1:

                getattr(self, f'neg_mem{label_cpu}')._dequeue_and_enqueue(h)



        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets,contra_loss = self.inst_eval(A, h,label_cpu, classifier,i,epoch)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)


        M = torch.mm(A, h)

        M = M+embed_batch   ###fusion

        logits = self.classifiers(M)  # 1 * K
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        result = {
            'bag_logits': logits,
            'attention_raw': A_raw,
            'M': M
        }
        if instance_eval:
            result['contra_loss']=contra_loss

        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
            result['inst_loss'] = total_inst_loss
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return result
