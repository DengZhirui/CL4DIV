from collections import *
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from list_cl.BertList import ListContrastive

MAXDOC = 50


class SelfAttnEnc(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dropout):
        super(SelfAttnEnc, self).__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=400, dropout=dropout,
                                                    batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=nlayers)

    def forward(self, input, mask):
        enc_out = self.enc(input, src_key_padding_mask=mask) # input [bs , seq_len , d_model]
        return enc_out # enc_out [bs , seq_len , d_model]


class DESA(nn.Module):
    def __init__(self, doc_d_model, doc_nhead, doc_nlayers, sub_d_model, sub_nhead, sub_nlayers, nhead, dropout, temperature, model_path_cll):
        super(DESA, self).__init__()
        LINEAR_OUT = 256
        self.linear_out = LINEAR_OUT
        self.linear1 = nn.Linear(doc_d_model, LINEAR_OUT)
        self.linear2 = nn.Linear(sub_d_model, LINEAR_OUT)
        self.linear3 = nn.Linear(18, 1)
        self.linear4 = nn.Linear(18+LINEAR_OUT+LINEAR_OUT+10, 1)
        self.model_path_cll = model_path_cll
        if model_path_cll != None:
            self.doc_attn = ListContrastive(temperature, doc_nhead, doc_nlayers, dropout)
            self.doc_attn.load_state_dict(torch.load(model_path_cll))
        else:
            self.doc_attn = SelfAttnEnc(LINEAR_OUT, doc_nhead, doc_nlayers, dropout) # [bs , seq_len , d_model]
        self.sub_attn = SelfAttnEnc(LINEAR_OUT, sub_nhead, sub_nlayers, dropout) # [bs , seq_len , d_model]
        self.dec_attn = nn.MultiheadAttention(LINEAR_OUT, nhead, dropout=dropout, batch_first=True)
        #self.classifier = MLP(self.linear_out*3, self.linear_out, 1)
        #self.loss_weights = nn.Parameter(torch.ones(2))

    #def weighted_loss(self, loss, index):
    #    return (loss / (self.loss_weights[index] * 2)) + (self.loss_weights[index] + 1).log()

    def forward(self, doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat, index_i=None, index_j=None,
        neg_qrel_feat=None, neg_subrel_feat=None, subrel_mask=None, intent_doci=None, intent_docj=None, mode='Train'):
        if self.model_path_cll != None:
            doc_emb = self.doc_attn.linear(doc_emb)
            doc_rep = self.doc_attn.doc_attn(doc_emb, doc_mask)  # [bs, sq(50), d_model]
        else:
            doc_emb = self.linear1(doc_emb)
            doc_rep = self.doc_attn(doc_emb, doc_mask)
        sub_rep = self.sub_attn(self.linear2(sub_emb), sub_mask)  # [bs, sq(10), d_model]
        doc_dec, _ = self.dec_attn(doc_rep, sub_rep, sub_rep) # [bs, sq(50), d_model]
        doc_dec = doc_rep
        if mode == 'Train':
            #intent_doci = torch.sum(doc_emb * intent_doci.unsqueeze(2).repeat(1, 1, self.linear_out), dim=1)
            #intent_docj = torch.sum(doc_emb * intent_docj.unsqueeze(2).repeat(1, 1, self.linear_out), dim=1)
            #intent_emb = self.classifier(
            #    torch.cat([intent_doci, intent_docj, torch.abs(intent_doci - intent_docj)], dim=-1))
            pos_index_select1 = torch.index_select(doc_rep.reshape((-1, doc_rep.shape[2])), 0,
                                                   (index_i.cuda() + torch.linspace(0, doc_rep.shape[0] - 1,
                                                doc_rep.shape[0]).cuda() * torch.tensor(
                                                       doc_rep.shape[0] + 1).cuda()).long())
            pos_index_select2 = torch.index_select(doc_dec.reshape((-1, doc_dec.shape[2])), 0,
                                                   (index_i.cuda() + torch.linspace(0, doc_dec.shape[0] - 1,
                                                   doc_dec.shape[0]).cuda() * torch.tensor(
                                                       doc_dec.shape[0] + 1).cuda()).long())
            pos_concat = torch.cat([pos_qrel_feat, pos_index_select1, pos_index_select2,
                                    self.linear3(pos_subrel_feat).squeeze(2)], dim=1)  # pos_subrel[bs, sq(10), 18]
            pos_out = self.linear4(pos_concat)
            neg_index_select1 = torch.index_select(doc_rep.reshape((-1, doc_rep.shape[2])), 0,
                    (index_j.cuda() + torch.linspace(0, doc_rep.shape[0] - 1, doc_rep.shape[0]).cuda() *
                     torch.tensor(doc_rep.shape[0] + 1).cuda()).long())
            neg_index_select2 = torch.index_select(doc_dec.reshape((-1, doc_dec.shape[2])), 0,
                    (index_j.cuda() + torch.linspace(0, doc_dec.shape[0] - 1, doc_dec.shape[0]).cuda() * torch.tensor(
                                                                doc_dec.shape[0] + 1).cuda()).long())

            neg_concat = torch.cat([neg_qrel_feat, neg_index_select1,
                                    neg_index_select2, self.linear3(neg_subrel_feat).squeeze(2)], dim=1)
            neg_out = self.linear4(neg_concat)
            #acc, loss = self.list_pairwise_loss(pos_out, neg_out, torch.sigmoid(intent_emb).squeeze(-1), weight, intent_label)
            return pos_out, neg_out#, torch.sigmoid(intent_emb).squeeze(-1)
        else:
            # [1,50,18]/[1,50,256]/[1,50,256]/[1,50,10]
            #print(pos_qrel_feat.shape, doc_rep.shape, doc_dec.shape, pos_subrel_feat.shape, self.linear3(pos_subrel_feat).shape)
            pos_concat = torch.cat([pos_qrel_feat, doc_rep, doc_dec,
                                    self.linear3(pos_subrel_feat).squeeze(3)], dim=2)  # pos_subrel[bs, sq(10), 18]
            pos_out = self.linear4(pos_concat)
            return pos_out.squeeze(2).squeeze(0) # [50]


class MLP(nn.Module):
    def __init__(self, input_size, hid_size, output_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, output_size)
        )

    def forward(self, input):
        output = self.mlp(input)
        return output


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
