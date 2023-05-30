import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class SelfAttnEnc(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dropout):
        super(SelfAttnEnc, self).__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=400, dropout=dropout,
                                                    batch_first=True)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=nlayers)

    def forward(self, input, mask):
        enc_out = self.enc(input, src_key_padding_mask=mask) # input [bs , seq_len , d_model]
        return enc_out # enc_out [bs , seq_len , d_model]


class ListContrastive(nn.Module):
    def __init__(self, temperature, doc_nhead, doc_nlayers, dropout=0):
        super(ListContrastive, self).__init__()
        LINEAR_OUT = 256
        self.linear = nn.Linear(768, LINEAR_OUT)
        self.doc_attn = SelfAttnEnc(LINEAR_OUT, doc_nhead, doc_nlayers, dropout)
        self.temperature = temperature
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        init.xavier_normal_(self.linear.weight)

    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """

        batch_size = batch_data["input_ids"].size(0)

        input_ids = batch_data["input_ids"].float()
        attention_mask = batch_data["attention_mask"]
        sent_rep1 = self.doc_attn(self.linear(input_ids), attention_mask).reshape(batch_size, -1)

        input_ids_pos = batch_data["input_ids_pos"].float()
        attention_mask_pos = batch_data["attention_mask_pos"]
        sent_rep2 = self.doc_attn(self.linear(input_ids_pos), attention_mask_pos).reshape(batch_size, -1)

        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # [batch]
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)  # [batch]

        batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (
                    torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (
                    torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]

        batch_self_11 = batch_self_11 / self.temperature
        batch_cross_12 = batch_cross_12 / self.temperature

        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)
        batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_first.shape[1]) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]

        batch_self_22 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep2) / (
                    torch.einsum("ad,bd->ab", sent_norm2, sent_norm2) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep1) / (
                    torch.einsum("ad,bd->ab", sent_norm2, sent_norm1) + 1e-6)  # [batch, batch]

        batch_self_22 = batch_self_22 / self.temperature
        batch_cross_21 = batch_cross_21 / self.temperature

        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        contras_loss = self.cl_loss(batch_predict, batch_label)

        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        return contras_loss, acc

