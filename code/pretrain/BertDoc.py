import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BertContrastive(BertPreTrainedModel):
    # _keys_to_ignore_on_load_unexpected = [r"encoder.layer.11.attention.self.query"]
    # class BertContrastive(nn.Module):
    def __init__(self, config):
        super(BertContrastive, self).__init__(config)
        self.bert = BertModel(config)
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def set_parameters(self, temperature, max_seq_length, loss_para, additional_tokens):
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        self.loss_para = loss_para
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + additional_tokens)

    def initial_parameters(self, layers):
        fixed_modules = [self.bert.encoder.layer[num] for num in layers]
        for modules in fixed_modules:
            module_l = [modules.attention.self.query, modules.attention.self.key, modules.attention.self.value,
                        modules.attention.output.dense, modules.intermediate.dense, modules.output.dense]
            for module in module_l:
                self.__weight_init__(module)
            # self._init_weights(module)

    def __weight_init__(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Conv2d):
            init.xavier_normal_(module.weight)

    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        batch_size = batch_data["input_ids"].size(0)

        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                            'token_type_ids': token_type_ids}
        sent_rep = self.bert(**bert_inputs)[1]
        sent_norm = sent_rep.norm(dim=-1, keepdim=True)  # [batch]

        input_ids_pos = batch_data["input_ids_pos"]
        attention_mask_pos = batch_data["attention_mask_pos"]
        token_type_ids_pos = batch_data["token_type_ids_pos"]
        bert_inputs_pos = {'input_ids': input_ids_pos, 'attention_mask': attention_mask_pos,
                            'token_type_ids': token_type_ids_pos}
        sent_rep_pos = self.bert(**bert_inputs_pos)[1]
        sent_norm_pos = sent_rep_pos.norm(dim=-1, keepdim=True)  # [batch]

        batch_self11 = torch.einsum("ad,bd->ab", sent_rep, sent_rep) / (
                torch.einsum("ad,bd->ab", sent_norm, sent_norm) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep, sent_rep_pos) / (
                torch.einsum("ad,bd->ab", sent_norm, sent_norm_pos) + 1e-6)  # [batch, batch]
        batch_self11 = batch_self11 / self.temperature[0]
        batch_cross_12 = batch_cross_12 / self.temperature[0]
        batch_first = torch.cat([batch_self11, batch_cross_12], dim=-1)  # [batch, batch * 2]
        batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]

        batch_self22 = torch.einsum("ad,bd->ab", sent_rep_pos, sent_rep_pos) / (
                torch.einsum("ad,bd->ab", sent_norm_pos, sent_norm_pos) + 1e-6)  # [batch, batch]
        batch_cross21 = torch.einsum("ad,bd->ab", sent_rep_pos, sent_rep) / (
                torch.einsum("ad,bd->ab", sent_norm_pos, sent_norm) + 1e-6)  # [batch, batch]
        batch_self22 = batch_self22 / self.temperature[0]
        batch_cross21 = batch_cross21 / self.temperature[0]
        batch_second = torch.cat([batch_self22, batch_cross21], dim=-1)  # [batch, batch * 2]
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        contras_loss = self.cl_loss(batch_predict, batch_label)

        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        # aug2
        input_ids_rnd1 = batch_data["input_ids_rnd1"]
        attention_mask_rnd1 = batch_data["attention_mask_rnd1"]
        token_type_ids_rnd1 = batch_data["token_type_ids_rnd1"]
        bert_inputs_rnd1 = {'input_ids': input_ids_rnd1, 'attention_mask': attention_mask_rnd1,
                            'token_type_ids': token_type_ids_rnd1}
        sent_rep_rnd1 = self.bert(**bert_inputs_rnd1)[1]
        sent_norm_rnd1 = sent_rep_rnd1.norm(dim=-1, keepdim=True)  # [batch]

        input_ids_rnd2 = batch_data["input_ids_rnd2"]
        attention_mask_rnd2 = batch_data["attention_mask_rnd2"]
        token_type_ids_rnd2 = batch_data["token_type_ids_rnd2"]
        bert_inputs_rnd2 = {'input_ids': input_ids_rnd2, 'attention_mask': attention_mask_rnd2,
                            'token_type_ids': token_type_ids_rnd2}
        sent_rep_rnd2 = self.bert(**bert_inputs_rnd2)[1]
        sent_norm_rnd2 = sent_rep_rnd2.norm(dim=-1, keepdim=True)  # [batch]

        batch_self_rnd11 = torch.einsum("ad,bd->ab", sent_rep_rnd1, sent_rep_rnd1) / (
                torch.einsum("ad,bd->ab", sent_norm_rnd1, sent_norm_rnd1) + 1e-6)  # [batch, batch]
        batch_cross_rnd12 = torch.einsum("ad,bd->ab", sent_rep_rnd1, sent_rep_rnd2) / (
                torch.einsum("ad,bd->ab", sent_norm_rnd1, sent_norm_rnd2) + 1e-6)  # [batch, batch]
        batch_self_rnd11 = batch_self_rnd11 / self.temperature[1]
        batch_cross_rnd12 = batch_cross_rnd12 / self.temperature[1]
        batch_first_rnd = torch.cat([batch_self_rnd11, batch_cross_rnd12], dim=-1)  # [batch, batch * 2]
        batch_arange_rnd = torch.arange(batch_size).to(torch.cuda.current_device())
        mask_rnd = F.one_hot(batch_arange_rnd, num_classes=batch_size * 2) * -1e10
        batch_first_rnd += mask_rnd
        batch_label_rnd1 = batch_arange_rnd + batch_size  # [batch]

        batch_self_rnd22 = torch.einsum("ad,bd->ab", sent_rep_rnd2, sent_rep_rnd2) / (
                torch.einsum("ad,bd->ab", sent_norm_rnd2, sent_norm_rnd2) + 1e-6)  # [batch, batch]
        batch_cross_rnd21 = torch.einsum("ad,bd->ab", sent_rep_rnd2, sent_rep_rnd1) / (
                torch.einsum("ad,bd->ab", sent_norm_rnd2, sent_norm_rnd1) + 1e-6)  # [batch, batch]
        batch_self_rnd22 = batch_self_rnd22 / self.temperature[1]
        batch_cross_rnd21 = batch_cross_rnd21 / self.temperature[1]
        batch_second_rnd = torch.cat([batch_self_rnd22, batch_cross_rnd21], dim=-1)  # [batch, batch * 2]
        batch_second_rnd += mask_rnd
        batch_label_rnd2 = batch_arange_rnd + batch_size  # [batch]

        batch_predict_rnd = torch.cat([batch_first_rnd, batch_second_rnd], dim=0)
        batch_label_rnd = torch.cat([batch_label_rnd1, batch_label_rnd2], dim=0)  # [batch * 2]
        contras_loss_rnd = self.cl_loss(batch_predict_rnd, batch_label_rnd)

        batch_logit_rnd = batch_predict_rnd.argmax(dim=-1)
        acc_rnd = torch.sum(batch_logit_rnd == batch_label_rnd).float() / (batch_size * 2)
        return self.loss_para[0] * contras_loss + self.loss_para[1] * contras_loss_rnd, acc


