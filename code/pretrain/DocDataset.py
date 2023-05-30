import linecache
from torch.utils.data import Dataset
import numpy as np
import random
import gzip
import pickle
from util.utils import set_seed

set_seed()


class ContrasDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, batch_size,
                 aug_strategy=["pas_deletion", "pas_reorder", "dropout"]):
        super(ContrasDataset, self).__init__()
        set_seed()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self._aug_strategy = aug_strategy
        self._rnd = random.Random(0)
        with gzip.open('../data/baseline_data/token_content_50.pkl.gz', 'rb') as f:
            self.doc_token_dict = pickle.load(f)
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def anno_main(self, qd_pairs, type=1):
        set_seed()
        if type == 1:
            query_tok = self._tokenizer.tokenize(qd_pairs[0]) + ["[SEP]"]
            doc_tok = self.doc_token_dict[qd_pairs[1]]
            all_qd_toks = ["[CLS]"] + query_tok + doc_tok
            all_qd_toks = all_qd_toks[:self._max_seq_length]
        elif type == 2:
            all_qd_toks = qd_pairs
        segment_ids = [0] * len(all_qd_toks)
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids

    def augmentation(self, sequence, strategy):
        set_seed()
        query_tok = self._tokenizer.tokenize(sequence[0]) + ["[SEP]"]
        doc_tok = self.doc_token_dict[sequence[1]]
        aug_sequence = ["[CLS]"] + query_tok + doc_tok
        aug_sequence = aug_sequence[:self._max_seq_length]
        pas_len = (len(aug_sequence) - len(query_tok) - 1) // 8  # a document is split into 8 passages
        random_positions = -1
        if strategy == "pas_deletion":
            random_positions = self._rnd.sample(list(range(len(query_tok) + 1, len(aug_sequence) - pas_len)), 1)[0]
            for mask_position in range(random_positions, random_positions + pas_len, 1):
                aug_sequence[mask_position] = "[pas_del]"
        elif strategy == "pas_reorder":
            change_pos = self._rnd.sample(list(range(8)), 2)
            random_positions = change_pos
            begin = len(query_tok) + 1
            tmp = aug_sequence[begin + change_pos[0] * pas_len:begin + (change_pos[0] + 1) * pas_len]
            aug_sequence[begin + change_pos[0] * pas_len:begin + (change_pos[0] + 1) * pas_len] = \
                aug_sequence[begin + change_pos[1] * pas_len: begin + (change_pos[1] + 1) * pas_len]
            aug_sequence[begin + change_pos[1] * pas_len: begin + (change_pos[1] + 1) * pas_len] = tmp
        elif strategy == "dropout":
            pass
        else:
            assert False
        return aug_sequence, random_positions

    def __getitem__(self, idx):
        set_seed()
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")

        # aug1
        qd_anchor = [line[1], line[2]]
        qd_positive = [line[1], line[3]]
        input_ids, attention_mask, segment_ids = self.anno_main(qd_anchor)
        input_ids_pos, attention_mask_pos, segment_ids_pos = self.anno_main(qd_positive)

        # aug2
        random_qd_pairs1 = qd_anchor.copy()
        random_qd_pairs2 = qd_anchor.copy()
        aug_strategy = self._aug_strategy
        strategy1 = self._rnd.choice(aug_strategy)
        random_qd_pairs1, random_pos1 = self.augmentation(random_qd_pairs1, strategy1)

        # build positive pair with strategy1 == strategy2
        random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy1)
        while random_pos1 == random_pos2 and strategy1 != "dropout":
            random_qd_pairs2 = qd_anchor.copy()
            random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy1)

        input_ids_rnd1, attention_mask_rnd1, segment_ids_rnd1 = self.anno_main(random_qd_pairs1, 2)
        input_ids_rnd2, attention_mask_rnd2, segment_ids_rnd2 = self.anno_main(random_qd_pairs2, 2)

        batch = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids,
            'attention_mask': attention_mask,
            'input_ids_pos': input_ids_pos,
            'token_type_ids_pos': segment_ids_pos,
            'attention_mask_pos': attention_mask_pos,
            'input_ids_rnd1': input_ids_rnd1,
            'token_type_ids_rnd1': segment_ids_rnd1,
            'attention_mask_rnd1': attention_mask_rnd1,
            'input_ids_rnd2': input_ids_rnd2,
            'token_type_ids_rnd2': segment_ids_rnd2,
            'attention_mask_rnd2': attention_mask_rnd2,
        }
        return batch

    def __len__(self):
        return self._total_data

