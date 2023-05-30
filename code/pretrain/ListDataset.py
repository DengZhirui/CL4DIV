import sys, os, argparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import torch
import linecache
from torch.utils.data import Dataset
import random
from util.utils import load_embedding_list, load_embedding, set_seed
set_seed()

class ContrasDatasetList(Dataset):
    def __init__(self, filename, batch_size, tmp_dir, aug_strategy=["sent_deletion", "term_deletion", "qd_reorder"]):
        super(ContrasDatasetList, self).__init__()
        set_seed()
        self._filename = filename
        self.batch_size = batch_size
        self._aug_strategy = aug_strategy
        self._rnd = random.Random(0)
        self.doc_emb = load_embedding_list(tmp_dir+'doc_bert_cll.emb')
        self.query_emb = load_embedding(tmp_dir+'query_bert_cll.emb')
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def anno_main(self, qd_pairs, type=1):
        doc_reps = []
        for doc_id in qd_pairs:
            doc_reps.append(self.doc_emb[doc_id])
        doc_mask = torch.tensor([1] * len(doc_reps) + [0] * (50 - len(doc_reps)))
        doc_reps = torch.tensor(doc_reps + [[0] * 768] * (50 - len(doc_reps)))
        return doc_reps, doc_mask

    def augmentation(self, sequence, strategy):
        set_seed()
        query_tok = self._tokenizer.tokenize(sequence[0]) + ["[SEP]"]
        doc_tok = self.doc_token_dict[sequence[1]]
        aug_sequence = ["[CLS]"] + query_tok + doc_tok
        aug_sequence = aug_sequence[:self._max_seq_length]
        pas_len = (len(aug_sequence) - len(query_tok) - 1) // 8  # a document is split into 8 passages
        random_positions = -1
        if strategy == "pas_deletion":
            random_positions = self._rnd.sample(list(range(len(query_tok), len(aug_sequence) - pas_len)), 1)[0]
            for mask_position in range(random_positions, random_positions + pas_len, 1):
                aug_sequence[mask_position] = "[pas_del]"
        elif strategy == "pas_reorder":
            change_pos = self._rnd.sample(list(range(8)), 2)
            begin = len(query_tok) + 1
            tmp = aug_sequence[begin + change_pos[0] * pas_len:begin + (change_pos[0] + 1) * pas_len]
            aug_sequence[begin + change_pos[0] * pas_len:begin + (change_pos[0] + 1) * pas_len] = \
                aug_sequence[begin + change_pos[1] * pas_len: begin + (change_pos[1] + 1) * pas_len]
            aug_sequence[begin + change_pos[1] * pas_len: begin + (change_pos[1] + 1) * pas_len] = tmp
        else:
            assert False
        return aug_sequence, random_positions

    def __getitem__(self, idx):
        set_seed()
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")

        # aug1
        query = line[0]
        length = int(line[1])
        qd_anchor = line[2:2+length]
        qd_positive = line[2+length:2+2*length]

        doc_reps, doc_mask = self.anno_main(qd_anchor)
        doc_reps_pos, doc_mask_pos = self.anno_main(qd_positive)

        batch = {
            'input_ids': doc_reps,
            'attention_mask': doc_mask,
            'input_ids_pos': doc_reps_pos,
            'attention_mask_pos': doc_mask_pos,
        }
        return batch

    def __len__(self):
        return self._total_data

