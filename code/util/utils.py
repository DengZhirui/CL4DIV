import os, csv
import sys
sys.path.append('./')
import random
import numpy as np
import torch
import math
import gzip, pickle
import collections
from sklearn.model_selection import KFold
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def split_list(origin_list, n):
    res_list = []
    L = len(origin_list)
    N = int(math.ceil(L / float(n)))
    begin = 0
    end = begin + N
    while begin < L:
        if end < L:
            temp_list = origin_list[begin:end]
            res_list.append(temp_list)
            begin = end
            end += N
        else:
            temp_list = origin_list[begin:]
            res_list.append(temp_list)
            break
    return res_list


def separate_fold():
    all_qids = np.load('../data/baseline_data/all_qids.npy')
    fold = 1
    fold_d = {}
    for train_ids, test_ids in KFold(5).split(all_qids):
        train_ids.sort()
        test_ids.sort()
        train_qids = [str(all_qids[i]) for i in train_ids]
        test_qids = [str(all_qids[i]) for i in test_ids]
        fold_d[fold] = [train_qids, test_qids]
        fold += 1
    with gzip.open('../data/attn_data/fold_d.json', 'wb') as fdf:
        pickle.dump(fold_d, fdf)


def load_embedding(path):
    emb = {}  # doc_emb[docid] = doc2vec
    with open(path, 'r') as doc_emb_f:
        for line in doc_emb_f:
            line = line[:-1].split('\t')
            emb[line[0]] = np.array([float(val) for val in line[1:]])
    return emb


def load_embedding_list(path):
    emb = {}  # doc_emb[docid] = doc2vec
    with open(path, 'r') as doc_emb_f:
        for line in doc_emb_f:
            line = line[:-1].split('\t')
            emb[line[0]] = [float(val) for val in line[1:]]
    return emb


def read_rel_feat(path):
    rel_feat = {}
    f = csv.reader(open(path, 'r'), delimiter = ',')
    next(f)
    for line in f:
        if line[0] not in rel_feat:
            rel_feat[line[0]] = {}
        if line[1] not in rel_feat[line[0]]:
            rel_feat[line[0]][line[1]] = np.array([float(val) for val in line[2:]])
    return rel_feat

