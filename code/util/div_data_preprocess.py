import sys, os, argparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import gzip, pickle, torch, time
from tqdm import tqdm
import multiprocessing
from bs4 import BeautifulSoup
from util.utils import set_seed, load_embedding, split_list, read_rel_feat, load_embedding_list
from transformers import BertTokenizer, BertModel, BertConfig
from pretrain.BertDoc import BertContrastive
from util.div_type import CLDataset
from torch.utils.data import DataLoader


def gen_bert_embedding(bert_model_path, model_path_cl, max_seq_length, loss_para_cl, tmp_dir, _fold, temperature, state=0):
    gen_doc_embedding(bert_model_path, model_path_cl, max_seq_length, loss_para_cl, tmp_dir, _fold, temperature, state)
    gen_query_embedding(bert_model_path, model_path_cl, max_seq_length, loss_para_cl, tmp_dir, _fold, temperature, state)


def gen_doc_embedding(bert_model_path, model_path_cl, max_length, loss_para, tmp_dir, fold, temperature=0, state=0):
    set_seed()
    doc_emb_dict = {}
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_config = BertConfig.from_pretrained(bert_model_path, output_hidden_states=False)
    if state == 0:
        additional_tokens = 1
        tokenizer.add_tokens("[pas_del]")
        bert = BertContrastive.from_pretrained(bert_model_path).cuda()
        bert.set_parameters(temperature=temperature, max_seq_length=max_length, loss_para=loss_para, additional_tokens=additional_tokens)
        bert.load_state_dict(torch.load(model_path_cl))
    else:
        bert = BertModel.from_pretrained(bert_model_path, config=bert_config).cuda()
    bert.eval()
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    with gzip.open('../data/baseline_data/token_content_50.pkl.gz', 'rb') as f:
        token_dict_all = pickle.load(f)
    doc_emb_f = open(tmp_dir+'doc_bert_cl'+str(fold)+'.emb', 'w')
    with torch.no_grad():
        for doc in tqdm(token_dict_all):
            if not token_dict_all[doc]:
                print(doc)
                continue
            doc_id = tokenizer.convert_tokens_to_ids(token_dict_all[doc])
            doc_encoding = tokenizer.encode_plus(doc_id, max_length=max_length, add_special_tokens=True, truncation=True)
            doc_input_ids, doc_token_type_ids = doc_encoding['input_ids'], doc_encoding['token_type_ids']
            doc_attention_mask = [1] * len(doc_input_ids)
            padding_length = max_length - len(doc_input_ids)
            doc_input_ids += ([pad_token] * padding_length)
            doc_attention_mask += ([0] * padding_length)
            doc_token_type_ids += ([pad_token_segment_id] * padding_length)
            doc_bert_input = {'input_ids': torch.LongTensor([doc_input_ids]).cuda(),
                              'attention_mask': torch.LongTensor([doc_attention_mask]).cuda(),
                              'token_type_ids': torch.LongTensor([doc_token_type_ids]).cuda()}
            if state == 0:
                doc_emb_dict[doc] = bert.bert(**doc_bert_input)[1].squeeze(0).cpu()
            else:
                doc_emb_dict[doc] = bert(**doc_bert_input)[1].squeeze(0).cpu()
            doc_emb_f.write(doc)
            for i in range(doc_emb_dict[doc].shape[0]):
                doc_emb_f.write('\t'+str(float(doc_emb_dict[doc][i])))
            doc_emb_f.write('\n')
    doc_emb_f.close()


def gen_query_embedding(bert_model_path, model_path_cl, max_seq_length, loss_para, tmp_dir, fold, temperature=0, state=0):
    set_seed()
    query_emb = load_embedding('../data/baseline_data/query.emb')
    token_dict = {}
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    max_length = 0
    for query in query_emb.keys():
        sent = tokenizer.tokenize(query)
        token_dict[query] = sent
        max_length = max(max_length, len(sent))
    print('query max length: %d'%(max_length))
    max_length = max_length + 1

    bert_config = BertConfig.from_pretrained(bert_model_path, output_hidden_states=False)
    if state == 0:
        additional_tokens = 1
        tokenizer.add_tokens("[pas_del]")
        bert = BertContrastive.from_pretrained(bert_model_path).cuda()
        bert.set_parameters(temperature=temperature, max_seq_length=max_length, loss_para=loss_para, additional_tokens=additional_tokens)
        bert.load_state_dict(torch.load(model_path_cl))
    else:
        bert = BertModel.from_pretrained(bert_model_path, config=bert_config).cuda()
    bert.eval()
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    doc_emb_f = open(tmp_dir+'query_bert_cl'+str(fold)+'.emb', 'w')
    with torch.no_grad():
        for query in tqdm(token_dict):
            doc_id = tokenizer.convert_tokens_to_ids(token_dict[query])
            doc_encoding = tokenizer.encode_plus(doc_id, max_length=max_length, add_special_tokens=True, truncation=True)
            doc_input_ids, doc_token_type_ids = doc_encoding['input_ids'], doc_encoding['token_type_ids']
            doc_attention_mask = [1] * len(doc_input_ids)
            padding_length = max_length - len(doc_input_ids)
            doc_input_ids += ([pad_token] * padding_length)
            doc_attention_mask += ([0] * padding_length)
            doc_token_type_ids += ([pad_token_segment_id] * padding_length)
            doc_bert_input = {'input_ids': torch.LongTensor([doc_input_ids]).cuda(),
                              'attention_mask': torch.LongTensor([doc_attention_mask]).cuda(),
                              'token_type_ids': torch.LongTensor([doc_token_type_ids]).cuda()}
            if state == 0:
                query_bert_emb = bert.bert(**doc_bert_input)[1].squeeze(0).cpu()
            else:
                query_bert_emb = bert(**doc_bert_input)[1].squeeze(0).cpu()
            doc_emb_f.write(query)
            for i in range(query_bert_emb.shape[0]):
                doc_emb_f.write('\t' + str(float(query_bert_emb[i])))
            doc_emb_f.write('\n')
    doc_emb_f.close()


def divide_five_fold_train_test(tmp_dir, emb_dir, _fold, type=None):
    qd = pickle.load(open('../data/attn_data/div_query.data', 'rb'))
    train_data = pickle.load(open('../data/attn_data/listpair_train.data', 'rb'))
    rel_feat = read_rel_feat('../data/baseline_data/rel_feat.csv')

    data_dir = tmp_dir+'fold/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with gzip.open('../data/attn_data/fold_d.json', 'rb') as f:
        fold_d = pickle.load(f)
    for fold in fold_d:
        if fold != _fold:
            continue
        res_dir = data_dir + 'fold' + str(fold) + '/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        train_qids = fold_d[fold][0]
        test_qids = fold_d[fold][1]
        '''{qid:, query:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:} '''
        if type != None:
            fold = type
        EMB_TYPE = [emb_dir + 'doc_bert_cl' + str(fold) + '.emb', 768]
        doc_emb = load_embedding_list(EMB_TYPE[0])
        query_emb = load_embedding_list(emb_dir + 'query_bert_cl' + str(fold) + '.emb')
        gen_data_file_train(train_qids, qd, train_data, doc_emb, query_emb, rel_feat, res_dir + 'train_data.pkl',
                            EMB_TYPE)
        gen_data_file_test(test_qids, qd, train_data, doc_emb, query_emb, rel_feat, res_dir + 'test_data.pkl', EMB_TYPE)


def gen_data_file_train(train_qids, qd, train_data, doc_emb, query_emb, rel_feat, save_path, EMB_TYPE):
    with gzip.open('../data/attn_data/intent_coverage.pkl.gz', 'rb') as f:
        intent_coverage = pickle.load(f)
    data_list = []  # {qid:, query:, doclist:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:}
    #max_d, max_s, max_ps, max_ns = 0, 0, 0, 0
    for qid in tqdm(train_qids):
        doc2vec = [doc_emb[docid] for docid in qd[qid].best_docs_rank]
        sub2vec = [query_emb[query_sugg] for query_sugg in qd[qid].query_suggestion]

        for i in range(len(train_data[qid])):
            '''
            max_d, max_s, max_ps, max_ns = max(max_d, len(doc2vec)), max(max_s, len(sub2vec)), \
                                           max(max_ps, len(temp['pos_subrel_feat'])), \
                                           max(max_ns, len(temp['neg_subrel_feat']))
            '''
            temp = {}
            temp['qid'] = qid
            temp['query'] = qd[qid].query
            temp['doclist'] = qd[qid].best_docs_rank
            temp['doc2vec_mask'] = torch.tensor([1]*len(doc2vec)+[0]*(50-len(doc2vec)))
            temp['sub2vec_mask'] = torch.tensor([1]*len(sub2vec)+[0]*(10-len(sub2vec)))
            temp['doc2vec'] = torch.tensor(doc2vec+[[0]*EMB_TYPE[1]]*(50-len(doc2vec)))
            temp['sub2vec'] = torch.tensor(sub2vec+[[0]*EMB_TYPE[1]]*(10-len(sub2vec)))
            temp['positive_mask'] = train_data[qid][i][1]
            temp['negative_mask'] = train_data[qid][i][2]
            temp['weight'] = train_data[qid][i][3]
            pos_id = qd[qid].best_docs_rank[int(torch.argmax(train_data[qid][i][1]))]
            neg_id = qd[qid].best_docs_rank[int(torch.argmax(train_data[qid][i][2]))]
            temp['intent_doci'] = torch.nn.functional.one_hot(torch.tensor([int(torch.argmax(train_data[qid][i][1]))]), num_classes=50).squeeze(0)
            temp['intent_docj'] = torch.nn.functional.one_hot(torch.tensor([int(torch.argmax(train_data[qid][i][2]))]), num_classes=50).squeeze(0)
            if tuple((pos_id, neg_id)) in intent_coverage[qid] or tuple((pos_id, neg_id)) in intent_coverage[qid]:
                temp['intent_label'] = 1
            else:
                temp['intent_label'] = 0
            temp['pos_qrel_feat'] = torch.Tensor(rel_feat[qd[qid].query][pos_id])
            temp['neg_qrel_feat'] = torch.Tensor(rel_feat[qd[qid].query][neg_id])
            temp['subrel_feat_mask'] = torch.tensor([1]*len(qd[qid].query_suggestion)+[0]*(10-len(qd[qid].query_suggestion)))
            temp['pos_subrel_feat'] = torch.tensor([rel_feat[query_sugg][pos_id] for query_sugg in
                                                qd[qid].query_suggestion]+[[0]*18]*(10-len(qd[qid].query_suggestion)))
            temp['neg_subrel_feat'] = torch.tensor([rel_feat[query_sugg][neg_id] for query_sugg in
                                       qd[qid].query_suggestion]+[[0]*18]*(10-len(qd[qid].query_suggestion)))
            data_list.append(temp)
    torch.save(data_list, save_path)


def gen_data_file_test(test_qids, qd, test_data, doc_emb, query_emb, rel_feat, save_path, EMB_TYPE):
    data_list = {}  # {qid:, query:, doclist:, doc2vec:, sub2vec:, rel_feat:, sub_rel_feat:, list_pair:}
    for qid in tqdm(test_qids):
        data_list[qid] = {}
        doc2vec = [doc_emb[docid] for docid in qd[qid].best_docs_rank]
        sub2vec = [query_emb[query_sugg] for query_sugg in qd[qid].query_suggestion]
        data_list[qid]['qid'] = qid
        data_list[qid]['query'] = qd[qid].query
        data_list[qid]['doclist'] = qd[qid].best_docs_rank
        data_list[qid]['doc2vec_mask'] = torch.tensor([1] * len(doc2vec) + [0] * (50 - len(doc2vec)))
        data_list[qid]['sub2vec_mask'] = torch.tensor([1] * len(sub2vec) + [0] * (10 - len(sub2vec)))
        data_list[qid]['doc2vec'] = torch.tensor(doc2vec + [[0] * EMB_TYPE[1]] * (50 - len(doc2vec)))
        data_list[qid]['sub2vec'] = torch.tensor(sub2vec + [[0] * EMB_TYPE[1]] * (10 - len(sub2vec)))
        data_list[qid]['pos_qrel_feat'] = torch.Tensor([rel_feat[qd[qid].query][pos_id]
                                    for pos_id in qd[qid].best_docs_rank]+[[0]*18]*(50-len(qd[qid].best_docs_rank)))
        pos_subrel_feat = [] # 50*10*18
        subrel_mask = []
        for pos_id in qd[qid].best_docs_rank:
            temp1 = []
            for query_sugg in qd[qid].query_suggestion:
                temp1.append(rel_feat[query_sugg][pos_id])
            temp2 = [0]*len(temp1)+[1]*(10-len(temp1))
            temp1.extend([[0]*18]*(10-len(temp1)))
            pos_subrel_feat.append(temp1)
            subrel_mask.append(temp2)
        subrel_mask.extend([[0] * 10] * (50 - len(pos_subrel_feat)))
        pos_subrel_feat.extend([[[0]*18]*10]*(50-len(pos_subrel_feat)))
        data_list[qid]['subrel_feat_mask'] = torch.tensor(subrel_mask)
        data_list[qid]['pos_subrel_feat'] = torch.tensor(pos_subrel_feat)
    torch.save(data_list, save_path)


def get_fold_loader(fold, train_data, BATCH_SIZE):
    set_seed()
    input_list = []
    starttime = time.time()
    print('Begin loading fold {} training data'.format(fold))
    for i in range(len(train_data)):
        input_list.append([train_data[i]['doc2vec'],
                           train_data[i]['sub2vec'],
                           train_data[i]['doc2vec_mask'],
                           train_data[i]['sub2vec_mask'],
                           train_data[i]['weight'],
                           torch.argmax(train_data[i]['positive_mask']),
                           torch.argmax(train_data[i]['negative_mask']),
                           train_data[i]['pos_qrel_feat'],
                           train_data[i]['neg_qrel_feat'],
                           train_data[i]['pos_subrel_feat'],
                           train_data[i]['neg_subrel_feat'],
                           train_data[i]['subrel_feat_mask'],
                           train_data[i]['intent_doci'],
                           train_data[i]['intent_docj'],
                           train_data[i]['intent_label']])
    desa_dataset = DESADataset(input_list)
    loader = DataLoader(
        dataset=desa_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print('Training data loaded!')
    endtime = time.time()
    print('Total time  = ', round(endtime - starttime, 2), 'secs')
    return loader

