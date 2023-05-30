import sys, os, argparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import os, gzip, pickle, multiprocessing
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel, BertConfig
from util.utils import set_seed, split_list, separate_fold


def extract_doc_content(save_dir, split_times):
    set_seed()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    extract_doc_content_sub1(save_dir, split_times)


def extract_doc_content_sub1(save_dir, split_times):
    set_seed()
    clu_data_dir = '../data/clueweb_original_fast/'
    clu_datas = os.listdir(clu_data_dir)
    jobs = []
    task_list = split_list(clu_datas, split_times)
    for task in task_list:
        p = multiprocessing.Process(target=doc_content_token_worker,
                                    args=(task, clu_data_dir, save_dir, task_list.index(task)))
        jobs.append(p)
        p.start()


def doc_content_token_worker(clu_datas, clu_data_dir, save_dir, task_id):
    set_seed()
    token_dict = {}
    for clu_data in tqdm(clu_datas):
        with open(clu_data_dir+clu_data, 'r', encoding="ISO-8859-1") as f:
            f = f.read().lower()
            f = f[f.find('http/'):]
            f = f[f.find('content-length') + 21:]
            if clu_data == 'clueweb09-en0003-73-16166.txt' or clu_data == 'clueweb09-en0025-58-05321.txt':
                f = f[f.find(('<html')):]
            content = BeautifulSoup(f).get_text()
            token_dict[clu_data[:-4]] = content
    with gzip.open(save_dir+str(task_id)+'.pkl.gz', 'wb') as f:
        pickle.dump(token_dict, f)


def extract_doc_content_sub2(save_dir, split_times):
    set_seed()
    token_dict_all = {}
    for task_id in range(split_times):
        with gzip.open(save_dir+str(task_id) + '.pkl.gz', 'rb') as f:
            docs = pickle.load(f)
            token_dict_all = dict(token_dict_all, **docs)
    with gzip.open(save_dir+'doc_content_all.pkl.gz', 'wb') as f:
        pickle.dump(token_dict_all, f)


def gen_doc_topic_T5(save_dir, k=10):
    set_seed()

    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    if torch.cuda.is_available():
        model = model.cuda()

    save_file = open(save_dir + 'gen_query.txt', 'a+')
    docs_file = open(save_dir+'doc_id.txt', 'a+')


    gen_query = {}
    with gzip.open(save_dir + 'doc_content_all.pkl.gz', 'rb') as f:
        docs = pickle.load(f)
        print(len(list(docs.keys())))
        doc_ids = list(docs.keys())
        for doc_id in tqdm(doc_ids):
            if doc_id == 'clueweb09-en0025-86-27304':
                continue
            docs_file.write(str(doc_id)+'\t')
            save_file.write(str(doc_id))
            gen_query[doc_id] = []
            doc_text = docs[doc_id]
            input_ids = tokenizer.encode(doc_text, return_tensors='pt', truncation=True).cuda()
            outputs = model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                # num_beams=10,
                # top_k=10,
                num_return_sequences=k,
                # num_beam_groups=2
            )
            for i in range(k):
                _str = tokenizer.decode(outputs[i].cpu(), skip_special_tokens=True)
                save_file.write('\t'+_str)
            save_file.write('\n')
    save_file.close()
    docs_file.close()


def gen_doc_embedding(save_dir, bert_model_path):
    '''concatenate subtopics by T5 to generate embedding for documents'''
    set_seed()
    doc_list = []

    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_config = BertConfig.from_pretrained(bert_model_path, output_hidden_states=False)
    bert = BertModel.from_pretrained(bert_model_path, config=bert_config).cuda()
    bert.eval()
    data_file = open(save_dir + 'gen_query.txt', 'r')
    doc_query_d = {}
    lines = data_file.readlines()
    flag = 0
    with torch.no_grad():
        for line in tqdm(lines):
            line = line[:-1].split('\t')
            doc_query_d[line[0]] = line[1:]
            doc_list.append(line[0])

            token = tokenizer.tokenize(' [SEP] '.join(line[1:]))
            doc_id = tokenizer.convert_tokens_to_ids(token)
            doc_encoding = tokenizer.encode_plus(doc_id, add_special_tokens=True, truncation=True, return_tensors='pt') # , max_length=512, padding='max_length'
            doc_encoding['input_ids'], doc_encoding['token_type_ids'], doc_encoding['attention_mask'] = \
                doc_encoding['input_ids'].cuda(), doc_encoding['token_type_ids'].cuda(), doc_encoding['attention_mask'].cuda()
            bert_out = bert(**doc_encoding)[1].cpu()
            if flag == 0:
                output = bert_out
                flag = 1
            else:
                output = torch.cat([output, bert_out], dim=0)

    norm = output.norm(dim=1, keepdim=True)
    similarity = torch.einsum("ab,bc->ac", output, output.transpose(0,1)) / (
            torch.einsum("ab,bc->ac", norm, norm.transpose(0,1)) + 1e-20)

    data_file.close()
    torch.save(doc_list, save_dir+'doc_id_list.pkl')
    torch.save(output, save_dir+'data_embedding.pkl')
    torch.save(similarity, save_dir+'doc_cos_similarity.pkl')


def gen_cl_pair(save_dir, topk):
    set_seed()
    
    if not os.path.exists(save_dir+'T5_cl/'):
        os.makedirs(save_dir+'T5_cl/')
    
    doc_id_list = torch.load(save_dir+'doc_id_list.pkl')
    similarity = torch.load(save_dir+'doc_cos_similarity.pkl')
    qd = pickle.load(open('../data/attn_data/div_query.data', 'rb'))

    separate_fold()

    with gzip.open('../data/attn_data/fold_d.json', 'rb') as f:
        fold_d = pickle.load(f)
    for fold in tqdm(fold_d):
        train_qids = fold_d[fold][0]
        test_qids = fold_d[fold][1]
        train_doc = open(save_dir+'T5_cl/train'+str(fold)+'.txt', 'w')
        test_doc = open(save_dir + 'T5_cl/test' + str(fold) + '.txt', 'w')
        for qid in qd:
            doc_list = qd[qid].best_docs_rank
            doc_index = [doc_id_list.index(val) for val in doc_list]
            for doc_i in doc_index:
                doc_matrix = similarity[doc_i]
                select_similarity = doc_matrix.index_select(0, torch.LongTensor(doc_index))
                value, indices = torch.topk(select_similarity, min(topk+1, len(doc_index)))
                for j in range(1, len(indices)):
                    if value[j] > 0.99:
                        anchor, positive = doc_list[indices[0]], doc_list[indices[j]]
                        if qid in train_qids:
                            train_doc.write('1\t'+qd[qid].query+'\t'+anchor+'\t'+positive+'\n')
                        else:
                            test_doc.write('1\t' + qd[qid].query + '\t' + anchor + '\t' + positive + '\n')

        train_doc.close()
        test_doc.close()


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default="extract_doc_content", help="run mode")
    parser.add_argument("--bert_model_path", default="../bert-base-uncased/", type=str, help="")
    args = parser.parse_args()
    
    '''generate contrastive learning data for SUB and DOC tasks'''
    split_times = 10
    save_dir = '../data/data_content/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.mode == 'extract_doc_content':
        '''extract document content from Clueweb dataset'''
        extract_doc_content(save_dir=save_dir, split_times=split_times)
        extract_doc_content_sub2(save_dir, split_times)
    
    elif args.mode == 'gen_data':
        '''generate subtopics for each document with T5'''
        gen_doc_topic_T5(save_dir, k=20)
        
        '''generate document embedding according to their subtopics'''
        gen_doc_embedding(save_dir, args.bert_model_path)
        
        '''generate training pairs for SUB and DOC contrastive learning tasks'''
        gen_cl_pair(save_dir, 10)
    
    
