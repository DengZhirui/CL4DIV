import numpy as np
import pandas as pd
import torch

MAXDOC = 50
REL_LEN = 18


def adjust_graph(A, rel_score_list, degree_tensor, selected_doc_id):
    '''
    adjust adjancent matrix A during the testing process, set the selected doc degree = 0
    :param rel_score_list: initial relevance of the document
    :param degree_tensor: degree tensor of each document
    :return: adjacent matrix A, degree tensor
    '''
    ''' connect selected document to the query node '''
    A[0, selected_doc_id + 1, 0] = rel_score_list[selected_doc_id]
    A[0, 0, selected_doc_id + 1] = rel_score_list[selected_doc_id]
    ''' remove edges between selected document and candidates '''
    A[0, selected_doc_id + 1, 1:] = torch.tensor([0.0] * 50).float()
    A[0, 1:, selected_doc_id + 1] = torch.tensor([0.0] * 50).float()
    ''' set the degree of selected document '''
    degree_tensor[0, selected_doc_id] = torch.tensor(0.0)
    return A, degree_tensor


def get_metrics_20(csv_file_path):
    all_qids=range(1,201)
    del_index=[94,99]
    all_qids=np.delete(all_qids,del_index)
    qids=[str(i) for i in all_qids]

    df=pd.read_csv(csv_file_path)

    alpha_nDCG_20=df.loc[df['topic'].isin(qids)]['alpha-nDCG@20'].mean()
    NRBP_20=df.loc[df['topic'].isin(qids)]['NRBP'].mean()
    ERR_IA_20=df.loc[df['topic'].isin(qids)]['ERR-IA@20'].mean()
    # Pre_IA_20=df.loc[df['topic'].isin(qids)]['P-IA@20'].mean()
    S_rec_20=df.loc[df['topic'].isin(qids)]['strec@20'].mean()
    return alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20


def get_metric_nDCG_random(model, test_tuple, div_q, qid):
    '''
    get the alpha-nDCG for the input query, the input document list are randomly shuffled.
    :param test_tuple: the features of the test query qid, test_turple = {}
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the alpha-nDCG for the test query
    '''
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if not test_tuple:
        return 0
    else:
        doc_mask = test_tuple['doc2vec_mask'].unsqueeze(0) # [1,50]
        sub_mask = test_tuple['sub2vec_mask'].unsqueeze(0) # [1,10]
        doc_emb = test_tuple['doc2vec'].unsqueeze(0).float() # [1, 50, 100]
        sub_emb = test_tuple['sub2vec'].unsqueeze(0).float() # [1,10,100]
        pos_qrel_feat = test_tuple['pos_qrel_feat'].unsqueeze(0).float() # [1,50,18]
        subrel_feat_mask = test_tuple['subrel_feat_mask'].unsqueeze(0)
        pos_subrel_feat = test_tuple['pos_subrel_feat'].unsqueeze(0).float() # [1,50,10,18]

        doc_mask.requires_grad = False
        sub_mask.requires_grad = False
        doc_emb.requires_grad = False
        sub_emb.requires_grad = False
        pos_qrel_feat.requires_grad = False
        subrel_feat_mask.requires_grad = False
        pos_subrel_feat.requires_grad = False

        if torch.cuda.is_available():
            doc_mask, sub_mask, doc_emb, sub_emb, pos_qrel_feat, subrel_feat_mask, pos_subrel_feat =\
                doc_mask.cuda(), sub_mask.cuda(), doc_emb.cuda(), sub_emb.cuda(), pos_qrel_feat.cuda(), \
                subrel_feat_mask.cuda(), pos_subrel_feat.cuda()
        #print(doc_emb.shape, sub_emb.shape, doc_mask.shape, sub_mask.shape, pos_qrel_feat.shape, pos_subrel_feat.shape)
        score = model(doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat, mode='Test')
        result = list(np.argsort(score[:len(test_tuple['doclist'])].cpu().detach().numpy()))
        if len(result) > 0:
            new_docs_rank = []
            eval_res = []
            for i in range(len(result)-1, -1, -1):
                new_docs_rank.append(test_tuple['doclist'][result[i]])
                eval_res.append((test_tuple['doclist'][result[i]], float(score[result[i]])))
            #new_docs_rank = [test_tuple['doclist'][result[i]] for i in range(len(result)-1, len(result)-len(test_tuple['doclist'])-1, -1)]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
    return metric, eval_res


def evaluate_accuracy(y_pred, y_label):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    for i in range(num):
        pred = (y_pred[i] > 0.5).astype(int)
        label = y_label[i]
        acc = 1 if pred == label else 0
        all_acc += acc
        count += 1
    return all_acc / count
