# -*- coding:utf-8 -*-
from torch.nn.utils import clip_grad_norm_
import copy
from util.model import *
import util.div_data_preprocess as DP
from util.T5_data_preprocess import *
import util.evaluate as EV
from util.div_type import *
from util.utils import set_seed
set_seed()

import sys
MAXDOC=50
sys.setrecursionlimit(10000)

#LR_list=[0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
LR_list=[0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]


def list_pairwise_loss(score_1, score_2, delta):
    set_seed()
    acc = torch.Tensor.sum((score_1 -score_2 ) >0).item( ) /float(score_1.shape[0])
    loss = -torch.sum(delta * torch.Tensor.log(1e-8 +torch.sigmoid(score_1 - score_2)) ) /float(score_1.shape[0])
    return acc, loss


def decay_LR(lr):
    # return lr
    index = LR_list.index(lr)
    if index == -1 or lr <= 1e-6:
        return LR_list[0] 
    else:
        return LR_list[index+1]


def run(PERIOD, BATCH_SIZE, EPOCH, LR, DROPOUT, EMB_LEN, model_path, logger, tmp_dir, fold, temperature,
        old_best_model_list, max_metric_list, model_path_cll=None, rs_folder=''):
    set_seed()
    ''' load randomly shuffled queries '''
    logger.info('old_best_model_list={}'.format(old_best_model_list))
    qd = pickle.load(open('../data/attn_data/div_query.data', 'rb'))
    fold_p = tmp_dir +'fold/'
    final_metrics = []
    best_model_list = []
    eval_best_all_fold = {}
    for _fold in os.listdir(fold_p):
        fold_time = int(_fold[-1])
        if fold_time != fold:
            continue
        
        best_model = old_best_model_list[0]#[fold_time-1]
        decay_flag = False if best_model == "" else True
        max_metric = max_metric_list#[fold_time-1]
        
        train_data = torch.load(fold_p +_fold +'/train_data.pkl')
        test_data = torch.load(fold_p + _fold + '/test_data.pkl')
        desa_data_loader = DP.get_fold_loader(fold_time, train_data, BATCH_SIZE)

        logger.info('Fold = {}'.format(fold_time))
        model = DESA(EMB_LEN, 8, 2, EMB_LEN, 8, 2, 8, DROPOUT, temperature, model_path_cll)
        if torch.cuda.is_available():
            model = model.cuda()
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
        params = list(model.parameters())
        if fold_time == 1:
            logger.info('Parameters Length = {}'.format(len(params)))
            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            logger.info('* number of parameters: %d' % n_params)

        all_steps = len(desa_data_loader)
        patience = 0
        weights = [0.4,0.4,0.4,0.39,0.39]
        for epoch in range(EPOCH):
            logger.info('Start Training...')
            model.train()
            epoch_iterator = tqdm(desa_data_loader, desc='BATCH', ncols=80)
            for step, train_data in enumerate(epoch_iterator):
                tr_doc_emb, tr_sub_emb, tr_doc_mask, tr_sub_mask, tr_weight, tr_index_i, tr_index_j, \
                tr_pos_qrel_feat, tr_neg_qrel_feat, \
                tr_pos_subrel_feat, tr_neg_subrel_feat, tr_subrel_mask = train_data
                if torch.cuda.is_available():
                    doc_emb = tr_doc_emb.cuda()
                    sub_emb = tr_sub_emb.cuda()
                    doc_mask = tr_doc_mask.cuda()
                    sub_mask = tr_sub_mask.cuda()
                    weight = tr_weight.cuda()
                    index_i = tr_index_i.cuda()
                    index_j = tr_index_j.cuda()
                    pos_qrel_feat = tr_neg_qrel_feat.cuda()
                    neg_qrel_feat = tr_neg_qrel_feat.cuda()
                    pos_subrel_feat = tr_pos_subrel_feat.cuda()
                    neg_subrel_feat = tr_neg_subrel_feat.cuda()
                    subrel_mask = tr_subrel_mask.cuda()
                score_1, score_2 = model(doc_emb, sub_emb, doc_mask, sub_mask, pos_qrel_feat, pos_subrel_feat,
                                         index_i, index_j, neg_qrel_feat, neg_subrel_feat, subrel_mask)
                acc, loss = list_pairwise_loss(score_1, score_2, weight)
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                opt.step()
                epoch_iterator.set_postfix(lr=LR, loss=loss)
                if (step + 1) % (all_steps // 50) == 0:
                    model.eval()
                    metrics = []
                    eval_ress = {}
                    for qid in test_data:
                        metric, eval_res = EV.get_metric_nDCG_random(model, test_data[str(qid)], qd[str(qid)], str(qid))
                        metrics.append(metric)
                        eval_ress[qid] = eval_res
                    avg_alpha_NDCG = np.mean(metrics)
                    if max_metric < avg_alpha_NDCG:
                        if best_model != "" and os.path.exists(best_model):
                            os.remove(best_model)
                        max_metric = avg_alpha_NDCG
                        logger.info('max avg_alpha_NDCG updated: {}'.format(max_metric))
                        model_filename = model_path + '_alpha_NDCG_' + str(max_metric) + '.pickle'
                        torch.save(model.state_dict(), model_filename)
                        logger.info('save file at: {}'.format(model_filename))
                        #if best_model != "" and best_model[6:] in os.listdir('../model/'):
                        #    command = 'rm ' + best_model
                        #    os.system(command)
                        best_model = model_filename
                        eval_best = copy.deepcopy(eval_ress)
                        patience = 0

                    else:
                        patience += 1
                    model.train()
                    if (epoch > 0 or decay_flag) and patience > 2:
                        new_lr = 0.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = decay_LR(param_group['lr'])
                            new_lr = param_group['lr']
                        patience = 0
                        if new_lr < 1e-10:
                            break
                        logger.info("decay lr: {}, load model: {}".format(new_lr, best_model))
                        model.load_state_dict({k.replace('module.', ''):v for k, v in torch.load(best_model).items()}, strict=False)
            model.eval()
            metrics = []
            eval_ress = {}
            for qid in test_data:
                metric, eval_res = EV.get_metric_nDCG_random(model, test_data[str(qid)], qd[str(qid)], str(qid))
                metrics.append(metric)
                eval_ress[qid] = eval_res
            avg_alpha_NDCG = np.mean(metrics)
            if max_metric < avg_alpha_NDCG:
                if best_model != "" and os.path.exists(best_model):
                    os.remove(best_model)
                max_metric = avg_alpha_NDCG
                logger.info('max avg_alpha_NDCG updated: {}'.format(max_metric))
                model_filename = model_path + '_alpha_NDCG_' + str(max_metric) + '.pickle'
                torch.save(model.state_dict(), model_filename)
                logger.info('save file at: {}'.format(model_filename))
                best_model = model_filename
                eval_best = copy.deepcopy(eval_ress)
            if epoch == (EPOCH - 1):
                final_metrics.append(max_metric)
                best_model_list.append(best_model)
                mt=np.sum([weights[i]*final_metrics[i] for i in range(len(final_metrics))])/np.sum([weights[j] for j in range(len(final_metrics))])
                logger.info('metric current={}'.format(mt))

    logger.info('final list = {}'.format(final_metrics))
    logger.info('Begin Final Evaluate....')

    if final_metrics[0] <= max_metric_list:
        return best_model_list, final_metrics[0]

    eval_best_all_fold = dict(eval_best_all_fold, **eval_best)
    if not os.path.exists(tmp_dir +rs_folder):
        os.makedirs(tmp_dir +rs_folder)
    with open(tmp_dir +rs_folder +'output_best' +str(fold) +'.txt', 'w') as judge_f:
        qids = list(eval_best_all_fold.keys())
        qids.sort(key=int)
        for qid in qids:
            for i in range(len(eval_best_all_fold[qid])):
                judge_f.write(str(qid ) +' Q0  ' +eval_best_all_fold[qid][i][0 ] +'  ' +str( i +1 ) +'  ' +str
                    (eval_best_all_fold[qid][i][1] ) +' indri\n')
    return best_model_list, final_metrics[0]


def run_test(BATCH_SIZE, EPOCH, LR, DROPOUT, EMB_LEN, model_path, logger, tmp_dir, fold, temperature,
             best_model_pre, best_andcg_pre, model_path_cll=None, rs_folder=''):
    set_seed()
    ''' load randomly shuffled queries '''
    qd = pickle.load(open('../data/attn_data/div_query.data', 'rb'))
    fold_p = tmp_dir +'fold/'
    eval_best_all_fold = {}
    for _fold in os.listdir(fold_p):
        fold_time = int(_fold[-1])
        if fold_time != fold:
            continue
        test_data = torch.load(fold_p + _fold + '/test_data.pkl')
        # fold_time += 1
        logger.info('Fold = {}'.format(fold_time))
        model = DESA(EMB_LEN, 8, 2, EMB_LEN, 8, 2, 8, DROPOUT, temperature, model_path_cll)
        if best_model_pre != "":
            model.load_state_dict(torch.load(best_model_pre))
        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()
        eval_ress = {}
        for qid in test_data:
            metric, eval_res = EV.get_metric_nDCG_random(model, test_data[str(qid)], qd[str(qid)], str(qid))
            eval_ress[qid] = eval_res

        eval_best = copy.deepcopy(eval_ress)
        eval_best_all_fold = dict(eval_best_all_fold, **eval_best)

    if not os.path.exists(tmp_dir +rs_folder):
        os.makedirs(tmp_dir +rs_folder)
    with open(tmp_dir +rs_folder +'output_best ' +str(fold) +'.txt', 'w') as judge_f:
        qids = list(eval_best_all_fold.keys())
        qids.sort(key=int)
        for qid in qids:
            for i in range(len(eval_best_all_fold[qid])):
                judge_f.write(str(qid ) +' Q0  ' +eval_best_all_fold[qid][i][0 ] +'  ' +str( i +1 ) +'  ' +str
                    (eval_best_all_fold[qid][i][1] ) +' indri\n')
    return









