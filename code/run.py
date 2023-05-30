# -*- coding:utf-8 -*-
import os
import logging
import argparse
import util.div_data_preprocess as DIVDP
#from util.div_run import run
from util.div_type import div_dataset
import pretrain.runT5cl as CL
import pretrain.runListcl as CLL
from util.utils import set_seed
set_seed()


MAXDOC = 50
REL_LEN = 18


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--mode', type=str, default="cl_T5", help="run mode")
    parser.add_argument('--comment', type=str, default="ORIGIN", help="run comment")
    parser.add_argument("--bert_model_path", default="../bert-base-uncased/", type=str, help="")
    parser.add_argument("--log_path", default="../log/", type=str, help="The path to save log.")
    parser.add_argument("--bert_emb_len", type=int, default=256, help="")
    parser.add_argument('--fold', type=int, default=1, help="the training fold for diversification")
    
    # setting for SUB and DOC tasks
    parser.add_argument('--per_gpu_batch_size_cl', type=int, default=8, help="the batch size")
    parser.add_argument('--per_gpu_test_batch_size_cl', type=int, default=8, help="the batch size")
    parser.add_argument('--lr_cl', type=float, default=7e-5, help="")
    parser.add_argument('--temperature0', type=float, default=0.4, help="")
    parser.add_argument('--temperature1', type=float, default=0.1, help="")
    parser.add_argument('--loss_para0', type=float, default=0.3, help="")
    parser.add_argument('--loss_para1', type=float, default=0.7, help="")
    parser.add_argument('--epochs_cl', type=int, default=3, help="the training epoches")
    parser.add_argument("--model_path_cl", default="../model/doc/", type=str, help="The path to save model.")
    parser.add_argument("--aug_strategy", default="pas_deletion,pas_reorder,dropout", type=str, help="")

    # setting for SEQ task
    parser.add_argument('--per_gpu_batch_size_cll', type=int, default=1048, help="the batch size")
    parser.add_argument('--per_gpu_test_batch_size_cll', type=int, default=4096, help="the batch size")
    parser.add_argument('--lr_cll', type=float, default=1e-3, help="")
    parser.add_argument('--temperature_cll', type=float, default=0.8, help="")
    parser.add_argument('--epochs_cll', type=int, default=1, help="the training epoches")
    parser.add_argument("--model_path_cll", default="../model/seq/", type=str, help="The path to save model.")

    # setting for diversified ranking
    parser.add_argument('--per_gpu_batch_size', type=int, default=4, help="the batch size")
    parser.add_argument('--per_gpu_test_batch_size', type=int, default=256, help="the batch size")
    parser.add_argument('--lr', type=float, default=0.03, help="")
    parser.add_argument('--epochs', type=int, default=3, help="the training epoches")
    parser.add_argument('--dropout', type=float, default=0, help="the dropout rate")
    parser.add_argument("--model_path", default="../model/div/", type=str, help="The path to save model.")
    parser.add_argument("--div_model", default="CL4DIV", type=str, help="")

    args = parser.parse_args()

    if not os.path.exists(args.model_path_cl):
        os.makedirs(args.model_path_cl)

    if not os.path.exists(args.model_path_cll):
        os.makedirs(args.model_path_cll)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    bs_cl = args.per_gpu_batch_size_cl * 2#torch.cuda.device_count()
    bs_cl_ts = args.per_gpu_test_batch_size_cl * 2#torch.cuda.device_count()
    aug_strategy = args.aug_strategy.split(",")
    model_path_cl = args.model_path_cl + args.comment\
                    +'.'+str(args.bert_emb_len)\
                    +'.'+str(bs_cl)\
                    +'.'+str(bs_cl_ts)\
                    +'.'+str(args.lr_cl)\
                    +'.'+str(args.temperature0)\
                    +'.'+str(args.temperature1)\
                    +'.'+str(args.loss_para0)\
                    +'.'+str(args.loss_para1)\
                    +'.'+str(args.epochs_cl)\
                    +'.'+".".join(aug_strategy)\
                    +'.'+str(args.fold)

    bs = args.per_gpu_batch_size
    bs_ts = args.per_gpu_test_batch_size
    model_path = args.model_path + args.comment\
                 +'.'+str(args.bert_emb_len)\
                 +'.' + str(bs)\
                 +'.' + str(bs_ts)\
                 +'.' + str(args.lr)\
                 +'.' + str(args.epochs)\
                 +'.' + str(args.dropout)+ args.div_model\
                 +'.' + str(args.fold)

    bs_cll = args.per_gpu_batch_size_cll * 2#torch.cuda.device_count()
    bs_cll_ts = args.per_gpu_test_batch_size_cll * 2#torch.cuda.device_count()
    model_path_cll = args.model_path_cll + args.comment\
                    +'.'+str(args.bert_emb_len)\
                    +'.'+str(bs_cll)\
                    +'.'+str(bs_cll_ts)\
                    +'.'+str(args.lr_cll)\
                    +'.'+str(args.temperature_cll)\
                    +'.'+str(args.epochs_cll)\
                    +'.'+ str(args.fold)

    tmp_prefix1 = args.comment\
                    + '.' + str(args.bert_emb_len) \
                    + '.' + str(bs_cl) \
                    + '.' + str(bs_cl_ts) \
                    + '.' + str(args.lr_cl) \
                    + '.' + str(args.temperature0)\
                    + '.' + str(args.temperature1) \
                    + '.' + str(args.loss_para0)\
                    + '.' + str(args.loss_para1) \
                    + '.' + str(args.epochs_cl) \
                    + '.' + ".".join(aug_strategy)
    tmp_prefix2 = str(bs_cll) \
                    + '.' + str(bs_cll_ts) \
                    + '.' + str(args.lr_cll) \
                    + '.' + str(args.temperature_cll) \
                    + '.' + str(args.epochs_cll)
    tmp_prefix3 = str(bs) \
                    + '.' + str(bs_ts) \
                    + '.' + str(args.lr) \
                    + '.' + str(args.epochs) \
                    + '.' + str(args.dropout) \
                    + '.' + args.div_model

    log_path = args.log_path + tmp_prefix1+'_'+tmp_prefix2+'_'+tmp_prefix3+'.log'
    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.mode == "cl_T5":
        CL.train_model(args.bert_model_path, args.bert_emb_len, bs_cl, bs_cl_ts, args.lr_cl,
                       [args.temperature0, args.temperature1],
                       [args.loss_para0, args.loss_para1], args.epochs_cl, model_path_cl,
                       aug_strategy, logger, args.fold)
    elif args.mode == "cl_list":
        CLL.train_model(bs_cll, bs_cll_ts, args.lr_cll, args.temperature_cll, args.epochs_cll,
                        model_path_cll, logger, args.fold, '../data/attn_data/')
    elif args.mode == "cl_div":
        tmp_dir = '../tmp/'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        DIVDP.gen_bert_embedding(args.bert_model_path, model_path_cl, args.bert_emb_len,
                                 [args.loss_para0, args.loss_para1], tmp_dir, args.fold,
                                 [args.temperature0, args.temperature1])
        DIVDP.divide_five_fold_train_test(tmp_dir, tmp_dir, args.fold)

        best_model_list = ['']
        max_metric_list = 0
        epoch_list = [3,3,3,4,4,4]
        for i in range(len(epoch_list)):
            logger.info('PERIOD = {}'.format(i))
            best_model_list, max_metric_list = run(i, bs, epoch_list[i], args.lr, args.dropout, 768, model_path, logger, tmp_dir, args.fold, 
                                                    [args.temperature0, args.temperature1], best_model_list, max_metric_list, model_path_cll, rs_folder=str(args.epochs_cl)+'/')
            logger.info('hh best_model_list={}'.format(best_model_list))
            logger.info('hh max_metric_list={}'.format(max_metric_list))
        logger.info('done!')
    else:
        print('Incorrect Command!')

