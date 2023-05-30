import os, argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer, BertModel
from pretrain.BertList import ListContrastive
from pretrain.ListDataset import ContrasDatasetList
from util.utils import set_seed
set_seed()


def train_model(bs_cl, bs_cl_ts, lr_cl, temperature, epochs_cl, model_path_cl, logger, fold, emb_dir):
    set_seed()
    model = ListContrastive(temperature, 8, 2, 0)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('* number of parameters: %d' % n_params)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    train_data = '../data/attn_data/train_list.pos.' + str(fold) + '.txt'
    test_data = '../data/attn_data/dev_list.pos.' + str(fold) + '.txt'
    fit(model, train_data, test_data, bs_cl, bs_cl_ts, lr_cl, temperature, epochs_cl, model_path_cl, logger, emb_dir)


def train_step(model, train_data, loss_func):
    set_seed()
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].cuda()
    contras_loss, acc = model.forward(train_data)
    return contras_loss, acc


def fit(model, X_train, X_test, bs_cl, bs_cl_ts, lr_cl, temperature, epochs_cl, model_path_cl, logger, emb_dir):
    set_seed()
    train_dataset = ContrasDatasetList(X_train, bs_cl, emb_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=bs_cl, shuffle=True, num_workers=6)
    optimizer = AdamW(model.parameters(), lr=lr_cl, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    t_total = int(len(train_dataset) * epochs_cl // bs_cl)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(len(train_dataset) // args.batch_size), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // bs_cl
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = 1e4

    for epoch in range(epochs_cl):
        logger.info("Epoch " + str(epoch + 1) + "/" + str(epochs_cl) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=120)
        flag = 0
        for i, training_data in enumerate(epoch_iterator):
            loss, acc = train_step(model, training_data, bce_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                lr_cl = param_group['lr']
            epoch_iterator.set_postfix(lr=lr_cl, loss=loss.mean().item(), acc=acc.mean().item())

            if i > 0 and i % 100 == 0:
                logger.info("Step " + str(i) + ": " + str(loss.item()) + "\n")

            if i > 0 and i % (one_epoch_step // 20) == 0:
                best_result_n = evaluate(model, X_test, best_result, model_path_cl, bs_cl_ts, logger, emb_dir)
                if best_result_n >= best_result:
                    flag += 1
                else:
                    flag = 0
                if flag == 2:
                    break
                best_result = best_result_n
                model.train()

            avg_loss += loss.item()
        if flag == 2:
            break
        cnt = len(train_dataset) // bs_cl + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result_n = evaluate(model, X_test, best_result, model_path_cl, bs_cl_ts, logger, emb_dir)


def evaluate(model, X_test, best_result, model_path_cl, bs_cl_ts, logger, emb_dir, is_test=False):
    set_seed()
    y_test_loss, y_test_acc = predict(model, X_test, bs_cl_ts, emb_dir)
    result = np.mean(y_test_loss)
    y_test_acc = np.mean(y_test_acc)

    if not is_test and result < best_result:
        best_result = result
        logger.info("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), model_path_cl)

    return best_result


def predict(model, X_test, bs_cl, emb_dir):
    set_seed()
    model.eval()
    test_loss = []
    test_dataset = ContrasDatasetList(X_test, bs_cl, emb_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_cl, shuffle=False, num_workers=6)
    y_test_loss = []
    y_test_acc = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].cuda()
            test_loss, test_acc = model.forward(test_data)
            test_loss = test_loss.mean()
            test_acc = test_acc.mean()
            y_test_loss.append(test_loss.item())
            y_test_acc.append(test_acc.item())
    y_test_loss = np.asarray(y_test_loss)
    y_test_acc = np.asarray(y_test_acc)
    return y_test_loss, y_test_acc


