import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from pretrain.BertDoc import BertContrastive
from pretrain.DocDataset import ContrasDataset
from tqdm import tqdm
from util.utils import set_seed
set_seed()


def train_model(bert_model_path, max_seq_length, bs_cl, bs_cl_ts, lr_cl, temperature, loss_para, epochs_cl,
                model_path_cl, aug_strategy, logger, fold):
    set_seed()
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    additional_tokens = 1
    tokenizer.add_tokens("[pas_del]")
    model = BertContrastive.from_pretrained(bert_model_path)
    model.set_parameters(temperature=temperature, max_seq_length=max_seq_length, loss_para=loss_para,
                         additional_tokens=additional_tokens)

    model.initial_parameters([9, 10, 11])
    fixed_modules = [model.bert.encoder.layer[6:]]
    for module in fixed_modules:
        for param in module.parameters():
            param.requires_grad = False

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('* number of parameters: %d' % n_params)
    # if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    train_data = '../data/data_content/T5_cl/train' + str(fold) + '.txt'
    test_data = '../data/data_content/T5_cl/test' + str(fold) + '.txt'
    fit(model, train_data, test_data, tokenizer, max_seq_length, bs_cl, bs_cl_ts, lr_cl, epochs_cl, model_path_cl,
        aug_strategy, logger)


def train_step(model, train_data, loss_func):
    set_seed()
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].cuda()
    contras_loss, acc = model.forward(train_data)
    return contras_loss, acc


def fit(model, X_train, X_test, tokenizer, max_seq_length, bs_cl, bs_cl_ts, lr_cl, epochs_cl, model_path_cl,
        aug_strategy, logger):
    set_seed()
    train_dataset = ContrasDataset(X_train, max_seq_length, tokenizer, bs_cl, aug_strategy=aug_strategy)
    train_dataloader = DataLoader(train_dataset, batch_size=bs_cl, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=lr_cl, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    t_total = int(len(train_dataset) * epochs_cl // bs_cl)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(len(train_dataset) // args.batch_size), num_training_steps=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.0), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // bs_cl
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = 1e4

    for epoch in range(epochs_cl):
        logger.info("Epoch " + str(epoch + 1) + "/" + str(epochs_cl) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=120)
        print()
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

            if i > 0 and i % (one_epoch_step // 5) == 0:
                best_result_n = evaluate(model, X_test, best_result, model_path_cl, max_seq_length, bs_cl_ts,
                                         aug_strategy, logger, tokenizer)
                if best_result_n != best_result:
                    flag = 1
                best_result = best_result_n
                model.train()

            avg_loss += loss.item()

        cnt = len(train_dataset) // bs_cl + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result_n = evaluate(model, X_test, best_result, model_path_cl, max_seq_length, bs_cl_ts, aug_strategy,
                                 logger, tokenizer)
        if best_result_n != best_result:
            flag = 1
        best_result = best_result_n
        if flag == 0:
            logger.info("Early Stop!")
            break


def evaluate(model, X_test, best_result, model_path_cl, max_seq_length, bs_cl_ts, aug_strategy, logger, tokenizer,
             is_test=False):
    set_seed()
    y_test_loss, y_test_acc = predict(model, X_test, max_seq_length, bs_cl_ts, aug_strategy, tokenizer)
    result = np.mean(y_test_loss)
    y_test_acc = np.mean(y_test_acc)

    if not is_test and result < best_result:
        best_result = result
        logger.info("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), model_path_cl)

    return best_result


def predict(model, X_test, max_seq_length, bs_cl, aug_strategy, tokenizer):
    set_seed()
    model.eval()
    test_loss = []
    test_dataset = ContrasDataset(X_test, max_seq_length, tokenizer, bs_cl, aug_strategy=aug_strategy)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_cl, shuffle=False, num_workers=8)
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

