import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, RandomSampler

from config import DEVICE
import model_utils
import utils

def train_epoch(args, model, traindata, optimizer, scheduler, tokenizer):
    """
        Input = model : MMBertForPretraining, traindata : torch.utils.data.Dataset, optimizer : AdamW, scheduler : warmup_start, tokenizer : BertTokenizer
        Do train model in set epoch.

        Using Randomsampler and Dataloader, make traindataset to trainDataloader that do training.
        Datalodaer has collate function. collate function does padding at all examples.

        If args.mlm is True, do masking at text(visual, speech)_id.

        After finishing padding and masking, get outputs using model. Next, calculate loss.

        return training_loss divided by training step.
    """
    #Make Dataloader
    trainSampler = RandomSampler(traindata)
    trainDataloader = DataLoader(
        traindata, sampler=trainSampler, batch_size=args.train_batch_size, collate_fn=model_utils.collate
    )

    #Train
    train_loss = 0.0
    text_loss = 0.0
    visual_loss = 0.0
    speech_loss = 0.0
    label_loss = 0.0
    nb_tr_steps = 0
    model.train()
    for step, batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        text_batch, visual_batch, speech_batch, attention_batch, _, _  = batch

        #if args.mlm is true, do masking.
        text_inputs, text_mask_labels = model_utils.mask_tokens(text_batch[0], tokenizer, args) if args.mlm else (text_batch[0], text_batch[0])
        twv_ids, visual_mask_labels = model_utils.mask_tokens(visual_batch[0], tokenizer, args) if args.mlm else (visual_batch[0], visual_batch[0])
        tws_ids, speech_mask_labels = model_utils.mask_tokens(speech_batch[0], tokenizer, args) if args.mlm else (speech_batch[0], speech_batch[0])

        visual_inputs = visual_batch[1].to(DEVICE)
        visual_mask_labels = torch.cat((visual_mask_labels, visual_mask_labels),dim=-1).to(DEVICE)

        speech_inputs = speech_batch[1].to(DEVICE)
        speech_mask_labels = torch.cat((speech_mask_labels, speech_mask_labels),dim=-1).to(DEVICE)

        visual_attention_masks = (attention_batch[0].to(DEVICE), visual_batch[4].to(DEVICE))
        speech_attention_masks = (attention_batch[1].to(DEVICE), speech_batch[4].to(DEVICE))

        #get outputs using model(MMbertForpretraining)

        inputs = (text_inputs.to(DEVICE), visual_inputs, speech_inputs, twv_ids.to(DEVICE), tws_ids.to(DEVICE))
        token_types = (text_batch[2].to(DEVICE), visual_batch[3].to(DEVICE), speech_batch[3].to(DEVICE))
        attention = (text_batch[3].to(DEVICE), visual_attention_masks, speech_attention_masks)
        mmlm_label = (text_mask_labels.to(DEVICE), visual_mask_labels, speech_mask_labels)
        ap_label = (visual_batch[2].to(DEVICE), speech_batch[2].to(DEVICE))

        outputs, _  = model(
            input_ids = inputs,
            token_type_ids = token_types,
            attention_mask = attention,
            masked_labels = mmlm_label,
            ap_label = ap_label,
            sentiment = text_batch[-1].to(DEVICE),
        )
        
        #Need to check
        loss = outputs[0]
        T_loss = outputs[1]
        V_loss = outputs[2]
        S_loss = outputs[3]
        ap_loss= outputs[4]
        L_loss = outputs[5]

        loss.mean().backward()

        train_loss += loss.mean().item()
        if T_loss is not None:
            text_loss += T_loss.mean().item()
        if V_loss is not None:
            visual_loss += V_loss.mean().item()
        if S_loss is not None:
            speech_loss += S_loss.mean().item()

        label_loss += L_loss.mean().item()
        nb_tr_steps +=1

        if (step + 1)& args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
    return train_loss / nb_tr_steps, text_loss / nb_tr_steps ,visual_loss / nb_tr_steps , speech_loss / nb_tr_steps , ap_loss / nb_tr_steps, label_loss / nb_tr_steps

def eval_epoch(args, model, valDataset, tokenizer):
    """
        Input = model : MMBertForPretraining, valdata : torch.utils.data.Dataset, optimizer : AdamW, scheduler : warmup_start, tokenizer : BertTokenizer
        Do eval model in set epoch.

        Using Randomsampler and Dataloader, make valdataset to valDataloader that do evaling.
        Datalodaer has collate function. collate function does padding at all examples.

        If args.mlm is True, do masking at text(visual, speech)_id.

        After finishing padding and masking, get outputs using model with no_grad. Next, calculate loss.

        return eval_loss divided by dev_step.
    """

    model.eval()
    dev_loss = 0
    text_loss = 0.0
    visual_loss = 0.0
    speech_loss = 0.0
    label_loss = 0.0
    nb_dev_steps = 0

    valSampler = RandomSampler(valDataset)
    valDataloader = DataLoader(
        valDataset, sampler=valSampler, batch_size=args.val_batch_size, collate_fn = model_utils.collate
    )
    preds = []
    labels = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            text_batch, visual_batch, speech_batch, attention_batch, _, _  = batch
            text_inputs, text_mask_labels = model_utils.mask_tokens(text_batch[0], tokenizer, args) if args.mlm else (text_batch[0], text_batch[0])
            twv_ids, visual_mask_labels = model_utils.mask_tokens(visual_batch[0], tokenizer, args) if args.mlm else (visual_batch[0], visual_batch[0])
            tws_ids, speech_mask_labels = model_utils.mask_tokens(speech_batch[0], tokenizer, args) if args.mlm else (speech_batch[0], speech_batch[0])

            visual_inputs = visual_batch[1].to(DEVICE)
            visual_mask_labels = torch.cat((visual_mask_labels, visual_mask_labels),dim=-1).to(DEVICE)

            speech_inputs = speech_batch[1].to(DEVICE)
            speech_mask_labels = torch.cat((speech_mask_labels, speech_mask_labels),dim=-1).to(DEVICE)

            visual_attention_masks = (attention_batch[0].to(DEVICE), visual_batch[4].to(DEVICE))
            speech_attention_masks = (attention_batch[1].to(DEVICE), speech_batch[4].to(DEVICE))

            #get outputs using model(MMbertForpretraining)

            inputs = (text_inputs.to(DEVICE), visual_inputs, speech_inputs, twv_ids.to(DEVICE), tws_ids.to(DEVICE))
            token_types = (text_batch[2].to(DEVICE), visual_batch[3].to(DEVICE), speech_batch[3].to(DEVICE))
            attention = (text_batch[3].to(DEVICE), visual_attention_masks, speech_attention_masks)
            mmlm_label = (text_mask_labels.to(DEVICE), visual_mask_labels, speech_mask_labels)
            ap_label = (visual_batch[2].to(DEVICE), speech_batch[2].to(DEVICE))

            outputs, logits  = model(
                input_ids = inputs,
                token_type_ids = token_types,
                attention_mask = attention,
                masked_labels = mmlm_label,
                ap_label = ap_label,
                sentiment = text_batch[-1].to(DEVICE),
            )
            logits = logits.detach().cpu().numpy()
            label_ids = text_batch[-1].detach().cpu().numpy()

            loss = outputs[0]
            T_loss = outputs[1]
            V_loss = outputs[2]
            S_loss = outputs[3]
            ap_loss= outputs[4]
            L_loss = outputs[5]
            #for colab
            #logits = np.expand_dims(logits,axis=-1)

            dev_loss += loss.mean().item()
            nb_dev_steps +=1

            preds.extend(logits)
            labels.extend(label_ids)

            if T_loss is not None:
                text_loss += T_loss.mean().item()
            if V_loss is not None:
                visual_loss += V_loss.mean().item()
            if S_loss is not None:
                speech_loss += S_loss.mean().item()

            label_loss += L_loss.mean().item()

        preds = np.array(preds)
        labels = np.array(labels)           

    return dev_loss / nb_dev_steps, text_loss / nb_dev_steps ,visual_loss / nb_dev_steps , speech_loss / nb_dev_steps , ap_loss / nb_dev_steps, label_loss / nb_dev_steps, preds,labels

def test_CE_score_model(preds,y_test):
    """
        Input = preds, y_test
        
        Using model's emotion detection, cal MAE, ACC, F_score in mosei dataset

        return acc, MAE, F_score
    """
    #MAE
    mae = np.mean(np.absolute(preds - y_test))

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, f_score

def test_MSE_score_model(preds,y_test, use_zero=False):
    """
        Input = preds, y_test
        
        Using model's sentiment analysis, cal MAE, ACC, F_score in mosei dataset

        return acc, MAE, F_score
    """
    mae = np.mean(np.absolute(preds - y_test))

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, f_score

def train(args, model, train_dataset, val_dataset, test_dataset, optimizer, scheduler, tokenizer, logger):
    """
    Train using train_epoch, eval_epoch, test_score_model.

    Adopt EarlyStopping checking valid loss.
    """
    val_losses = []
    test_accuracy = []

    model_save_path = utils.make_date_dir("./model_save")
    logger.info("Model save path: {}".format(model_save_path))

    best_loss = float('inf')
    best_acc = 0
    patience = 0

    if args.num_labels == 1 or args.num_labels == 7:
        test_score = test_MSE_score_model
    else:
        test_score = test_CE_score_model

    for epoch in range(int(args.n_epochs)):
        patience += 1

        logger.info("=====================Train======================")
        train_loss,text_loss,visual_loss,speech_loss,ap_loss,label_loss = train_epoch(args, model, train_dataset, optimizer, scheduler, tokenizer)
        logger.info("[Train Epoch {}] Joint Loss : {} AP Loss : {} Text Loss : {} Visual Loss : {} Speech Loss : {} Label Loss : {}".format(epoch+1,train_loss,ap_loss,text_loss,visual_loss,speech_loss,label_loss))

        logger.info("=====================Valid======================")
        valid_loss,text_loss,visual_loss,speech_loss,ap_loss,label_loss,preds,labels = eval_epoch(args, model, val_dataset, tokenizer)
        logger.info("[Val Epoch {}] Joint Loss : {} AP Loss : {} Text Loss : {} Visual Loss : {} Speech Loss : {} Label Loss : {}".format(epoch+1,valid_loss,ap_loss,text_loss,visual_loss,speech_loss,label_loss))

        logger.info("=====================Test======================")
        _, text_loss,visual_loss,speech_loss,_ ,label_loss,preds,labels = eval_epoch(args, model, test_dataset, tokenizer)
        test_acc,test_mae,test_f_score = test_score(preds,labels)

        logger.info("[Epoch {}] Test_ACC : {}, Test_MAE : {}, Test_F_Score: {}".format(epoch+1,test_acc,test_mae,test_f_score))

        if test_acc > best_acc:
            torch.save(model.state_dict(),os.path.join(model_save_path,'model_'+str(epoch+1)+".pt"))
            best_epoch = epoch
            best_loss = valid_loss
            best_acc = test_acc
            best_mae = test_mae
            best_f_score = test_f_score
            best_preds = preds
            best_labels = labels
            patience = 0

        if patience == 25:
            numpy_save_path = utils.make_date_dir("./numpy_save")
            logger.info("Model save path: {}".format(numpy_save_path))
            np.save(os.path.join(numpy_save_path,'predict.npy'),best_preds)
            np.save(os.path.join(numpy_save_path,'target.npy'),best_labels)
            break

        val_losses.append(valid_loss)
        test_accuracy.append(test_acc)

    logger.info("\n Alpha: {} Beta: {}".format(args.alpha, args.beta))
    logger.info("\n[Best Epoch {}] Best_ACC : {}, Best_MAE : {}, Best_F_Score: {}".format(best_epoch+1, best_acc, best_mae, best_f_score))