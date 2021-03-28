import os
import pickle
import argparse
import logging
from tqdm import tqdm, trange

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from MMBertDataset import MMBertIEMOCAPDataset
#To modify model name MMBertForPretraining -> MMBertForPreTraining
from MMBertForPretraining import MMBertForPretraining
from config import DEVICE, MOSEIVISUALDIM, MOSIVISUALDIM, SPEECHDIM
import config
import utils
import model_utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

parser= argparse.ArgumentParser()
parser.add_argument("--num_labels",type=int,default=8)
parser.add_argument("--model",type=str,choices=["bert-base-uncased","bert-large-uncased"],default="bert-base-uncased")
parser.add_argument("--learning_rate",type=float,default=1e-5)
parser.add_argument("--warmup_proportion",type=float,default=1)
parser.add_argument("--n_epochs",type=int,default=100)
parser.add_argument("--train_batch_size",type=int,default=8)
parser.add_argument("--val_batch_size",type=int,default=2)
parser.add_argument("--test_batch_size",type=int,default=1)
parser.add_argument("--gradient_accumulation_step",type=int,default=1)
parser.add_argument("--mlm",type=bool,default=True)
parser.add_argument("--mlm_probability",type=float,default = 0.15)
parser.add_argument("--max_seq_length",type=int, default = 20)

args = parser.parse_args()

if args.dataset == 'mosi':
    VISUALDIM = MOSIVISUALDIM
else:
    VISUALDIM = MOSEIVISUALDIM

logger, log_dir = utils.get_logger(os.path.join('./logs'))


def prepareForTraining(numTrainOptimizationSteps):
    """
        Input = numTrainOptimizationSteps : Int

        prepareForTraining sets model, optimizer, scheduler.
        
        Model is custom model(MMBertForPretraining) that is influenced by pretrained model like 'bert-based-uncased'

        Use AdamW optimizer with weight_decay(0.01), but don't apply at bias and LayerNorm.

        Use waramup scheduler using Input

        return model : class MMBertForPretraining, optimizer : Admaw, scheduler : warmup_start
    """
    model = MMBertForPretraining.from_pretrained(args.model, num_labels=args.num_labels)
    model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
                    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params" : [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params" : [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr = args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = numTrainOptimizationSteps,
        num_training_steps = args.warmup_proportion * numTrainOptimizationSteps,
    )
    
    return model, optimizer, scheduler

def convertTofeatures(samples):
    features = []
    for idx in range(len(samples['text'])):
        features.append(
            ((samples['text'][idx],samples['vision'][idx],samples['audio'][idx]),
            samples['labels'][idx])
        )
    return features

def get_tokenizer(model):
    """
    Load tokenizer
    # Will be global variable
    """
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "bert-large-uncased":
        return BertTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'bert-large-uncased', but get {}".format(model)
            )

def makeDataset(data):
    """
        Input : data [List]

        Load tokenzier using args.model(bert-base-uncased or bert-large-uncased).If you want, you can change another.
        With Input and tokenizer, we convert raw data to features using at training stage.

        #I think this part is error, so i will change it.
        After converting raw to feature, make dataset using torch.utils.data.dataset in MMBertDataset.py

        #Future work : tokenizer will be global variable
        Return : dataset, tokenizer
    """
    tokenizer = get_tokenizer(args.model)
    features = convertTofeatures(data,tokenizer)

    #Need to modify
    dataset = MMBertIEMOCAPDataset(tokenizer,features)
    
    return dataset, tokenizer

def loadDataset():
    """
        load Data from pickle by producing at pre_processing.py

        Data Strcuture
        data    ----train = (word,visual,speech),label(sentimnet),segment(situation number)
                |
                ----val = (word,visual,speech),label(sentimnet),segment(situation number)
                |
                ----test = (word,visual,speech),label(sentimnet),segment(situation number)
        
        #Future work : tokenizer will be global variable
        return (trainDataset : torch.utils.data.Dataset, valDataset : torch.utils.data.Dataset, testDataset : torch.utils.data.Dataset, numTrainOpimizationSteps,tokenizer)
    """
    #If you don't save pkl to byte form, then you may change read mode.
    logger.info("**********Load IEMOCAP Dataset**********")
    with open("iemocap_data.pkl",'br') as fr:
        data = pickle.load(fr)
        
    trainData = data["train"]
    valData = data["val"]
    testData = data["test"]
    
    logger.info("**********Split Train Dataset**********")
    trainDataset, tokenizer = makeDataset(trainData)
    logger.info("The Length of TrainDataset : {}".format(len(trainDataset)))
    logger.info("**********Finish Train makeDataset**********")

    logger.info("**********Split Valid Dataset**********")
    valDataset, _ = makeDataset(valData)
    logger.info("The Length of ValDataset : {}".format(len(valDataset)))
    logger.info("**********Finish Valid makeDataset**********")

    logger.info("**********Split Test Dataset**********")
    testDataset, _ = makeDataset(testData)
    logger.info("The Length of TestDataset : {}".format(len(testDataset)))
    logger.info("**********Finish Test makeDataset**********")

    #maybe warmup start?
    numTrainOptimizationSteps = (int(len(trainData)/ args.train_batch_size / args.gradient_accumulation_step)) * args.n_epochs
    
    return (trainDataset,valDataset,testDataset,numTrainOptimizationSteps,tokenizer)

    
def mask_tokens(inputs, tokenizer, args):
    """
        Need more modify because of Joint sentence dimension error
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag"
        )
    
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,args.mlm_probability, device = DEVICE)
    special_tokens_mask =[
        tokenizer.get_special_tokens_mask(val,already_has_special_tokens=True) for val in labels.tolist()
    ]

    #Shape probelm
    #RuntimeError: The expanded size of the tensor (35) must match the existing size (51) at non-singleton dimension 2.  Target sizes: [4, 51, 35].  Tensor sizes: [4, 51]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,dtype=torch.bool,device=DEVICE),value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill(padding_mask.cuda(), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape,0.8,device=DEVICE)).bool() & masked_indices
    #Check this line necessary
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    #indices_random = torch.bernoulli(torch.full(labels.shape,0.5,device=DEVICE)).bool() & masked_indices & ~indices_replaced
    #Must make total_vocab_size in globals
    #random_words = torch.randint(config.total_vocab_size,labels.shape,dtype = torch.long,device=DEVICE)
    #inputs[indices_random] = random_words[indices_random]

    return inputs,labels


def train_epoch(model,traindata,optimizer,scheduler,tokenizer):
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
    epochs_trained = 0
    train_loss = 0.0
    text_loss = 0.0
    visual_loss = 0.0
    speech_loss = 0.0
    label_loss = 0.0
    nb_tr_steps = 0
    model.train()
    for step, batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        text_ids, text_label, text_token_type_ids, text_attention_masks,text_sentiment = batch[0], batch[1], batch[2].long(), batch[3], batch[4]
        twv_ids, visual_ids, visual_label,visual_token_type_ids, visual_attention_masks, visual_sentiment = batch[5], batch[6], batch[7], batch[8], batch[9], batch[10]
        tws_ids, speech_ids, speech_label,speech_token_type_ids, speech_attention_masks, speech_sentiment = batch[11], batch[12], batch[13], batch[14], batch[15], batch[16]
        twv_attention_mask, tws_attention_mask = batch[17], batch[18]

        #if args.mlm is true, do masking.
        text_inputs, text_mask_labels = mask_tokens(text_ids,tokenizer,args) if args.mlm else (text_ids,text_ids)
        twv_ids, visual_mask_labels = mask_tokens(twv_ids,tokenizer,args) if args.mlm else (visual_ids, visual_ids)
        tws_ids, speech_mask_labels = mask_tokens(tws_ids,tokenizer,args) if args.mlm else (speech_ids, speech_ids)
        
        #Make tensor cpu to cuda
        text_inputs = text_inputs.to(DEVICE)
        text_mask_labels = text_mask_labels.to(DEVICE)
        text_label = text_label.to(DEVICE)

        visual_inputs = visual_ids.to(DEVICE)
        visual_mask_labels = visual_mask_labels.to(DEVICE)
        visual_label = visual_label.to(DEVICE)

        speech_inputs = speech_ids.to(DEVICE)
        speech_mask_labels = speech_mask_labels.to(DEVICE)
        speech_label = speech_label.to(DEVICE)

        text_token_type_ids = text_token_type_ids.to(DEVICE)
        visual_token_type_ids = visual_token_type_ids.to(DEVICE)
        speech_token_type_ids = speech_token_type_ids.to(DEVICE)

        text_attention_masks = text_attention_masks.to(DEVICE)
        #visual_attention_masks = visual_attention_masks.to(DEVICE)
        #speech_attention_masks = speech_attention_masks.to(DEVICE)

        text_sentiment = text_sentiment.to(DEVICE)
        visual_sentiment = visual_sentiment.to(DEVICE)
        speech_sentiment = speech_sentiment.to(DEVICE)

        twv_ids = twv_ids.to(DEVICE)
        tws_ids = tws_ids.to(DEVICE)

        visual_attention_masks = (twv_attention_mask, visual_attention_masks)
        speech_attention_masks = (tws_attention_mask, speech_attention_masks)

        # get outputs using model(MMbertForpretraining)

        outputs,_ = model(
            text_input_ids = text_inputs,
            visual_input_ids = visual_inputs,
            speech_input_ids = speech_inputs,
            text_with_visual_ids = twv_ids,
            text_with_speech_ids = tws_ids,
            text_token_type_ids = text_token_type_ids,
            visual_token_type_ids = visual_token_type_ids,
            speech_token_type_ids = speech_token_type_ids,
            text_attention_mask = text_attention_masks,
            visual_attention_mask = visual_attention_masks,
            speech_attention_mask = speech_attention_masks,
            text_masked_lm_labels = text_mask_labels,
            visual_masked_lm_labels = visual_mask_labels,
            speech_masked_lm_labels = speech_mask_labels,
            text_next_sentence_label = None,
            visual_next_sentence_label = visual_label,
            speech_next_sentence_label = speech_label,
            text_sentiment = text_sentiment,
        )

        #Need to check
        loss = outputs[0]
        T_loss = outputs[1]
        V_loss = outputs[2]
        S_loss = outputs[3]
        L_loss = outputs[4]

        
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
        
    return train_loss / nb_tr_steps, text_loss / nb_tr_steps ,visual_loss / nb_tr_steps , speech_loss / nb_tr_steps , label_loss / nb_tr_steps

def eval_epoch(model,valDataset,optimizer,scheduler,tokenizer):
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
    nb_dev_examples,nb_dev_steps = 0,0

    valSampler = RandomSampler(valDataset)
    valDataloader = DataLoader(
        valDataset, sampler=valSampler, batch_size=args.val_batch_size, collate_fn = model_utils.collate
    )
    preds = []
    labels = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valDataloader,desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            text_ids, text_label, text_token_type_ids, text_attention_masks,text_sentiment = batch[0], batch[1], batch[2].long(), batch[3], batch[4]
            twv_ids, visual_ids, visual_label, visual_token_type_ids, visual_attention_masks, visual_sentiment = batch[5], batch[6], batch[7], batch[8], batch[9], batch[10]
            tws_ids, speech_ids, speech_label, speech_token_type_ids, speech_attention_masks, speech_sentiment = batch[11], batch[12], batch[13], batch[14], batch[15], batch[16]
            twv_attention_mask, tws_attention_mask = batch[17], batch[18]

            text_inputs, text_mask_labels = mask_tokens(text_ids,tokenizer,args) if args.mlm else (text_ids,text_ids)
            twv_ids, visual_mask_labels = mask_tokens(twv_ids,tokenizer,args) if args.mlm else (visual_ids, visual_ids)
            tws_ids, speech_mask_labels = mask_tokens(tws_ids,tokenizer,args) if args.mlm else (speech_ids, speech_ids)

            visual_mask_labels = visual_mask_labels.to(DEVICE)
            speech_mask_labels = speech_mask_labels.to(DEVICE)

            # print(text_inputs.shape)
            # print(visual_ids.shape)
            # print(speech_ids.shape)
            # print(twv_ids.shape)
            # print(tws_ids.shape)
            # print(text_token_type_ids.shape)
            # print(visual_token_type_ids.shape)
            # print(speech_token_type_ids.shape)
            # print(text_attention_masks.shape)
            # print(visual_attention_masks.shape)
            # print(speech_attention_masks.shape)
            # print(text_mask_labels.shape)
            # print(visual_mask_labels.shape)
            # print(speech_mask_labels.shape)
            # print(visual_label.shape)
            # print(speech_label.shape)
            # print(text_sentiment.shape)

            visual_attention_masks = (twv_attention_mask, visual_attention_masks)
            speech_attention_masks = (tws_attention_mask, speech_attention_masks)

            outputs,logits = model(
                text_input_ids = text_inputs,
                visual_input_ids = visual_ids,
                speech_input_ids = speech_ids,
                text_with_visual_ids = twv_ids,
                text_with_speech_ids = tws_ids,
                text_token_type_ids = text_token_type_ids,
                visual_token_type_ids = visual_token_type_ids,
                speech_token_type_ids = speech_token_type_ids,
                text_attention_mask = text_attention_masks,
                visual_attention_mask = visual_attention_masks,
                speech_attention_mask = speech_attention_masks,
                text_masked_lm_labels = text_mask_labels,
                visual_masked_lm_labels = visual_mask_labels,
                speech_masked_lm_labels = speech_mask_labels,
                text_next_sentence_label = None,
                visual_next_sentence_label = visual_label,
                speech_next_sentence_label = speech_label,
                text_sentiment = text_sentiment,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = text_sentiment.detach().cpu().numpy()
            loss = outputs[0]
            T_loss = outputs[1]
            V_loss = outputs[2]
            S_loss = outputs[3]
            L_loss = outputs[4]

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

    return dev_loss / nb_dev_steps, text_loss / nb_dev_steps ,visual_loss / nb_dev_steps , speech_loss / nb_dev_steps , label_loss / nb_dev_steps, preds,labels

def test_score_model(preds,y_test):
    """
        Input = preds, y_test
        
        Using model's prediction, cal MAE, ACC, F_score in mosei dataset

        return acc, MAE, F_score
    """
    #MAE
    mae = np.mean(np.absolute(preds - y_test))

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, f_score

def train(model,trainDataset,valDataset,testDataset,optimizer,scheduler,tokenizer):
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
    for epoch in range(int(args.n_epochs)):
        patience += 1

        logger.info("=====================Train======================")
        train_loss,text_loss,visual_loss,speech_loss,label_loss = train_epoch(model,trainDataset,optimizer,scheduler,tokenizer)
        logger.info("[Train Epoch {}] Joint Loss : {} Text Loss : {} Visual Loss : {} Speech Loss : {} Label Loss : {}".format(epoch+1,train_loss,text_loss,visual_loss,speech_loss,label_loss))

        logger.info("=====================Valid======================")
        valid_loss,text_loss,visual_loss,speech_loss,label_loss,preds,labels = eval_epoch(model,valDataset,optimizer,scheduler,tokenizer)
        logger.info("[Val Epoch {}] Joint Loss : {} Text Loss : {} Visual Loss : {} Speech Loss : {} Label Loss : {}".format(epoch+1,valid_loss,text_loss,visual_loss,speech_loss,label_loss))

        logger.info("=====================Test======================")
        #test_acc,test_mae,test_f_score = test_score_model(preds,labels)
        test_acc,test_mae,test_f_score = test_mosi_score_model(preds,labels)

        logger.info("[Epoch {}] Test_ACC : {}, Test_MAE : {}, Test_F_Score: {}".format(epoch+1,test_acc,test_mae,test_f_score))

        if test_acc > best_acc:
            torch.save(model.state_dict(),os.path.join(model_save_path,'model_'+str(epoch+1)+".pt"))
            best_acc = test_acc
            patience = 0

        if patience == 15:
            break

        val_losses.append(valid_loss)
        test_accuracy.append(test_acc)


def main():
    logger.info("======================Load and Split Dataset======================")
    (
        trainDataset,
        valDataset,
        testDataset,
        numTrainOptimizationSteps,
        tokenizer
    ) = loadDataset()

    logger.info("======================Prepare For Training======================")
    model, optimizer, scheduler = prepareForTraining(numTrainOptimizationSteps)
    
    train(model, trainDataset, valDataset, testDataset, optimizer, scheduler,tokenizer)

if __name__=="__main__":
    try:
        main()
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
