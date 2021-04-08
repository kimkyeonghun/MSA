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

from MMBertDataset import MMBertDataset
#To modify model name MMBertForPretraining -> MMBertForPreTraining
from MMBertForPretraining import MMBertForPretraining

from config import DEVICE, MOSEIVISUALDIM, MOSIVISUALDIM, SPEECHDIM
import config
import utils
import model_utils


parser= argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,choices=["mosi","mosei"],default='mosei')
parser.add_argument("--emotion",type=str,default='sentiment')
parser.add_argument("--num_labels",type=int,default=1)
parser.add_argument("--model",type=str,choices=["bert-base-uncased","bert-large-uncased"],default="bert-base-uncased")
parser.add_argument("--learning_rate",type=float,default=1e-4)
parser.add_argument("--warmup_proportion",type=float,default=1)
parser.add_argument("--n_epochs",type=int,default=100)
parser.add_argument("--test_batch_size",type=int,default=1)
parser.add_argument("--gradient_accumulation_step",type=int,default=1)
parser.add_argument("--dir",type=str,required=True)
parser.add_argument("--model_num",type=str,required = True)
parser.add_argument("--max_seq_length",type=int, default = 100)
args = parser.parse_args()

if args.dataset == 'mosi':
    VISUALDIM = MOSIVISUALDIM
else:
    VISUALDIM = MOSEIVISUALDIM

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

def prepare_inputs(tokens, visual, speech, tokenizer):
    """
        Input = tokens : List, visual : List, speech : List, tokenizer : BertTokenizer
        
        Convert token to token_id and make (token,visual,speech) length to max_seq_length using padding.

        return input_ids : List, visual : List, speech : List, input_mask: List

    """
    #Need new visual and speech sep token
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    visual_SEP = np.zeros((1,VISUALDIM))
    visual = np.concatenate((visual,visual_SEP))
    speech_SEP = np.zeros((1,SPEECHDIM))
    speech = np.concatenate((speech,speech_SEP))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)

    pad_len = args.max_seq_length - len(input_mask)
    visual_padding = np.zeros((pad_len+1,VISUALDIM))
    visual = np.concatenate((visual,visual_padding))
    speech_padding = np.zeros((pad_len+1,SPEECHDIM))
    speech = np.concatenate((speech,speech_padding))

    padding = [0]*pad_len
    input_ids += padding
    input_mask += padding

    return input_ids, visual, speech, input_mask

def convertTofeatures(samples,tokenizer):
    """
        Input = samples : [List], tokenizer(will...)
            - samples[0] : (words,visual,speech),label, segment
                    -- they are pair that aligned by text pivot.
        
        Using tokenizer, toknize words and appent tokens list. In this time, toknizer makes "##__ token" because of wordcepiece. So make inversion list too.

        Using inversion list, make visual and speech length same sa tokens length.

        They have too many tokens.Therefore, truncate about max_seq_length == 200.

        In prepare_input, convert token to token_id and make (token,visual,speech) length to max_seq_length using padding.

        we store those things at features: List
        features - ((input_ids:token_ ids, visual, speech, input_mask), label, segment)

        return features
    """
    features = []
    for idx, sample in enumerate(samples):
        (words,visual,speech), label, segment = sample
        
        #Tokenize
        tokens, inversions = [],[]
        for i, word in enumerate(list(words)):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            #Because of '##__' case.
            inversions.extend([i]*len(tokenized))
        
        assert len(tokens) == len(inversions)
        
        #Make same length between token, visual, speech
        newVisual, newSpeech = [],[]
        for inv in inversions:
            newVisual.append(visual[inv,:])
            newSpeech.append(speech[inv,:])
        
        visual = np.array(newVisual)
        speech = np.array(newSpeech)

        #Truncate
        if len(tokens) > args.max_seq_length-2:
            tokens = tokens[: args.max_seq_length-2]
            visual = visual[: args.max_seq_length-2]
            speech = speech[: args.max_seq_length-2]

        #padding
        input_ids,visual,speech,input_mask = prepare_inputs(tokens,visual,speech,tokenizer)
        try:
            input_ids.detach().numpy().squeeze().shape[0]
        except:
            pass
        features.append(
            ((input_ids,visual,speech,input_mask),
            label,
            segment,words)
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
    dataset = MMBertDataset(tokenizer,features,args.dataset,args.emotion)
    
    return dataset, tokenizer

def test_epoch(model,testDataloader):
    """
        Input = model : MMBertForPretraining, testdata : torch.utils.data.Dataloader
        Do test model in set epoch.

        After finishing padding, get outputs using model.

        return predict and true
    """
    model.eval()
    preds = []
    labels = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(testDataloader):
            rawData = batch[-1]
            batch = tuple(t.to(DEVICE) for t in batch[:-1])

            text_ids, text_label, text_token_type_ids, text_attention_masks,text_sentiment = batch[0], batch[1], batch[2].long(), batch[3], batch[4]
            twv_ids, visual_ids, visual_label, visual_token_type_ids, visual_attention_masks, visual_sentiment = batch[5], batch[6], batch[7], batch[8], batch[9], batch[10]
            tws_ids, speech_ids, speech_label, speech_token_type_ids, speech_attention_masks, speech_sentiment = batch[11], batch[12], batch[13], batch[14], batch[15], batch[16]
            twv_attention_mask, tws_attention_mask = batch[17], batch[18]
            
            text_inputs, text_mask_labels = mask_tokens(text_ids,tokenizer,args) if False else (text_ids,text_ids)
            twv_ids, visual_mask_labels = mask_tokens(twv_ids,tokenizer,args) if False else (twv_ids, visual_ids)
            tws_ids, speech_mask_labels = mask_tokens(tws_ids,tokenizer,args) if False else (tws_ids, speech_ids)

            visual_mask_labels = visual_mask_labels.to(DEVICE)
            speech_mask_labels = speech_mask_labels.to(DEVICE)

            visual_attention_masks = (twv_attention_mask, visual_attention_masks)
            speech_attention_masks = (tws_attention_mask, speech_attention_masks)

            _,logits = model(
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
                text_masked_lm_labels = None,
                visual_masked_lm_labels = None,
                speech_masked_lm_labels = None,
                text_next_sentence_label = None,
                visual_next_sentence_label = None,
                speech_next_sentence_label = None,
                text_sentiment = text_sentiment,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = text_sentiment.detach().cpu().numpy()
            
            preds.extend(logits)
            targets.extend(rawData)
            labels.extend(label_ids)
        preds = np.array(preds)
        labels = np.array(labels)

        #Save results
        import pickle
        with open('target.pkl','wb') as f:
            pickle.dump(targets,f)

        np.save('./preds',preds)
        np.save('./y_test',labels)

    return preds, labels


def ACC7(value):
    """
    for 7 label
    """
    for i,v in enumerate(value):
        if v < -2:
            value[i] = -3
        elif -2 <= v < -1:
            value[i] = -2
        elif -1 <= v < 0:
            value[i] = -1
        elif v==0:
            value[i] = 0
        elif 0 < v <= 1:
            value[i] = 1
        elif 1 < v <= 2:
            value[i] = 2
        elif v > 2:
            value[i] = 3
    return value

def ACC3(preds,y_test):
    """
    for 2 label except 0
    """
    newPreds = []
    newYtest = []
    for i,(p,y) in enumerate(zip(preds,y_test)):
        if y > 0:
            newPreds.append(1)
            if p > 0:
                newYtest.append(1)
            else:
                newYtest.append(0)
        elif y < 0:
            newPreds.append(0)
            if p > 0:
                newYtest.append(1)
            else:
                newYtest.append(0)

    return newPreds,newYtest

def test_score_model(model,testDataset):
    """
        Input = model : MMBertForPretraining, testDataset : torch.utils.data.Dataset
        
        Using model's prediction, cal MAE, ACC, F_score

        return acc, MAE, F_score
    """

    testSampler = RandomSampler(testDataset)
    testDataloader = DataLoader(
        testDataset, sampler=testSampler, batch_size=args.test_batch_size, collate_fn = model_utils.collate
    )    

    preds, y_test = test_epoch(model,testDataloader)

    mae = np.mean(np.absolute(preds - y_test))

    y_test7 = ACC7(y_test)
    preds7 = ACC7(preds)

    acc7 = accuracy_score(y_test7, preds7)

    preds2 = preds >= 0
    y_test2 = y_test >= 0

    acc2 = accuracy_score(y_test2, preds2)
    f_score = f1_score(y_test2, preds2, average="weighted")

    preds3, y_test3 = ACC3(preds, y_test)

    acc3 = accuracy_score(y_test3, preds3)
    
    return acc7, acc3, acc2, mae, f_score

def main():
    with open("cmu_{}.pkl".format(args.dataset),'br') as fr:
        data = pickle.load(fr)

    testData = data["test"]

    testDataset, _ = makeDataset(testData)
    numTrainOptimizationSteps = (int(len(testData)/ args.test_batch_size / args.gradient_accumulation_step)) * args.n_epochs
    model, optimizer, scheduler = prepareForTraining(numTrainOptimizationSteps)

    model_path = os.path.join('./model_save',args.dir,'model_'+args.model_num+'.pt')
    model.load_state_dict(torch.load(model_path,map_location=DEVICE))
    
    print("===================LOAD COMPLETE===================")

    test_acc7, test_acc3, test_acc2, test_mae,test_f_score = test_score_model(model,testDataset)

    print("Test_ACC7 : {}, Test_ACC3 : {}, Test_ACC2 : {}, Test_MAE : {}, Test_F_Score: {}".format(test_acc7,test_acc3,test_acc2,test_mae,test_f_score))

if __name__== '__main__':
    main()