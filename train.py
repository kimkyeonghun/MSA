import pickle
import argparse
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequenc
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm, trange

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertModel
from transformers.optimization import AdamW

from MMBertDataset import MMBertDataset
from MMBertForPretraining import MMBertForPretraining
from config import DEVICE

args= argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,choices=["mosi","mosei"],default='mosi')
parser.add_argument("--model",type=str,choices=["bert-base-uncased","bert-large=uncased"],default="bert-base-uncased")
parser.add_argument("--learning_rate",type=float,default=1e-5)
parser.add_argument("--warmup_proportion",type=float,default=1)
parser.add_argument("--n_epochs",type=int,default=50)
parser.add_argument("--train_batch_size",type=int,default=48)
parser.add_argument("--gradient_accumulation_step",type=int,default=1)

args = parser.parse_args()

def prepareForTraining(numTrainOptimizationSteps):
    model = MMBertForPretraining.from_pretrained(args.model, num_labels=1)
    
    model.to(DEVICE)
                    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params" : [
                p for n, p in param_optimizer if not and(nd in n for nd in no_decay)
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


def convertTofeatures(samples,tokenizer):
    features = []
    for idx, sample in enumerate(samples):
        (words,visual,speech), label, segment = sample
        
        tokens, inversions = [],[]
        for i, word in enumerate(list(words)):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            #Because of '##__' case.
            inversions.extend([i]*len(tokenized))
        
        assert len(tokens) == len(inversions)
        
        newVisual, newSpeech = [],[]
        for inv in inversions:
            newVisual.append(visual[inv,:])
            newSpeech.append(speech[inv,:])
        
        visual = np.array(newVisual)
        speech = np.array(newSpeech)
        
        features.append(
            ((words,visual,speech),
            label,
            segment)
        )
    return features

def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "bert-large-uncased":
        return BertTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'bert-large-uncased', but get {}".format(model)
            )

def makeDataset(data):
    tokenizer = get_tokenizer(args.model)
    
    features = convertTofeatures(data,tokenizer)

    config = BertModel.from_pretrained(args.model).config
    
    #Need to modify
    dataset = MMBertDataset(tokenizer,features,config)
    
    return dataset, tokenizer

def loadDataset():
    with open("cmu_{}.pkl".format(args.dataset)) as fr:
        data = pickle.load(fr)
        
    trainData = data["train"]
    valData = data["val"]
    testData = data["test"]
    
    trainDataset, tokenizer = makeDataset(trainData)
    valDataset, _ = makeDataset(valData)
    testDataset, _ = makeDataset(testData)
    
    numTrainOptimizationSteps = (int(len(trianDataset)/ args.train_batch_size / args.gradient_accumulation_step)) *args.n_epochs
    
    return (trainDataset,valDataset,testDataset,numTrainOptimizationSteps,tokenizer)

def pad_example(examples,padding_value=tokenizer.pad_token_id):
        if tokenizer._pad_token is None:
            return pad_sequence(examples,batch_first=True)
        return pad_sequence(examples,batch_first=True, padding_value=padding_value)

def collate(examples):
    text_examples = [None]*len(examples)
    text_label = [None]*len(examples)
    text_type_ids = [None]*len(examples)

    visual_examples = [None]*len(examples)
    visual_label = [None]*len(examples)
    visual_type_ids = [None]*len(examples)

    speech_examples = [None]*len(examples)
    speech_label = [None]*len(examples)
    speech_type_ids = [None]*len(examples)

    for i, (te,tl,tti,ve,vl,vti,se,sl,sti) in enumerate(examples):
        text_examples[i] = te
        visual_examples[i] = ve
        speech_examples[i] = se

        text_label[i] = tl
        visual_label[i] = vl
        speech_label[i] = sl

        text_type_ids[i] = tti
        visual_type_ids[i] = vti
        speech_type_ids[i] = sti

    padded_text_ids = pad_example(text_examples)
    text_attention_mask = torch.ones(padded_text_ids.shape,dtype=torch.int64)
    text_attention_mask[(padded_text_ids == 0)] = 0

    padded_visual_ids = pad_example(visual_examples)
    visual_attention_mask = torch.ones(padded_visual_ids.shape,dtype=torch.int64)
    visual_attention_mask[(padded_visual_ids == 0)] = 0

    padded_speech_ids = pad_example(speech_examples)
    speech_attention_mask = torch.ones(padded_speech_ids.shape,dtype=torch.int64)
    speech_attention_mask[(padded_speech_ids == 0)] = 0

    return padded_text_ids, torch.tensor(text_label,dtype=torch.int64),pad_example(text_type_ids,padding_value=0),text_attention_mask,
    padded_visual_ids, torch.tensor(visual_label,dtype=torch.int64),pad_example(visual_type_ids,padding_value=0),visual_attention_mask,
    padded_speech_ids, torch.tensor(speech_label,dtype=torch.int64),pad_example(speech_type_ids,padding_value=0),speech_attention_mask

def train_epoch(model,traindata,optimizr,scheduler,tokenizer):
    trainSampler = RandomSampler(traindata)
    trainDataloader = DataLoader(
        traindata, sampler=trainSampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    #Train
    epochs_trained = 0
    train_loss = 0.0
    model.train()
    for step, batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        text_ids,text_label,text_token_type_ids,text_attention_masks = batch[0],batch[1],batch[2],batch[3]
        visual_ids,visual_label,visual_token_type_ids,visual_attention_masks = batch[4],batch[5],batch[6],batch[7]
        speech_ids,text_label,speech_token_type_ids,speech_attention_masks = batch[8],batch[9],batch[10],batch[11]
    

def train(model,trainDataset,valDataset,testDataset,optimizer,scheduler,tokenizer):
    val_loss = []
    test_accuracy = []
    for epoch in range(int(args.n_epochs)):
        train_loss = train_epoch(model,trainDataset,optimizer,scheduler,tokenizer)

def main():
    (
        trainDataset,
        valDataset,
        testDataset,
        numTrainOptimizationSteps,
        tokenizer
    ) = loadDataset()
    
    model, opimizer, scheduler = prepareForTraining(numTrainOptimizationSteps)
    
    train(model, trainDataset, valDataset, testDataset, optimizer, scheduler,tokenizer)

if __name__=="__main__":
    main()