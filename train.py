import pickle
import argparse
import numpy as np

import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm, trange

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.configuration_bert import BertConfig
#from transformers.modeling_bert import BertModel
from transformers.optimization import AdamW

from MMBertDataset import MMBertDataset
#To modify model name MMBertForPretraining -> MMBertForPreTraining
from MMBertForPretraining import MMBertForPretraining
from config import DEVICE, VISUALDIM, SPEECHDIM

parser= argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,choices=["mosi","mosei"],default='mosei')
parser.add_argument("--model",type=str,choices=["bert-base-uncased","bert-large-uncased"],default="bert-base-uncased")
parser.add_argument("--learning_rate",type=float,default=1e-5)
parser.add_argument("--warmup_proportion",type=float,default=1)
parser.add_argument("--n_epochs",type=int,default=50)
parser.add_argument("--train_batch_size",type=int,default=4)
parser.add_argument("--gradient_accumulation_step",type=int,default=1)
parser.add_argument("--mlm",action="store_true")
parser.add_argument("--mlm_probability",type=float,default = 0.15)
parser.add_argument("--max_seq_length",type=int, default = 200)

args = parser.parse_args()

def prepareForTraining(numTrainOptimizationSteps):
    model = MMBertForPretraining.from_pretrained(args.model, num_labels=1)
    
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
    visual_sep = np.zeros((1,VISUALDIM))
    visual = np.concatenate((visual_sep,visual,visual_sep))
    speech_sep = np.zeros((1,SPEECHDIM))
    speech = np.concatenate((speech_sep,speech,speech_sep))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #segment_id = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    visual_padding = np.zeros((pad_length,VISUALDIM))
    visual = np.concatenate((visual,visual_padding))

    speech_padding = np.zeros((pad_length,SPEECHDIM))
    speech = np.concatenate((speech,speech_padding))

    padding = [0] * pad_length

    input_ids += padding
    #segment_id += padding
    input_mask += padding

    #deleted segment_id(I think this is not required.)
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
        
        features.append(
            ((input_ids,visual,speech,input_mask),
            label,
            segment)
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

    ##Because of segmentation fault
    config =BertConfig()

    #Need to modify
    dataset = MMBertDataset(tokenizer,features,config)
    
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
        return (trainDataset : torch.utils.data.Dataset,valDataset : torch.utils.data.Dataset,testDataset : torch.utils.data.Dataset, numTrainOpimizationSteps,tokenizer)
    """
    #If you don't save pkl to byte form, then you may change read mode.
    with open("cmu_{}.pkl".format(args.dataset),'br') as fr:
        data = pickle.load(fr)
        
    trainData = data["train"]
    valData = data["val"]
    testData = data["test"]
    
    trainDataset, tokenizer = makeDataset(trainData)
    print("Finish Train makeDataset")
    valDataset, _ = makeDataset(valData)
    print("Finish val makeDataset")
    testDataset, _ = makeDataset(testData)
    print("Finish test makeDataset")

    #maybe warmup start?
    numTrainOptimizationSteps = (int(len(trainData)/ args.train_batch_size / args.gradient_accumulation_step)) *args.n_epochs
    
    return (trainDataset,valDataset,testDataset,numTrainOptimizationSteps,tokenizer)


def mask_tokens(inputs, tokenizer,args):

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag"
        )
    
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,args.mlm_probability)
    special_tokens_mask =[
        tokenizer.get_special_tokens_mask(val,already_has_special_tokens=True) for val in labels.tolist()
    ]

    #No glue token

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,dtype=torch.boll),values=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape,0.8)).bool() & masked_indices
    #Check this line necessary
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape,0.5)).bool() & masked_indices & ~indices_replaced
    #Must make total_vocab_size in globals
    random_words = torch.randint(globals.total_vocab_size,labels.shape,dtype = torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs,lables

def train_epoch(model,traindata,optimizr,scheduler,tokenizer):

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

        return padded_text_ids, torch.tensor(text_label,dtype=torch.int64),pad_example(text_type_ids,padding_value=0),text_attention_mask,\
        padded_visual_ids, torch.tensor(visual_label,dtype=torch.int64),pad_example(visual_type_ids,padding_value=0),visual_attention_mask,\
        padded_speech_ids, torch.tensor(speech_label,dtype=torch.int64),pad_example(speech_type_ids,padding_value=0),speech_attention_mask

    trainSampler = RandomSampler(traindata)
    trainDataloader = DataLoader(
        traindata, sampler=trainSampler, batch_size=args.train_batch_size, collate_fn=collate
    )
    #Train
    epochs_trained = 0
    train_loss = 0.0
    nb_tr_steps = 0
    model.train()
    for step, batch in enumerate(tqdm(trainDataloader,desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        text_ids,text_label,text_token_type_ids,text_attention_masks = batch[0],batch[1],batch[2],batch[3]
        visual_ids,visual_label,visual_token_type_ids,visual_attention_masks = batch[4],batch[5],batch[6],batch[7]
        speech_ids,speech_label,speech_token_type_ids,speech_attention_masks = batch[8],batch[9],batch[10],batch[11]

        text_inputs, text_mask_labels = mask_tokens(text_ids,tokenizer,args) if args.mlm else (text_ids,text_ids)
        visual_inputs, visual_mask_labels = mask_tokens(visual_ids,tokenizer,args) if args.mlm else (visual_ids, visual_ids)
        speech_inputs, speech_mask_labels = mask_tokens(speech_ids,tokenizer,args) if args.mlm else (speech_ids, speech_ids)
        
        text_inputs = text_inputs.to(DEVICE)
        text_mask_labels = text_mask_labels.to(DEVICE)
        text_label = text_label.to(DEVICE)

        visual_inputs = visual_inputs.to(DEVICE)
        visual_mask_labels = visual_mask_labels.to(DEVICE)
        visual_label = visual_label.to(DEVICE)

        speech_inputs = speech_inputs.to(DEVICE)
        speech_mask_labels = speech_mask_labels.to(DEVICE)
        speech_label = speech_label.to(DEVICE)

        text_token_type_ids = text_token_type_ids.to(DEVICE)
        visual_token_type_ids = visual_token_type_ids.to(DEVICE)
        speech_token_type_ids = speech_token_type_ids.to(DEVICE)

        text_attention_masks = text_attention_masks.to(DEVICE)
        visual_attention_masks = visual_attention_masks.to(DEVICE)
        speech_attention_masks = speech_attention_masks.to(DEVICE)

        outputs = model(
            text_input_ids = text_inputs,
            visual_input_ids = visual_inputs,
            speech_input_ids = speech_inputs,
            text_token_type_ids = text_attention_masks,
            visual_token_type_ids = visual_attention_masks,
            speech_token_type_ids = speech_token_type_ids,
            text_attention_mask = text_attention_masks,
            visual_attention_mask = visual_attention_masks,
            speech_attention_mask = speech_attention_masks,
            text_masked_lm_labels = text_mask_labels,
            visual_masked_lm_labels = visual_mask_labels,
            speech_masked_lm_labels = speech_mask_labels,
            text_next_sentence_label = text_label,
            visual_next_sentence_label = visual_label,
            speech_next_sentence_label = speech_label,
        )

        #Need to check
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1),label_ids.view(-1))

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps +=1

        if (step + 1)& args.gradient_accumulation_step == 0:
            opimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return tr_loss / nb_tr_steps

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
    
    model, optimizer, scheduler = prepareForTraining(numTrainOptimizationSteps)
    
    train(model, trainDataset, valDataset, testDataset, optimizer, scheduler,tokenizer)

if __name__=="__main__":
    main()