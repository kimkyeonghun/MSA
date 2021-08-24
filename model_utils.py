from torch.nn.utils.rnn import pad_sequence
import torch

from config import DEVICE

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

    # indices_random = torch.bernoulli(torch.full(labels.shape,0.5,device=DEVICE)).bool() & masked_indices & ~indices_replaced
    # # Must make total_vocab_size in globals
    # random_words = torch.randint(config.total_vocab_size,labels.shape,dtype = torch.long,device=DEVICE)
    # inputs[indices_random] = random_words[indices_random]

    return inputs,labels

def pad_example(examples,padding_value=0):
    #padding_value will be tokenizer.pad_token_id
    """
        Do Padding using pad_sequence in torch.nn.utils.rnn.
        padding_value's default is tokenizer.pad_token_id, but we use padding_value as 0.
    """
    # if tokenizer._pad_token is None:
    #     return pad_sequence(examples,batch_first=True)
    return pad_sequence(examples,batch_first=True, padding_value=padding_value)

def collate(examples):
    """
        collate is used at collate_fun in Dataloader.
        At each example like token_id, label, type_id, we make those to list.

        Next, do padding with function pad_example at each_token_id and make attention_mask tensor that divide which token is padding or not.
        
        mosei has text sentimnet that has torch.float

        return padded_text_ids : torch.tensor, text_label : torch.tensor, text_type_ids : torch.tensor, text_attention_mask : torch.tensor, text_sentiment : torch.tensor(torch.long)

        return is repeated each modality.
    """
    #Init None list.
    text_examples = [None]*len(examples)
    text_label = [None]*len(examples)
    text_type_ids = [None]*len(examples)
    text_sentiment = [None]*len(examples)

    tWv_examples = [None]*len(examples)
    visual_examples = [None]*len(examples)
    visual_label = [None]*len(examples)
    visual_type_ids = [None]*len(examples)
    visual_sentiment = [None] * len(examples)

    tWs_examples = [None]*len(examples)
    speech_examples = [None]*len(examples)
    speech_label = [None]*len(examples)
    speech_type_ids = [None]*len(examples)
    speech_sentiment = [None] * len(examples)
    rawData = [None] * len(examples)

    # make examples to each list.
    for i, (te,tl,tti,ts,twv,ve,vl,vti,vs,tws,se,sl,sti,ss,raw) in enumerate(examples):
        text_examples[i] = te
        visual_examples[i] = ve
        speech_examples[i] = se
        tWv_examples[i] = twv
        tWs_examples[i] = tws

        assert len(te)==len(ve)==len(se)==len(twv)==len(tws)

        text_label[i] = tl
        visual_label[i] = vl
        speech_label[i] = sl

        text_type_ids[i] = tti
        visual_type_ids[i] = vti
        speech_type_ids[i] = sti

        text_sentiment[i] = ts
        visual_sentiment[i] = vs
        speech_sentiment[i] = ss
        rawData[i] = raw

    if text_sentiment[0].dtype==torch.int64:
        text_sentiment = torch.tensor(text_sentiment,dtype = torch.long)
        visual_sentiment = torch.tensor(visual_sentiment,dtype = torch.long)
        speech_sentiment = torch.tensor(speech_sentiment,dtype = torch.long)
    else:
        text_sentiment = torch.tensor(text_sentiment,dtype = torch.float)
        visual_sentiment = torch.tensor(visual_sentiment,dtype = torch.float)
        speech_sentiment = torch.tensor(speech_sentiment,dtype = torch.float)

    #padding text and make attention_mask
    padded_text_ids = pad_example(text_examples)
    text_attention_mask = torch.ones(padded_text_ids.shape,dtype=torch.int64)
    #padding part is masking with 0.
    text_attention_mask[(padded_text_ids == 0)] = 0

    #padding visual and make attention_mask
    # padded_visual_ids = pad_example(visual_examples)
    visual_attention_mask = torch.ones(torch.tensor(visual_examples).shape,dtype=torch.int64)
    visual_attention_mask[(torch.tensor(visual_examples) == 0)] = 0

    visualWithtext_attention_mask = torch.ones(torch.tensor(tWv_examples).shape,dtype=torch.int64)
    visualWithtext_attention_mask[(torch.tensor(tWv_examples)==0)]==0

    #padding speech and make attention_mask
    # padded_speech_ids = pad_example(speech_examples)
    speech_attention_mask = torch.ones(torch.tensor(speech_examples).shape,dtype=torch.int64)
    speech_attention_mask[(torch.tensor(speech_examples) == 0)] = 0

    speechWithtext_attention_mask = torch.ones(torch.tensor(tWs_examples).shape,dtype=torch.int64)
    speechWithtext_attention_mask[(torch.tensor(tWs_examples)==0)]==0

    # MSE(text_sentiment - > torch.float), CE (torch.long)
    return padded_text_ids, torch.tensor(text_label,dtype=torch.int64),pad_example(text_type_ids,padding_value=0),text_attention_mask, text_sentiment,\
    torch.tensor(tWv_examples), torch.tensor(visual_examples), torch.tensor(visual_label,dtype=torch.int64),pad_example(visual_type_ids,padding_value=0), visual_attention_mask, visual_sentiment,\
    torch.tensor(tWs_examples), torch.tensor(speech_examples), torch.tensor(speech_label,dtype=torch.int64),pad_example(speech_type_ids,padding_value=0), speech_attention_mask, speech_sentiment,\
    visualWithtext_attention_mask, speechWithtext_attention_mask, rawData
        