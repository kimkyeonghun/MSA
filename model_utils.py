from torch.nn.utils.rnn import pad_sequence
import torch
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

    #padding text and make attention_mask
    #print(text_examples[0])
    padded_text_ids = pad_example(text_examples)
    #print(padded_text_ids[0])
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
    return padded_text_ids, torch.tensor(text_label,dtype=torch.int64),pad_example(text_type_ids,padding_value=0),text_attention_mask, torch.tensor(text_sentiment,dtype = torch.float),\
    torch.tensor(tWv_examples), torch.tensor(visual_examples), torch.tensor(visual_label,dtype=torch.int64),pad_example(visual_type_ids,padding_value=0), visual_attention_mask, torch.tensor(visual_sentiment,dtype = torch.float),\
    torch.tensor(tWs_examples), torch.tensor(speech_examples), torch.tensor(speech_label,dtype=torch.int64),pad_example(speech_type_ids,padding_value=0), speech_attention_mask, torch.tensor(speech_sentiment,dtype = torch.long),\
    visualWithtext_attention_mask, speechWithtext_attention_mask,rawData
    