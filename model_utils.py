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

    visual_examples = [None]*len(examples)
    visual_label = [None]*len(examples)
    visual_type_ids = [None]*len(examples)
    visual_sentiment = [None] * len(examples)

    speech_examples = [None]*len(examples)
    speech_label = [None]*len(examples)
    speech_type_ids = [None]*len(examples)
    speech_sentiment = [None] * len(examples)

    # make examples to each list.
    for i, (te,tl,tti,ts,ve,vl,vti,vs,se,sl,sti,ss) in enumerate(examples):
        text_examples[i] = te
        visual_examples[i] = ve
        speech_examples[i] = se

        text_label[i] = tl
        visual_label[i] = vl
        speech_label[i] = sl

        text_type_ids[i] = tti
        visual_type_ids[i] = vti
        speech_type_ids[i] = sti

        text_sentiment[i] = ts
        visual_sentiment[i] = vs
        speech_sentiment[i] = ss

    #padding text and make attention_mask
    padded_text_ids = pad_example(text_examples)
    text_attention_mask = torch.ones(padded_text_ids.shape,dtype=torch.int64)
    #padding part is masking with 0.
    text_attention_mask[(padded_text_ids == 0)] = 0

    #padding visual and make attention_mask
    padded_visual_ids = pad_example(visual_examples)
    visual_attention_mask = torch.ones(padded_visual_ids.shape,dtype=torch.int64)
    visual_attention_mask[(padded_visual_ids == 0)] = 0

    #padding speech and make attention_mask
    padded_speech_ids = pad_example(speech_examples)
    speech_attention_mask = torch.ones(padded_speech_ids.shape,dtype=torch.int64)
    speech_attention_mask[(padded_speech_ids == 0)] = 0

    #mosi and mosei text sentiment type
    return padded_text_ids, torch.tensor(text_label,dtype=torch.int64),pad_example(text_type_ids,padding_value=0),text_attention_mask, torch.tensor(text_sentiment,dtype = torch.long),\
    padded_visual_ids, torch.tensor(visual_label,dtype=torch.int64),pad_example(visual_type_ids,padding_value=0),visual_attention_mask, torch.tensor(visual_sentiment,dtype = torch.long),\
    padded_speech_ids, torch.tensor(speech_label,dtype=torch.int64),pad_example(speech_type_ids,padding_value=0),speech_attention_mask, torch.tensor(speech_sentiment,dtype = torch.long)
    