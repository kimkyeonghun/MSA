import os
import random

from config import MOSIVISUALDIM, MOSEIVISUALDIM, MELDVISUALDIM, SPEECHDIM, DEVICE

import torch
from torch.utils.data import Dataset
import numpy as np

from transformers.modeling_bert import BertEmbeddings

cudas = DEVICE

emotions = ['sentiment','happy','sad','anger','surprise','disgust','fear']

class MMBertDataset(Dataset):
    """
    MMBertDataset is to make text Sentence pair or joint Sentence pair between text and other modality.

    label = 1 -> not next sentence or not pair sentence
    label = 0  -> next sentence or pair sentnece

    1. create_next_sentence_pair(i,max_token_len = -1)
        -- Make text Sentence pair

    2. create_concat_joint sentence(i,mode,max_token_len = -1)
        -- Make joint Sentence pair

    3. count
        -- Count total len data

    4. __len__
        -- return count result

    5.__getitem__
        -- Do 2 and 3

            text_sentence, text_label, text_token_type_ids = create_next_sentence_pair(i,max_token_len = 75)
                text_sentence: Tuple([List]), text_label : torch.tensor, text_token_type_ids : torch.tensor

            tAv_sentence, tAv_label, tAv_token_type_ids = create_concat_joint_sentence(i,'visual',max_token_len = -1)
                tAv_sentence : Tuple([List]), tAv_label : torch.tensor, tAv_token_type_ids : torch.tensor

            tAs_sentence, tAs_label, tAs_token_type_ids = create_concat_joint_sentence(i,'speech',max_token_len = -1)
                tAs_sentence : Tuple([List]), tAs_label : torch.tensor, tAs_token_type_ids : torch.tensor

            return text_sentence, text_label, text_token_type_ids, tAv_sentence, tAv_label, tAv_token_type_ids, tAs_sentence, tAs_label, tAs_token_type_ids
    """
    def __init__(self,tokenizer,features,dataset,task, num_labels):
        self.tokenizer = tokenizer
        self.items = features
        self.total_item = self.count()
        self.dataset =dataset
        self.task = task
        self.num_labels = num_labels
        if self.dataset =='mosi':
            self.VISUALDIM = MOSIVISUALDIM
        elif self.dataset == 'mosei' or self.dataset == 'iemocap':
            self.VISUALDIM = MOSEIVISUALDIM
        elif self.dataset == 'meld':
            self.VISUALDIM = MELDVISUALDIM
    
    def sentiment_selection(self,sentiment,mode):
        if self.dataset == 'mosei':
            if self.task == "sentiment":
                if mode == '2':
                    if sentiment[0]>=0:
                        return torch.tensor([1])
                    else:
                        return torch.tensor([0])
                elif mode == '7':
                    return torch.tensor(sentiment[0])
                elif mode == '1':
                    return torch.tensor(sentiment[0])/3
            else:
                if mode == '2':
                    if torch.tensor(sentiment[emotions.index(self.task)]) !=0:
                        return torch.tensor([1])
                    else:
                        return torch.tensor([0])
                elif mode == '6':
                    return torch.argmax(torch.tensor(sentiment[1:]))
        elif self.dataset == 'mosi':
            if mode == '2':
                if sentiment >=0:
                    return torch.tensor([1])
                else:
                    return torch.tensor([0])
            elif mode =='7':
                return torch.tensor(sentiment)
            elif mode == '1':
                return torch.tensor(sentiment)/3
        elif self.dataset == 'iemocap':
            if mode == '2':
                return torch.argmax(torch.argmax(torch.tensor(sentiment),dim=-1))
        elif self.dataset == 'meld':
            if mode == '3':
                return torch.tensor(np.where(np.array(sentiment)==1)[0]) 

    def create_concat_joint_sentence(self, i, mode, max_token_len = -1):
        """
        If mode is 'visual', pairIndex = 1, elif mode is 'speech', pairIndex = 2

        If last sentencse, make label = 0(edge case)
        
        Choose random float between 0 and 1,
            if random number is over 0.5, make next sentence pair(label = 0)
            else, make next sentence unpair(label=1) using random chooice that do not allow duplicate select (label = 1)

        pairSentence : Tuple([List])
            ([CLS],text_sentence,pair_sentence)

        label : torch.tensor
            (1 or 0)
            
        token_type_ids : torch.tensor
            (First_sentence_token_type_ids(0) + Second_sentence_token_type_ids(1))

        return pairSentence, label, token_type_ids
        """
        pairIndex = -1
        firstIndex = -1
        secondIndex = -1
        edgeCase = False
        sentiment = 0
        label = -1
        
        if mode == 'visual':
            pairIndex = 1
            pairDim = self.VISUALDIM
        elif mode == 'speech':
            pairIndex = 2
            pairDim = SPEECHDIM
            
        assert pairIndex != -1

        sentiment = self.sentiment_selection(self.items[i][1][0],str(self.num_labels))

        if i == len(self.items)-1:
            firstIndex = i
            secondIndex = i
            label = 1
            edgeCase = True
        
        if not edgeCase:
            r = random.uniform(0,1)
            
            if r > 0.2:
                firstIndex = i
                secondIndex = i
                label = 1
            else:
                firstIndex = i
                secondIndex = random.choice(range(len(self.items)))
                while firstIndex == secondIndex:
                    secondIndex = random.choice(range(len(self.items)))
                label = 0
        
        #textSentence = list(self.items[firstIndex][0][0].detach().numpy().squeeze())
        textSentence = list(self.items[firstIndex][0][0])
        pairSentence = self.items[secondIndex][0][pairIndex]
        
        textTokenTypeIds = np.zeros(len(textSentence))
        pairTokenTypeIds = np.ones(len(pairSentence))

        return textSentence, pairSentence, torch.tensor(label,dtype=torch.int64,device=cudas),torch.cat((
            torch.tensor(textTokenTypeIds,device=cudas),
            torch.tensor(pairTokenTypeIds,device=cudas))
        ), sentiment

    def create_text_sentence(self, i, max_token_len = -1):
        """
        To be..
        """
        firstSentence = self.items[i][0][0]
        sentiment = self.sentiment_selection(self.items[i][1][0],str(self.num_labels))
        label = 0
        
        firstTokenTypeIds =  np.zeros(len(firstSentence))

        CLS = self.tokenizer.cls_token_id
        SEP = self.tokenizer.sep_token_id
        
        return torch.tensor(firstSentence), torch.tensor(label,dtype=torch.int64,device=cudas),\
        torch.tensor(firstTokenTypeIds,device=cudas),sentiment,self.items[i][-1]

    def count(self):
        """
            Data total length
        """
        return len(self.items)

    def __len__(self):
        """
            Print Data total length when using len() function
        """
        return self.total_item
    
    def __getitem__(self,i):
        text_sentence, text_label, text_token_type_ids, text_sentiment, rawData = self.create_text_sentence(i,max_token_len=75)
        text_sentence2,visual_sentence, tAv_label, tAv_token_type_ids, tAv_sentiment = self.create_concat_joint_sentence(i,'visual',max_token_len = -1)
        text_sentence3,speech_sentence, tAs_label, tAs_token_type_ids, tAs_sentiment = self.create_concat_joint_sentence(i,'speech',max_token_len = -1)

        return text_sentence, text_label, text_token_type_ids, text_sentiment,\
         text_sentence2, visual_sentence, tAv_label, tAv_token_type_ids, tAv_sentiment,\
         text_sentence3, speech_sentence, tAs_label, tAs_token_type_ids, tAs_sentiment,\
         rawData