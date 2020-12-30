import os
import random

from config import VISUALDIM, SPEECHDIM, DEVICE

import torch
from torch.utils.data import Dataset
import numpy as np

from transformers.modeling_bert import BertEmbeddings

cudas = DEVICE

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

    def __init__(self,tokenizer,features):
        self.tokenizer = tokenizer
        self.items = features
        self.total_item = self.count()
        
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
        label = -1
        
        if mode == 'visual':
            pairIndex = 1
            pairDim = VISUALDIM
        elif mode == 'speech':
            pairIndex = 2
            pairDim = SPEECHDIM
            
        assert pairIndex != -1
        
        if i == len(self.items)-1:
            firstIndex = i
            secondIndex = i
            label = 0
            edgeCase = True
        
        if not edgeCase:
            r = random.uniform(0,1)
            
            if r > 0.5:
                firstIndex = i
                secondIndex = i
                label = 0
            else:
                firstIndex = i
                secondIndex = random.choice(range(len(self.items)))
                while firstIndex == secondIndex:
                    secondIndex = random.choice(range(len(self.items)))
                label = 1
        
        textSentence = self.items[firstIndex][0][0]
        pairSentence = self.items[secondIndex][0][pairIndex]
        
        if max_token_len > 0:
            if (len(textSentence) + len(pairSentence)) > max_token_len:
                num_tokens_to_remove = (len(textSentence)+len(pairSentence)) - max_token_len
                first, second, _ = self.tokenizer.truncate_sequences(
                    ids = textSentence,
                    pair_ids = pairSentence,
                    num_tokens_to_remove = num_tokens_to_remove
                )
                textSentence = first
                pairSentence = second
        
        textTokenTypeIds = np.zeros(len(textSentence) + 2)
        pairTokenTypeIds = np.ones(len(pairSentence) + 1)

        CLS = torch.ones((1,pairDim),dtype=torch.float)*self.tokenizer.cls_token_id
        SEP = self.tokenizer.sep_token_id

        #textSentence = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens),dtype=torch.long).unsqueeze(-1)

        #embedding_output = self.embeddings(textSentence,token_type_ids=torch.tensor(textTokenTypeIds,dtype=torch.long))

        textSentence = torch.tensor(textSentence,dtype=torch.float).unsqueeze(-1)
        temp = textSentence
        while pairDim-1:
            textSentence = torch.cat((textSentence,temp),axis=1)
            pairDim-=1

        jointSentence = torch.cat((CLS,textSentence,torch.tensor(pairSentence,dtype=torch.float)))

        return jointSentence, torch.tensor(label,dtype=torch.int64,device=cudas),torch.cat((
            torch.tensor(textTokenTypeIds,device=cudas),
            torch.tensor(pairTokenTypeIds,device=cudas))
        )
    
    def create_next_sentence_pair(self, i, max_token_len = -1):
        """
        If last sentencse, make label =1
        Choose random float between 0 and 1,
            if random number is over 0.5, make next sentence pair(label = 0)
            else, make next sentence unpair(label=1) using random chooice that do not allow duplicate select (label = 1)

        jointSentence : Tuple([List])
            ([CLS],first_sentence,[SEP],second_sentence,[SEP])

        label : torch.tensor
            (1 or 0)

        token_type_ids : torch.tensor
            (First_sentence_token_type_ids(0) + Second_sentence_token_type_ids(1))

        return jointSentence, label, token_type_ids
        """
        firstSentence = None
        secondSentence = None
        
        if i == len(self.items)-1:
            firstSentence = self.items[i][0][0]
            secondSentence = self.items[i][0][0]
            label = 1
        else:
            firstSentence = self.items[i][0][0]
            r = random.uniform(0,1)
            
            if r > 0.5:
                nextIdx = i+1
                label = 0
            else:
                nextIdx = random.choice(range(len(self.items)))
                while i == nextIdx:
                    nextIdx = random.choice(range(len(self.items)))
                label = 1
            secondSentence = self.items[nextIdx][0][0]

        if max_token_len > 0:
            if (len(firstSentence) + len(secondSentence)) > max_token_len:
                num_tokens_to_remove = (len(firstSentence)+len(secondSentence)) - max_token_len
                first, second, _ = self.tokenizer.truncate_sequences(
                    ids = firstSentence,
                    pair_ids = secondSentence,
                    num_tokens_to_remove = num_tokens_to_remove
                )
                firstSentence = first
                secondSentence = second
        
        firstTokenTypeIds =  np.zeros(len(firstSentence)+2)
        secondTokenTypeIds =  np.ones(len(secondSentence)+1)

        CLS = self.tokenizer.cls_token_id
        SEP = self.tokenizer.sep_token_id

        #jointSentences = torch.tensor([CLS]+firstSentence+[SEP]+secondSentence+[SEP],dtype= torch.long).unsqueeze(-1)
        jointSentences = np.concatenate(([CLS],firstSentence,[SEP],secondSentence,[SEP]))

        #embedding_output = self.embeddings(jointSentences,token_type_ids=torch.tensor(np.concatenate((firstTokenTypeIds,secondTokenTypeIds)),dtype=torch.long))

        return torch.tensor(jointSentences), torch.tensor(label,dtype=torch.int64,device=cudas),\
        torch.cat((torch.tensor(firstTokenTypeIds,device=cudas),torch.tensor(secondTokenTypeIds,device=cudas)))

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
        text_sentence, text_label, text_token_type_ids = self.create_next_sentence_pair(i,max_token_len = 75)
        tAv_sentence, tAv_label, tAv_token_type_ids = self.create_concat_joint_sentence(i,'visual',max_token_len = -1)
        tAs_sentence, tAs_label, tAs_token_type_ids = self.create_concat_joint_sentence(i,'speech',max_token_len = -1)
        
        return text_sentence, text_label, text_token_type_ids, tAv_sentence, tAv_label, tAv_token_type_ids, tAs_sentence, tAs_label, tAs_token_type_ids