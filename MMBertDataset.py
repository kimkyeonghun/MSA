from torch.utils.data import Dataset
import torch
import random
import os
import numpy as np

from transformers.modeling_bert import BertEmbeddings

cudas = torch.device('cuda')

class MMBertDataset(Dataset):
    def __init__(self,tokenizer,features,config):
        self.tokenizer = tokenizer
        self.items = features
        self.embeddings = BertEmbeddings(config)
        self.total_item = self.count()

    def tokenToEmbedding(input_ids, position_ids=None, token_type_ids=None, inputs_embeds=None):
        embedding_output = self.embeddings(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            inputs_embeds = inputs_embeds
        )
        return embedding_output

        
    def create_concat_joint_sentence(self, i, mode, max_token_len = -1):
        
        pairIndex = -1
        firstIndex = -1
        secondIndex = -1
        edgeCase = False
        label = -1
        
        if mode == 'visual':
            pairIndex = 1
        elif mode == 'speech':
            pairIndex = 2
            
        assert pairIndex != -1
        
        if i == len(self.items)-1:
            firstIndex = i
            secondaIndex = i
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

        CLS = self.tokenizer.cls_token
        SEP = self.tokenizer.sep_token
        tokens = [CLS] + textSentence + [SEP]
        textSentence = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens),dtype=torch.long).unsqueeze(-1)

        embedding_output = self.embeddings(textSentence,token_type_ids=torch.tensor(textTokenTypeIds,dtype=torch.long))

        times = torch.tensor(pairSentence).size()[0]-1
        pairSentence2 = torch.tensor(pairSentence).unsqueeze(0)
        temp = pairSentence2
        while times:
            pairSentence2 = torch.cat((pairSentence2,temp),axis=0)
            times-=1

        return torch.cat((torch.tensor(embedding_output,dtype=torch.float,device=cudas),torch.tensor(pairSentence2,dtype=torch.float,device=cudas)),dim=-1),\
        torch.tensor(label,dtype=torch.int64,device=cudas),torch.cat((
            torch.tensor(textTokenTypeIds,device=cudas),
            torch.tensor(pairTokenTypeIds,device=cudas))
        )
    
    def create_next_sentence_pair(self, i, max_token_len = -1):
        
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

        jointSentences = torch.tensor([CLS]+firstSentence+[SEP]+secondSentence+[SEP],dtype= torch.long).unsqueeze(-1)

        embedding_output = self.embeddings(jointSentences,token_type_ids=torch.tensor(np.concatenate((firstTokenTypeIds,secondTokenTypeIds)),dtype=torch.long))

        return torch.tensor(embedding_output,device=cudas), torch.tensor(label,dtype=torch.int64,device=cudas),\
        torch.cat((torch.tensor(firstTokenTypeIds,device=cudas),torch.tensor(secondTokenTypeIds,device=cudas)))

    def count(self):
        return len(self.items)

    def __len__(self):
        return self.total_item
    
    def __getitem__(self,i):
        text_sentece, text_label, text_token_type_ids = self.create_next_sentence_pair(i,max_token_len = 75)
        tAv_sentence, tAv_label, tAv_token_type_ids = self.create_concat_joint_sentence(i,'visual',max_token_len = -1)
        tAs_sentence, tAs_label, tAs_token_type_ids = self.create_concat_joint_sentence(i,'speech',max_token_len = -1)
        
        return text_sentece, text_label, text_token_type_ids, tAv_sentence, tAv_label, tAv_token_type_ids, tAs_sentence, tAs_label, tAs_token_type_ids