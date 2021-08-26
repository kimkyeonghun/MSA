import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob,dataset):
        super().__init__()

        if dataset =='mosi':
            self.VISUALDIM = MOSIVISUALDIM
        elif dataset =='mosei' or dataset == 'iemocap':
            self.VISUALDIM = MOSEIVISUALDIM
        elif dataset == 'meld':
            self.VISUALDIM = MELDVISUALDIM

        if dataset == 'iemocap':
            TEXTDIM = 300
        else:
            TEXTDIM = 768

        self.W_cv = nn.Linear(self.VISUALDIM+TEXTDIM,TEXTDIM)
        self.W_cs = nn.Linear(SPEECHDIM+TEXTDIM,TEXTDIM)

        self.Wv = nn.Linear(self.VISUALDIM,TEXTDIM)
        self.Ws = nn.Linear(SPEECHDIM,TEXTDIM)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_embs, pair_ids , token_type_ids = None):
        assert input_embs != None, "You miss input_embs"
        assert pair_ids != None, "You miss pair_ids"
        concat_embs = torch.cat((input_embs,pair_ids.float()),dim=-1)
        # #print("concat_embs",concat_embs.shape)
        # #print(pair_ids.size())
        # if pair_ids.size()[-1] == self.VISUALDIM:
        #     visualW = F.relu(self.Wv(pair_ids.float()))
        #     #print("visualW",visualW.shape)
        #     embeddings = visualW*self.W_cv(concat_embs)
        # elif pair_ids.size()[-1] == SPEECHDIM:
        #     speechW = F.relu(self.Ws(pair_ids.float()))
        #     embeddings = speechW*self.W_cs(concat_embs)
        # else:
        #     raise Exception('Wrong Dimension')
        # #print("embeddings",embeddings.shape)
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)

                #print("concat_embs",concat_embs.shape)
        #print(pair_ids.size())
        if pair_ids.size()[-1] == self.VISUALDIM:
            pair_embeds = F.relu(self.Wv(pair_ids.float()))
        elif pair_ids.size()[-1] == SPEECHDIM:
            pair_embeds = F.relu(self.Ws(pair_ids.float()))
        else:
            raise Exception('Wrong Dimension')
        #print("embeddings",embeddings.shape)
        inputs_embeds = torch.cat((input_embs,pair_embeds),dim=1)
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class IEMOCAPEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(300,768)

    def forward(self, input_id):
        return self.W(input_id.float())


class MELDEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()  

    def forward(self, input_ids, position_ids, token_type_ids, inputs_embeds, embeddings):
        input_ids = input_ids[:,:,0]
        print(input_ids.shape)
        print(token_type_ids.shape)
        return embeddings(input_ids=input_ids.long(), position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        embedding_outputs = embeddings(input_ids=input_ids[:,0,:].long(), position_ids=position_ids, token_type_ids=None, inputs_embeds=inputs_embeds).unsqueeze(1)
        print(embedding_outputs.shape)
        for i in range(1,33):
            embedding_output = embeddings(input_ids=input_ids[:,i,:].long(), position_ids=position_ids, token_type_ids=None, inputs_embeds=inputs_embeds)
            embedding_outputs = torch.cat((embedding_outputs,embedding_output.unsqueeze(1)),dim=1)
        return embedding_outputs
