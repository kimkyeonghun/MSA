import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob,dataset):
        super().__init__()

        if dataset =='mosi':
            self.VISUALDIM = MOSIVISUALDIM
        elif dataset =='mosei':
            self.VISUALDIM = MOSEIVISUALDIM


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
        
        if pair_ids.size()[-1] == self.VISUALDIM:
            visualW = F.relu(self.Wv(pair_ids.float()))
            embeddings = visualW*self.W_cv(concat_embs)
        elif pair_ids.size()[-1] == SPEECHDIM:
            speechW = F.relu(self.Ws(pair_ids.float()))
            embeddings = speechW*self.W_cs(concat_embs)
        else:
            raise Exception('Wrong Dimension')

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class FuseGate(nn.Module):
    def __init__(self,hidden_size,dropout_prob,dataset):
        super().__init__()

        if dataset =='mosi':
            VISUALDIM = MOSIVISUALDIM
        elif dataset =='mosei':
            VISUALDIM = MOSEIVISUALDIM

        self.Wv = nn.Linear(1+VISUALDIM,1)
        self.Ws = nn.Linear(1+SPEECHDIM,1)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,text_sentence,pair_sentence,mode):
        if mode == 'visual':
            outputs = F.relu(self.Wv(torch.cat((text_sentence,pair_sentence),dim=1)))
        elif mode == 'speech':
            outputs = F.relu(self.Ws(torch.cat((text_sentence,pair_sentence),dim=1)))

        return outputs