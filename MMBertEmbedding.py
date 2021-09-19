import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob,dataset):
        super().__init__()

        if dataset =='mosi':
            self.VISUALDIM = MOSIVISUALDIM
            self.SPEECHDIM = CMUSPEECHDIM
        elif dataset =='mosei':
            self.VISUALDIM = MOSEIVISUALDIM
            self.SPEECHDIM = CMUSPEECHDIM
        elif dataset == 'ur_funny':
            self.VISUALDIM = FUNNYVISUALDIM
            self.SPEECHDIM = FUNNYSPEECHDIM

        self.W_cv = nn.Linear(self.VISUALDIM+TEXTDIM,TEXTDIM)
        self.W_cs = nn.Linear(self.SPEECHDIM+TEXTDIM,TEXTDIM)

        self.Wv = nn.Linear(self.VISUALDIM,TEXTDIM)
        self.Ws = nn.Linear(self.SPEECHDIM,TEXTDIM)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_embs, pair_ids , token_type_ids = None):
        assert input_embs != None, "You miss input_embs"
        assert pair_ids != None, "You miss pair_ids"

        if pair_ids.size()[-1] == self.VISUALDIM:
            pair_embeds = F.relu(self.Wv(pair_ids.float()))
        elif pair_ids.size()[-1] == self.SPEECHDIM:
            pair_embeds = F.relu(self.Ws(pair_ids.float()))
        else:
            raise Exception('Wrong Dimension')

        inputs_embeds = torch.cat((input_embs,pair_embeds),dim=1)
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        
        return embeddings