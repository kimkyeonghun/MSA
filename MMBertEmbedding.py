import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        self.Wv = nn.Linear(VISUALDIM,TEXTDIM)
        self.Ws = nn.Linear(SPEECHDIM,TEXTDIM)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids , token_type_ids = None, inputs_embeds = None):
        assert input_ids != None, "You miss input_ids"

        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device = DEVICE)
        
        if inputs_embeds is None:
            if input_shape[-1] == VISUALDIM:
                inputs_embeds = F.relu(self.Wv(input_ids))
            elif input_shape[-1] == SPEECHDIM:
                inputs_embeds = F.relu(self.Ws(input_ids))
            else:
                raise Exception('Wrong Dimension')
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class FuseGate(nn.Module):
    def __init__(self,hidden_size,dropout_prob):
        super().__init__()

        self.Wv = nn.Linear(VISUALDIM,1)
        self.Ws = nn.Linear(SPEECHDIM,1)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,jointSentence,mode):
        if mode == 'visual':
            outputs = torch.cat((jointSentence[0],F.relu(self.Wv(jointSentence[1]))))
        elif mode == 'speech':
            outputs = torch.cat((jointSentence[0],F.relu(self.Ws(jointSentence[1]))))

        return outputs