import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)

        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        
    def forward(self, x, y):
        x_pred = self.net(y)

        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)
        
        nce = -(pos - neg).mean()

        return nce

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob, dataset):
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

        self.W_cv = nn.Linear(self.VISUALDIM+TEXTDIM, TEXTDIM)
        self.W_cs = nn.Linear(self.SPEECHDIM+TEXTDIM, TEXTDIM)

        self.Wv = nn.Linear(self.VISUALDIM, TEXTDIM)
        self.Ws = nn.Linear(self.SPEECHDIM, TEXTDIM)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_embs, pair_ids):
        assert input_embs != None, "You miss input_embs"
        assert pair_ids != None, "You miss pair_ids"

        if pair_ids.size()[-1] == self.VISUALDIM:
            pair_embeds = F.relu(self.Wv(pair_ids.float()))
        elif pair_ids.size()[-1] == self.SPEECHDIM:
            pair_embeds = F.relu(self.Ws(pair_ids.float()))
        else:
            raise Exception('Wrong Dimension')

        inputs_embeds = torch.cat((input_embs, pair_embeds),dim=1)
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        
        return embeddings