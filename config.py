import os
import torch

DEVICE = torch.device("cuda")

total_vocab_size = 30522

#modlity dimension
TEXTDIM = 768
MOSEIVISUALDIM = 35
MOSIVISUALDIM = 47
SPEECHDIM = 74