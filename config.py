import os
import torch

DEVICE = torch.device("cuda")

total_vocab_size = 30522

#modlity dimension
TEXTDIM = 768
VISUALDIM = 35
SPEECHDIM = 74