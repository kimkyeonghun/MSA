import argparse
import os
import torch

DEVICE = torch.device("cuda")

total_vocab_size = 30522

#modlity dimension
#TEXTDIM = 768

TEXTDIM = 1024
MOSEIVISUALDIM = 35
MOSIVISUALDIM = 47
FUNNYVISUALDIM = 371
CMUSPEECHDIM = 74
FUNNYSPEECHDIM = 81
