import argparse
import os

import torchvision
from torchvision.transforms import v2 as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.models.self_supervised import SwAV
import yaml
from easydict import EasyDict

pl.seed_everything(20)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def config(confi_yaml):
    with open(confi_yaml,"r") as stream:
        config = yaml.safe_load(stream)
        
        
        

def main():
    swav = SwAV()