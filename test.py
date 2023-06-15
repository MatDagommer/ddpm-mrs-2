import argparse
import torch
import datetime
import json
import yaml
import os
import time
import sys
import selectors
from pytimedinput import timedInput
# from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_mrs import Data_Preparation
from main_model import DDPM
from denoising_model_small import ConditionalModel
from utils import train, evaluate
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from sklearn.model_selection import train_test_split
from dnresunet import DnResUNet

path = "config/" + "base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

foldername = "./check_points/test"
data_path = "/media/sail/Elements/JET_CNN/DL-DPM-Denoising/ddpm-mrs-2/data/"

output_path = foldername + "/model.pth"
base_model = ConditionalModel(80, 1).to("cuda:0")
model = DDPM(base_model, config, "cuda:0")
model.load_state_dict(torch.load(output_path))

train_set, val_set, test_set = Data_Preparation(data_path, 10, N_channels=1)
val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], num_workers=0)
test_loader = DataLoader(test_set, batch_size=config['train']['batch_size'], num_workers=0)

# evaluation at best epoch
# validation
print('evaluation of model at best epoch (validation set)')
evaluate(model, val_loader, 1, "cuda:0", foldername=foldername, filename="val_metrics")

#test
print('evaluation of model at best epoch (test set)')
evaluate(model, test_loader, 1, "cuda:0", foldername=foldername, filename="test_metrics")