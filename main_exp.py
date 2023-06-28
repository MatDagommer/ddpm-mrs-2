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
from aftnet1d import AFT_RACUNet
from sys import platform


def wait_for_input(timeout):
    # Create a selector object
    sel = selectors.DefaultSelector()
    # Register the standard input file object for read events
    sel.register(sys.stdin, selectors.EVENT_READ)
    print("EVENT_READ: ", selectors.EVENT_READ)
    time.sleep(5)
    # Wait for user input or timeout
    events = sel.select(timeout=timeout)


    if events:
        # User input received
        for key, _ in events:
            if key.fileobj is sys.stdin:
                user_input = sys.stdin.readline().strip()
                return user_input
    else:
        # Timeout reached, no user input
        return "y"
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="DDPM for ECG")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    # parser.add_argument('--n_type', type=int, default=1, help='noise version')
    parser.add_argument('--name', default='test', help='model name.')
    parser.add_argument('--af', type=int, default=10, help='acceleration factor. \
                        If 0, takes random samples with continuous af between 8 and 32.')
    parser.add_argument('--channels', type=int, default=1, help="number of channels. \
                        1: real part only. 2: real + imaginary parts.")
    parser.add_argument('--datapath', default="/media/sail/Elements/JET_CNN/DL-DPM-Denoising/ddpm-mrs-2/data/", \
                        help="data path.")
    parser.add_argument('--epochs', type=int, default=400, help="number of epochs.")
    parser.add_argument('--model', type=str, default="ddpm", help="Model to be used for training. \
                        Default: ddpm. Other options: cnn, aftnet")
    parser.add_argument('--fid', action='store_true', default=False, help='Use FID as an input.')
    parser.add_argument('--dilation', type=int, default=1, help="dilation to use for convolutions \
                        (CNN).")
    parser.add_argument('--wr', action='store_true', default=False, help="water peak removal.")
    args = parser.parse_args()
    print(args)

    if args.model == "cnn":
        args.config = "dnresunet.yaml"
        args.channels = 2

    cplx = False
    if args.model == "aftnet":
        args.config = "aftnet.yaml"
        args.fid = True # Always using FID as input with AFT-Net
        cplx = True
    
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if args.model == "ddpm":
        if config['train']['epochs'] != args.epochs:
            config['train']['epochs'] = args.epochs 
            print("Number of epochs: %d."%args.epochs)    
        
    if args.fid == True: # when in=FID and out=Spectra ==> need both real/imag parts
        args.channels = 2

    if platform == "win32":
        args.datapath = "C:/Users/matth/Documents/SAIL/data/"

    foldername = "./check_points/" + args.name
    if os.path.exists(foldername):
        status = True
        while status:
            answer, timedOut = timedInput("A model named %s already exists. Do you want to erase it (y/n)?"%args.name)
            # print("A model named %s already exists. Do you want to erase it (y/n)?"%args.name)
            if(timedOut):
                print("Timed out when waiting for input. Erasing model.")
                answer = "y"
            else:
                print(f"User-input: '{answer}'")

            if answer == "y":
                status = False
            elif answer == "n":
                new_name = input("Choose a new name: ")
                args.name = new_name
                foldername = "./check_points/" + args.name
                if not os.path.exists(foldername):
                    status = False


    print('Model saved in:', foldername)
    os.makedirs(foldername, exist_ok=True)
    data_path = args.datapath

    # print("batch size (training): ", config['train']['batch_size'])
    
    acceleration_factor = args.af
    train_set, val_set, test_set = Data_Preparation(data_path, acceleration_factor, \
                    N_channels=args.channels, fid=args.fid, waterRemoval=args.wr, cplx=cplx)
    
    # print("DATASET TYPE: ",type(train_set))
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config['train']['batch_size'], num_workers=0)
    
    if args.model == "ddpm":
        base_model = ConditionalModel(config['train']['feats'], args.channels).to(args.device)
        model = DDPM(base_model, config, args.device)
    elif args.model == "cnn":
        model = DnResUNet('dnresunet_model', 2, 2, args.device, args.dilation)
    elif args.model == "aftnet":
        model = AFT_RACUNet().to(args.device)
    
    train(model, config['train'], train_loader, args.device, 
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
    

    # retrieving best model
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))

    # evaluation at best epoch
    # validation
    print('evaluation of model at best epoch (validation set)')
    evaluate(model, val_loader, 1, args.device, foldername=foldername, fid=args.fid, filename="val_metrics")
    
    #test
    print('evaluation of model at best epoch (test set)')
    evaluate(model, test_loader, 1, args.device, foldername=foldername, fid=args.fid, filename="test_metrics")
    
    print("Training completed.")
    
    