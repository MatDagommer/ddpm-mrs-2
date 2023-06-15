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
    parser.add_argument('--af', type=int, default=10, help='acceleration factor')
    parser.add_argument('--channels', type=int, default=1, help="number of channels. \
                        1: real part only. 2: real + imaginary parts.")
    parser.add_argument('--datapath', default="/media/sail/Elements/JET_CNN/DL-DPM-Denoising/ddpm-mrs-2/data/", \
                        help="data path.")
    parser.add_argument('--epochs', type=int, default=400, help="number of epochs.")
    parser.add_argument('--model', type=str, default="ddpm", help="Model to be used for training. Default: ddpm. Other options: dnresunet")
    args = parser.parse_args()
    print(args)

    if args.model == "cnn":
        args.config = "dnresunet.yaml"
        args.channels = 2
    
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if args.model == "ddpm":
        if config['train']['epochs'] != args.epochs:
            config['train']['epochs'] = args.epochs 
            print("Set the number of epochs to %d."%args.epochs)    

    #foldername = "./check_points/noise_type_" + str(args.n_type) + "/"
    foldername = "./check_points/" + args.name
    if os.path.exists(foldername):
        status = True
        while status:
            answer, timedOut = timedInput("A model named %s already exists. Do you want to erase it (y/n)?"%args.name)
            # print("A model named %s already exists. Do you want to erase it (y/n)?"%args.name)
            # answer = wait_for_input(10)
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


    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    data_path = args.datapath
    
    acceleration_factor = args.af
    train_set, val_set, test_set = Data_Preparation(data_path, acceleration_factor, N_channels=args.channels)
    print("DATASET TYPE: ",type(train_set))
    # [X_train, y_train, X_test, y_test] = Data_Preparation(args.n_type)
    
    # X_train = torch.FloatTensor(X_train)
    # X_train = X_train.permute(0,2,1)
    
    # y_train = torch.FloatTensor(y_train)
    # y_train = y_train.permute(0,2,1)
    
    # X_test = torch.FloatTensor(X_test)
    # X_test = X_test.permute(0,2,1)
    
    # y_test = torch.FloatTensor(y_test)
    # y_test = y_test.permute(0,2,1)
    
    # train_val_set = TensorDataset(y_train, X_train)
    # test_set = TensorDataset(y_test, X_test)
    
    # train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    # train_set = Subset(train_val_set, train_idx)
    # val_set = Subset(train_val_set, val_idx)
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    if args.model == "ddpm":
        #base_model = ConditionalModel(64,8,4).to(args.device)
        base_model = ConditionalModel(config['train']['feats'], args.channels).to(args.device)
        model = DDPM(base_model, config, args.device)
    elif args.model == "cnn":
        model = DnResUNet('dnresunet_model', 2, 2, args.device)
    
    train(model, config['train'], train_loader, args.device, 
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
    
    #eval best (validation)
    print('evaluation (validation set)')
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #eval best
    # print('evaluation (validation set)')
    # foldername = "./check_points/noise_type_" + str(1) + "/"
    # output_path = foldername + "/model.pth"
    # model.load_state_dict(torch.load(output_path))
    # evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #don't use before final model is determined
    print('evaluation (test set)')
    evaluate(model, test_loader, 1, args.device, foldername=foldername)
    
    print("Training completed.")
    
    