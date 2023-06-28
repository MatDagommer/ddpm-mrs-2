from aftnet1d import AFT_RACUNet
from Data_Preparation.data_preparation_mrs import Data_Preparation
from utils import evaluate
from torch.utils.data import DataLoader

def add_metrics(model_name):
    data_path = "/media/sail/Elements/JET_CNN/DL-DPM-Denoising/ddpm-mrs-2/data/"

    data_set = Data_Preparation(data_path, acceleration_factor=0, N_channels=2, fid=True, waterRemoval=True, cplx=True)
    train_set, val_set, test_set = data_set

    test_loader = DataLoader(test_set, batch_size=10, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=10, num_workers=0)

    model = AFT_RACUNet().to("cuda:0")

    print('evaluation of model at best epoch (validation set)')
    evaluate(model, val_loader, 1, "cuda:0", \
             foldername="./check_points/" + model_name + "/", fid=True, filename="val_metrics")

    #test
    print('evaluation of model at best epoch (test set)')
    evaluate(model, test_loader, 1, "cuda:0", \
             foldername="./check_points/" + model_name + "/", fid=True, filename="test_metrics")