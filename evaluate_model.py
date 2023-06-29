from aftnet1d import AFT_RACUNet
from Data_Preparation.data_preparation_mrs import Data_Preparation
from utils import evaluate
from torch.utils.data import DataLoader
from main_model import DDPM
from denoising_model_small import ConditionalModel
from Data_Preparation.data_preparation_mrs import Data_Preparation
from utils import evaluate, lse_adjust, compute_metrics
from dnresunet import DnResUNet
import yaml

def evaluate_model(model_type, model_name, af=0, N_channels=2, fid=True, wr=True, cplx=False):

    if model_type == "ddpm":
        path = "config/base.yaml"
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        base_model = ConditionalModel(80, 1).to("cuda:0")
        model = DDPM(base_model, config, "cuda:0")
    elif model_type == "cnn":
        model = DnResUNet('dnresunet_model', 2, 2, "cuda:0", 1)
    elif model_type == "aftnet":
        model = AFT_RACUNet().to("cuda:0")

    data_path = "/media/sail/Elements/JET_CNN/DL-DPM-Denoising/ddpm-mrs-2/data/"

    data_set = Data_Preparation(data_path, acceleration_factor=af, N_channels=N_channels, fid=fid, waterRemoval=wr, cplx=cplx)
    train_set, val_set, test_set = data_set

    test_loader = DataLoader(test_set, batch_size=10, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=10, num_workers=0)