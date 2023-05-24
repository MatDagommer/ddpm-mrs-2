import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import savemat, loadmat
from scipy.fft import fft, fftshift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset, Dataset


# Data Preparation
# Prepare train / val / test sets (subject separation)

def Data_Preparation(data_path, acceleration_factor=10, N_channels=1):

    np.random.seed(1234)

    print("Loading raw data...")
    PH_invivoData = loadmat(os.path.join(data_path, "PH_InVivoData.mat"))
    FidsOFF = PH_invivoData['OFFdata']

    SpectraOFF = np.zeros_like(FidsOFF)
 
    for i in range(SpectraOFF.shape[1]):
        for j in range(SpectraOFF.shape[2]):
            SpectraOFF[:, i, j] = fftshift(fft(FidsOFF[:, i, j]))

    SpectraOFF_amp = np.abs(SpectraOFF)

    # Retrieve a maximum amplitude for each subject
    MAX_VAL = np.max(SpectraOFF_amp, axis=1)
    MAX_VAL = np.max(MAX_VAL, axis=0)

    repeat_ = np.repeat(np.expand_dims(MAX_VAL, axis=-1), SpectraOFF.shape[0], axis=-1)
    repeat_ = np.repeat(np.expand_dims(repeat_, axis=-1), SpectraOFF.shape[1], axis=-1)
    repeat_ = np.transpose(repeat_, (1, 2, 0))

    SpectraOFF = np.divide(SpectraOFF, repeat_)
    SpectraOFF_avg = np.mean(SpectraOFF, axis=1) #[length x #subjects]
    SpectraOFF = SpectraOFF[1000:1512]
    SpectraOFF_avg = SpectraOFF_avg[1000:1512]

    # IMPLEMENT SECOND CHANNEL HERE
    SpectraOFF = np.real(SpectraOFF)
    SpectraOFF_avg = np.real(SpectraOFF_avg)

    print("SpectraOFF.shape: ", SpectraOFF.shape)
    print("SpectraOFF_avg.shape: ", SpectraOFF_avg.shape)

    _, _, N_subjects = SpectraOFF.shape
    acceleration_factor = 10

    print("Separating train / val / test sets...")
    train_idx, val_idx = train_test_split(range(N_subjects), test_size=0.4)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)

    X_train = SpectraOFF[:, :, train_idx]
    y_train = SpectraOFF_avg[:, train_idx]
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    print("Starting validation dataset generation...")
    # [#samples x length x N_channels]
    X_val, y_val = retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, val_idx, N_channels, acceleration_factor)
    print("Starting validation dataset generation...")
    X_test, y_test = retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, test_idx, N_channels, acceleration_factor)

    print("Converting data to tensors...")

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0, 2, 1)
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0, 2, 1)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0, 2, 1)
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0, 2, 1)

    train_set = DynamicDataset(X_train, y_train, acceleration_factor)
    val_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    print("Done.")
    print("Dataset ready.")

    return train_set, val_set, test_set


class DynamicDataset(Dataset):
    
    def __init__(self, X_train, y_train, acceleration_factor, N_samples_per_subject=100):

        self.X_train = X_train
        self.y_train = y_train
        self.patch_size, self.N_acq, self.N_subjects = self.X_train.shape
        self.acceleration_factor = acceleration_factor
        self.N_samples_per_subject = N_samples_per_subject
        
    def __len__(self):

        return self.N_subjects * self.N_samples_per_subject

    def __getitem__(self, index):

        subject_idx = np.random.randint(0, self.N_subjects)
        sample_ids = np.random.randint(0, self.N_acq, self.N_acq // self.acceleration_factor)
        np.random.shuffle(sample_ids)
        noisy_batch = torch.mean(self.X_train[:, sample_ids, subject_idx], axis=1)
        clean_batch = self.y_train[:, subject_idx]
        clean_batch = clean_batch.view(-1, 1, self.patch_size)
        noisy_batch = noisy_batch.view(-1, 1, self.patch_size)

        return clean_batch, noisy_batch


def retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, idx, N_channels, acceleration_factor, N_samples_per_subject=100):
    
    patch_size, N_acq, _ = SpectraOFF.shape
    N_subjects = len(idx)

    clean_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))
    noisy_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))

    for i in tqdm(range(N_subjects)):
        for j in range(N_samples_per_subject):
            sample_idx = np.random.randint(0, N_acq, N_acq // acceleration_factor)
            np.random.shuffle(sample_idx)
            noisy_batch[i, j, :, 0] = np.mean(SpectraOFF[:, sample_idx, i], axis=1)
            clean_batch[i, j, :, 0] = SpectraOFF_avg[:, i]

    X = noisy_batch.reshape(-1, patch_size, N_channels)
    y = noisy_batch.reshape(-1, patch_size, N_channels)

    return X, y