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

def Data_Preparation(data_path, acceleration_factor, N_channels=2):

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

    SpectraOFF = np.expand_dims(np.divide(SpectraOFF, repeat_), axis=-1)
    SpectraOFF_avg = np.expand_dims(np.mean(SpectraOFF, axis=1), axis=-1) #[length x #subjects]
    SpectraOFF = SpectraOFF[1000:1512]
    SpectraOFF_avg = SpectraOFF_avg[1000:1512]


    # IMPLEMENT SECOND CHANNEL HERE
    if N_channels == 1:
        SpectraOFF = np.real(SpectraOFF)
        SpectraOFF_avg = np.real(SpectraOFF_avg)
    elif N_channels == 2:  
        SpectraOFF = np.concatenate((np.real(SpectraOFF), np.imag(SpectraOFF)), axis=-1)
        SpectraOFF_avg = np.concatenate((np.real(SpectraOFF_avg), np.imag(SpectraOFF_avg)), axis=-1)

    # OUTPUT SHAPES:
    # SpectraOFF: [length, N_acq, N_subj, N_channels]
    # SpectraOFF_avg: [length, N_subj, N_channels]

    print("SpectraOFF shape: ", SpectraOFF.shape)
    print("SpectraOFF_avg shape: ", SpectraOFF_avg.shape)
    

    _, _, N_subjects, _ = SpectraOFF.shape

    print("Separating train / val / test sets...")
    train_idx, val_idx = train_test_split(range(N_subjects), test_size=0.4)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)

    noisy_batch_train = SpectraOFF[:, :, train_idx]
    clean_batch_train = SpectraOFF_avg[:, train_idx]
    print("noisy_batch_train shape: ", noisy_batch_train.shape)
    print("clean_batch_train shape: ", clean_batch_train.shape)

    print("Starting validation dataset generation...")
    # OUTPUT [#samples x length x N_channels]
    clean_batch_val, noisy_batch_val = retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, val_idx, acceleration_factor)
    print("Starting validation dataset generation...")
    clean_batch_test, noisy_batch_test  = retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, test_idx, acceleration_factor)

    print("Converting data to tensors...")

    noisy_batch_train = torch.FloatTensor(noisy_batch_train)
    clean_batch_train = torch.FloatTensor(clean_batch_train)

    clean_batch_val = torch.FloatTensor(clean_batch_val)
    clean_batch_val = clean_batch_val.permute(0, 2, 1)
    noisy_batch_val = torch.FloatTensor(noisy_batch_val)
    noisy_batch_val = noisy_batch_val.permute(0, 2, 1)

    clean_batch_test = torch.FloatTensor(clean_batch_test)
    clean_batch_test = clean_batch_test.permute(0, 2, 1)
    noisy_batch_test = torch.FloatTensor(noisy_batch_test)
    noisy_batch_test = noisy_batch_test.permute(0, 2, 1)

    train_set = DynamicDataset(clean_batch_train, noisy_batch_train, acceleration_factor)
    val_set = TensorDataset(clean_batch_val, noisy_batch_val)
    test_set = TensorDataset(clean_batch_test, noisy_batch_test)

    torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    print("Done.")
    print("Dataset ready.")

    return train_set, val_set, test_set


class DynamicDataset(Dataset):
    
    def __init__(self, clean_batch_train, noisy_batch_train, acceleration_factor, N_samples_per_subject=100):

        self.clean_batch_train = clean_batch_train
        self.noisy_batch_train = noisy_batch_train
        self.patch_size, self.N_acq, self.N_subjects = self.noisy_batch_train.shape
        self.acceleration_factor = acceleration_factor
        self.N_samples_per_subject = N_samples_per_subject
        
    def __len__(self):

        return self.N_subjects * self.N_samples_per_subject

    def __getitem__(self, index):

        subject_idx = np.random.randint(0, self.N_subjects)
        sample_ids = np.random.randint(0, self.N_acq, self.N_acq // self.acceleration_factor)
        np.random.shuffle(sample_ids)
        noisy_batch = torch.mean(self.noisy_batch_train[:, sample_ids, subject_idx], axis=1)
        clean_batch = self.clean_batch_train[:, subject_idx]
        clean_batch = clean_batch.view(1, self.patch_size)
        noisy_batch = noisy_batch.view(1, self.patch_size)

        return clean_batch, noisy_batch


def retrieve_val_test_set(SpectraOFF, SpectraOFF_avg, idx, acceleration_factor, N_samples_per_subject=100):
    
    patch_size, N_acq, _, N_channels = SpectraOFF.shape
    N_subjects = len(idx)

    clean_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))
    noisy_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))

    for i in tqdm(range(N_subjects)):
        for j in range(N_samples_per_subject):
            sample_idx = np.random.randint(0, N_acq, N_acq // acceleration_factor)
            np.random.shuffle(sample_idx)
            noisy_batch[i, j, :, :] = np.mean(SpectraOFF[:, sample_idx, i, :], axis=1)
            clean_batch[i, j, :, :] = SpectraOFF_avg[:, i, :]

    noisy_batch = noisy_batch.reshape(-1, patch_size, N_channels)
    clean_batch = clean_batch.reshape(-1, patch_size, N_channels)

    return clean_batch, noisy_batch