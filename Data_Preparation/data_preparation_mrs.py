import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from scipy.io import savemat, loadmat
from scipy.fft import fft, ifft, fftshift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset, Dataset


# Data Preparation
# Prepare train / val / test sets (subject separation)

def Data_Preparation(data_path, acceleration_factor, N_channels=1, \
                     fid=False, cplx=False):

    np.random.seed(1234)

    print("Loading raw data...")
    PH_invivoData = loadmat(os.path.join(data_path, "PH_InVivoData.mat"))
    FidsOFF = PH_invivoData['OFFdata']
    FidsON = PH_invivoData['ONdata']

    SpectraOFF = np.zeros_like(FidsOFF)
    SpectraON = np.zeros_like(FidsON)

    for i in range(SpectraOFF.shape[1]):
        for j in range(SpectraOFF.shape[2]):
            SpectraOFF[:, i, j] = fftshift(fft(FidsOFF[:, i, j]))
            SpectraON[:, i, j] = fftshift(fft(FidsON[:, i, j]))


    # Water Peak Removal
    
    # if waterRemoval:
    #     print("Removing water peaks...")
    #     SpectraOFF[975:1075] = SpectraOFF[975:1075] - SpectraON[975:1075]

    # Normalizing Spectra + FIDs

    print("Normalizing data...")

    # if not fid:
    SpectraOFF_amp = np.abs(SpectraOFF)
    MAX_VAL = np.max(SpectraOFF_amp, axis=1)
    MAX_VAL = np.max(MAX_VAL, axis=0)

    # else:
    #     FidsOFF_amp = np.abs(FidsOFF)
    #     MAX_VAL = np.max(FidsOFF_amp, axis=1)
    #     MAX_VAL = np.max(MAX_VAL, axis=0)

    repeat_ = np.repeat(np.expand_dims(MAX_VAL, axis=-1), SpectraOFF.shape[0], axis=-1)
    repeat_ = np.repeat(np.expand_dims(repeat_, axis=-1), SpectraOFF.shape[1], axis=-1)
    repeat_ = np.transpose(repeat_, (1, 2, 0))

    SpectraOFF = np.divide(SpectraOFF, repeat_)
            
    SpectraOFF_avg = np.mean(SpectraOFF, axis=1) #[length x #subjects]

    if fid: # recomputing FID with normalized FFT (& optional Water Removal)
        for i in range(SpectraOFF.shape[1]):
                for j in range(SpectraOFF.shape[2]):
                    FidsOFF[:, i, j] = ifft(fftshift(SpectraOFF[:, i, j]))
        Input = FidsOFF
    else:
        Input = SpectraOFF

    Input = np.expand_dims(Input, axis=-1)
    SpectraOFF_avg = np.expand_dims(SpectraOFF_avg, axis=-1)

    # real part if channel==1, else (real,imag)
    print("Adjusting channels...")

    if N_channels == 1:
        Input = np.real(Input)
        SpectraOFF_avg = np.real(SpectraOFF_avg)
    elif N_channels == 2:  
        Input = np.concatenate((np.real(Input), np.imag(Input)), axis=-1)
        SpectraOFF_avg = np.concatenate((np.real(SpectraOFF_avg), np.imag(SpectraOFF_avg)), axis=-1)

    # OUTPUT SHAPES:
    # Input: [length, N_acq, N_subj, N_channels]
    # SpectraOFF_avg: [length, N_subj, N_channels]

    # print("Input shape: ", Input.shape)
    # print("SpectraOFF_avg shape: ", SpectraOFF_avg.shape)

    _, _, N_subjects, _ = Input.shape

    print("Separating train / val / test sets...")
    train_idx, val_idx = train_test_split(range(N_subjects), test_size=0.2)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)

    noisy_batch_train = Input[:, :, train_idx]
    clean_batch_train = SpectraOFF_avg[:, train_idx]

    # print("noisy_batch_train shape: ", noisy_batch_train.shape)
    # print("clean_batch_train shape: ", clean_batch_train.shape)

    print("Starting validation dataset generation...")
    # OUTPUT [#samples x length x N_channels]
    clean_batch_val, noisy_batch_val, _, _ = retrieve_val_test_set(Input, SpectraOFF_avg, val_idx, acceleration_factor)
    print("Starting test dataset generation...")
    clean_batch_test, noisy_batch_test, nb_samples_test, samples_ids_test  = retrieve_val_test_set(Input, SpectraOFF_avg, test_idx, acceleration_factor)


    with open(os.path.join(data_path, "test_ids"), "wb") as file:
        pickle.dump(test_idx, file)

    with open(os.path.join(data_path, "nb_samples_test"), "wb") as file:
        pickle.dump(nb_samples_test, file)

    with open(os.path.join(data_path, "samples_ids_test"), "wb") as file:
        pickle.dump(samples_ids_test, file)

    print("Converting data to tensors...")

    noisy_batch_train = torch.FloatTensor(noisy_batch_train)
    clean_batch_train = torch.FloatTensor(clean_batch_train)
    noisy_batch_val = torch.FloatTensor(noisy_batch_val)
    clean_batch_val = torch.FloatTensor(clean_batch_val)
    noisy_batch_test = torch.FloatTensor(noisy_batch_test)
    clean_batch_test = torch.FloatTensor(clean_batch_test)
    
    if cplx:
        print("Setting data as complex...")
        noisy_batch_train = torch.view_as_complex(noisy_batch_train)
        noisy_batch_train = torch.unsqueeze(noisy_batch_train, dim=-1)
        clean_batch_train = torch.view_as_complex(clean_batch_train)
        clean_batch_train = torch.unsqueeze(clean_batch_train, dim=-1)

        noisy_batch_val = torch.view_as_complex(noisy_batch_val)
        noisy_batch_val = torch.unsqueeze(noisy_batch_val, dim=-1)
        clean_batch_val = torch.view_as_complex(clean_batch_val)
        clean_batch_val = torch.unsqueeze(clean_batch_val, dim=-1)

        noisy_batch_test = torch.view_as_complex(noisy_batch_test)
        noisy_batch_test = torch.unsqueeze(noisy_batch_test, dim=-1)
        clean_batch_test = torch.view_as_complex(clean_batch_test)
        clean_batch_test = torch.unsqueeze(clean_batch_test, dim=-1)

    clean_batch_val = clean_batch_val.permute(0, 2, 1)
    noisy_batch_val = noisy_batch_val.permute(0, 2, 1)
    clean_batch_test = clean_batch_test.permute(0, 2, 1)
    noisy_batch_test = noisy_batch_test.permute(0, 2, 1)


    train_set = DynamicDataset(clean_batch_train, noisy_batch_train, acceleration_factor)
    val_set = TensorDataset(clean_batch_val, noisy_batch_val)
    test_set = TensorDataset(clean_batch_test, noisy_batch_test)

    print("Saving datasets...")

    torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    print("Done.")
    print("Datasets ready.")

    return train_set, val_set, test_set


class DynamicDataset(Dataset):
    
    def __init__(self, clean_batch_train, noisy_batch_train, acceleration_factor, N_samples_per_subject=100):

        self.clean_batch_train = clean_batch_train
        self.noisy_batch_train = noisy_batch_train
        self.patch_size, self.N_acq, self.N_subjects, self.N_channels = self.noisy_batch_train.shape
        self.acceleration_factor = acceleration_factor
        self.N_samples_per_subject = N_samples_per_subject

        # AF = 0 ONLY
        self.lb_R = 8 # lower bound R
        self.ub_R = 80 # upper bound R
        self.lb_nb_samples = self.N_acq // self.ub_R # lower bound number of samples
        self.ub_nb_samples = self.N_acq // self.lb_R # upper bound number of samples
        
    def __len__(self):

        return self.N_subjects * self.N_samples_per_subject

    def __getitem__(self, index):

        subject_idx = np.random.randint(0, self.N_subjects)
        if self.acceleration_factor == 0:
            sample_ids = np.random.randint(0, self.N_acq, \
                                           np.random.random_integers(self.lb_nb_samples, \
                                                                     self.ub_nb_samples))
        else:
            sample_ids = np.random.randint(0, self.N_acq, self.N_acq // self.acceleration_factor)
        np.random.shuffle(sample_ids)
        noisy_batch = torch.mean(self.noisy_batch_train[:, sample_ids, subject_idx], axis=1)
        clean_batch = self.clean_batch_train[:, subject_idx]
        clean_batch = clean_batch.view(self.N_channels, self.patch_size)
        noisy_batch = noisy_batch.view(self.N_channels, self.patch_size)

        return clean_batch, noisy_batch


def retrieve_val_test_set(Input, SpectraOFF_avg, idx, acceleration_factor, N_samples_per_subject=100):
    
    patch_size, N_acq, _, N_channels = Input.shape
    N_subjects = len(idx)
    nb_samples_list = []
    samples_ids_list = []

    clean_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))
    noisy_batch = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))

    # AF = 0 ONLY
    lb_R = 8 # lower bound R
    ub_R = 80 # upper bound R
    lb_nb_samples = N_acq // ub_R # lower bound number of samples
    ub_nb_samples = N_acq // lb_R # upper bound number of samples

    for i in tqdm(range(N_subjects), leave=False):
        for j in range(N_samples_per_subject):
            if acceleration_factor == 0:
                nb_samples = np.random.random_integers(lb_nb_samples, ub_nb_samples)
            else:
                nb_samples = N_acq // acceleration_factor
            nb_samples_list.append(nb_samples)
            sample_idx = np.random.randint(0, N_acq, nb_samples)
            np.random.shuffle(sample_idx)
            samples_ids_list.append(sample_idx)
            noisy_batch[i, j, :, :] = np.mean(Input[:, sample_idx, i, :], axis=1)
            clean_batch[i, j, :, :] = SpectraOFF_avg[:, i, :]

    noisy_batch = noisy_batch.reshape(-1, patch_size, N_channels)
    clean_batch = clean_batch.reshape(-1, patch_size, N_channels)

    return clean_batch, noisy_batch, nb_samples_list, samples_ids_list