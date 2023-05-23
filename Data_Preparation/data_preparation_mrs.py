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




# TEST 





def Data_Preparation(data_path, n_channels=1):

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

    # SpectraOFF_avg [length x #subjects]
    SpectraOFF_avg = np.mean(SpectraOFF, axis=1)

    patch_size, N_acq, N_subjects = SpectraOFF.shape
    N_samples_per_subject = 100
    N_channels = 1

    SpectraOFF_gt = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))
    SpectraOFF_noisy = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channels))

    print("GT shape: ", SpectraOFF_gt.shape)
    print("Noisy shape: ", SpectraOFF_noisy.shape)

    #ratio_R = np.linspace(10, 60, N_samples_per_subject)
    #ratio_R = list(ratio_R)
    #ratio_R = [int(i) for i in ratio_R]
    ratio_R = 60

    print("Starting dataset generation...")
    for i in tqdm(range(N_subjects)):
        for j in range(N_samples_per_subject):
            #sample_idx = np.random.randint(0, N_acq, N_acq // ratio_R[j])
            sample_idx = np.random.randint(0, N_acq, N_acq // ratio_R)
            np.random.shuffle(sample_idx)

            SpectraOFF_noisy[i, j, :, 0] = np.real(np.mean(SpectraOFF[:, sample_idx, i], axis=1))
            #SpectraOFF_noisy[i, j, :, 1] = np.imag(np.mean(SpectraOFF[:, sample_idx, i], axis=1))
            
            # Noise as a Target
            #SpectraOFF_gt[i, j, :, 0] = SpectraOFF_noisy[i, j, :, 0] - np.real(SpectraOFF_avg[:, i])
            #SpectraOFF_gt[i, j, :, 1] = SpectraOFF_noisy[i, j, :, 0] - np.imag(SpectraOFF_avg[:, i])
            
            # Clean Spectrum as a target
            SpectraOFF_gt[i, j, :, 0] = np.real(SpectraOFF_avg[:, i])
            #SpectraOFF_gt[i, j, :, 1] = np.imag(SpectraOFF_avg[:, i])

    # CROPPING Spectrum to have length = 512
    patch_size = 512
    SpectraOFF_gt = SpectraOFF_gt[:,:,1000:1512]
    SpectraOFF_noisy = SpectraOFF_noisy[:,:,1000:1512]
    

    train_idx, val_idx = train_test_split(range(N_subjects), test_size=0.4)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)

    X_train = SpectraOFF_noisy[train_idx].reshape(-1, patch_size, N_channels)
    X_val = SpectraOFF_noisy[val_idx].reshape(-1, patch_size, N_channels)
    X_test = SpectraOFF_noisy[test_idx].reshape(-1, patch_size, N_channels)

    y_train = SpectraOFF_gt[train_idx].reshape(-1, patch_size, N_channels)
    y_val = SpectraOFF_gt[val_idx].reshape(-1, patch_size, N_channels)
    y_test = SpectraOFF_gt[test_idx].reshape(-1, patch_size, N_channels)

    print("Converting data to tensors...")
    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0, 2, 1)
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0, 2, 1)

    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0, 2, 1)
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0, 2, 1)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0, 2, 1)
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0, 2, 1)

    # train_set = TensorDataset(X_train, y_train)
    # val_set = TensorDataset(X_val, y_val)
    # test_set = TensorDataset(X_test, y_test)
    # torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    # torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    # torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    print("Done.")

    # if n_channels == 2:
    # train_set = TensorDataset(X_train, y_train)
    # val_set = TensorDataset(X_val, y_val)
    # test_set = TensorDataset(X_test, y_test)

    train_set = TensorDataset(y_train, X_train)
    val_set = TensorDataset(y_val, X_val)
    test_set = TensorDataset(y_test, X_test)

    torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    # elif n_channels == 1:
    #     train_set = TensorDataset(X_train[:, 0:1], y_train[:, 0:1])
    #     val_set = TensorDataset(X_val[:, 0:1], y_val[:, 0:1])
    #     test_set = TensorDataset(X_test[:, 0:1], y_test[:, 0:1])
    #     torch.save(train_set, os.path.join(data_path, "train_set_real.pt"))
    #     torch.save(val_set, os.path.join(data_path, "val_set_real.pt"))
    #     torch.save(test_set, os.path.join(data_path, "test_set_real.pt"))
    print("Dataset ready.")

    return train_set, val_set, test_set

# def SaveTrainData():
    # load .pt datatensor


class DynamicDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):

        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample