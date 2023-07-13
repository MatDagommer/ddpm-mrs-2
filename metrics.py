import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import scipy
import skimage

def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


def COS_SIM(y, y_pred):
    cos_sim = []
    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim

    
def SNR(y1,y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)
    
    SNR = 10*np.log10(N/D)
    
    return SNR
    
def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out)-SNR(y_clean, y_in)
    
    
def PearsonCoef(x, y):
    result = scipy.stats.pearsonr(x, y)
    return result.statistic
    
def SpearmanCoef(x, y):
    result = scipy.stats.spearmanr(x, y)    
    return result.statistic

def PSNR(x, y):
    psnr = skimage.metrics.peak_signal_noise_ratio(x, y, data_range=np.max(np.concatenate((x, y))))
    return psnr

def RMSE(x, y):
    # ssim = skimage.metrics.structural_similarity(x, y, data_range=np.max(np.concatenate((x, y))))
    rmse = mean_squared_error(x,y)
    return rmse