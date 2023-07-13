import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import metrics
from main_model import EMA
import scipy
import skimage
import sklearn
from dnresunet import DnResUNet
from main_model import DDPM
from aftnet1d import AFT_RACUNet
from wavegrad import WaveGrad
from scipy.fft import fft, ifft, fftshift

loss_fn = torch.nn.L1Loss()

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=1, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    
    # ema = EMA(0.9)
    # ema.register(model)
    
    # Water Peak Removal:
    # Computing Loss on Cropped Interval
    crop_start = 1075
    crop_stop = 2048

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=.1, verbose=True
    )
    
    best_valid_loss = 1e10

    training_curves = {"train": [], "valid": []}
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                # print("clean_batch shape:", clean_batch.shape)
                # print("noisy_batch shape:", noisy_batch.shape)
                optimizer.zero_grad()
                
                if type(model) == DDPM:
                    loss = model(clean_batch, noisy_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                elif type(model) == DnResUNet or type(model) == AFT_RACUNet:
                    recon_batch = model(noisy_batch)
                    loss = loss_fn(recon_batch[:,:,crop_start:crop_stop], \
                                   clean_batch[:,:,crop_start:crop_stop])
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            
            training_curves["train"].append( avg_loss / batch_no )
            lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        if type(model) == DDPM:
                            loss = model(clean_batch, noisy_batch)
                        elif type(model) == DnResUNet or type(model) == AFT_RACUNet:
                            recon_batch = model(noisy_batch)
                            loss = loss_fn(recon_batch[:,:,crop_start:crop_stop], \
                                           clean_batch[:,:,crop_start:crop_stop])
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at", epoch_no,)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                
            training_curves["valid"].append( avg_loss_valid / batch_no )
    
    torch.save(model.state_dict(), final_path) 
    
    with open(foldername + "/training_curves.pkl", "wb") as file:
        pickle.dump(training_curves, file)    
   

def compute_metrics(clean, noisy):
    psnr, rmse, pcc, scc = [], [], [], []
    for i in range(clean.shape[0]):
        data_range = np.max(np.concatenate((clean[i].flatten(), noisy[i].flatten())))
        pcc_ = scipy.stats.pearsonr(clean[i].flatten(), noisy[i].flatten())
        pcc.append(pcc_.statistic)
        scc_ = scipy.stats.spearmanr(clean[i].flatten(), noisy[i].flatten())
        scc.append(scc_.statistic)
        psnr.append(skimage.metrics.peak_signal_noise_ratio(clean[i].flatten(), noisy[i].flatten(), data_range=data_range))
        # ssim.append(skimage.metrics.structural_similarity(clean[i].flatten(), noisy[i].flatten(), data_range=data_range))
        rmse.append(sklearn.metrics.mean_squared_error(clean[i].flatten(), noisy[i].flatten()))
    return psnr, rmse, pcc, scc

def lse_adjust(recon_batch, noisy_batch, amplitude=False):
    recon_batch_adjusted = np.zeros_like(recon_batch)
    for i in range(recon_batch.shape[0]):
        if amplitude:
            X = np.vstack((np.ones_like(recon_batch[i:i+1]), recon_batch[i:i+1])).T
            beta, alpha = np.linalg.pinv(X) @ noisy_batch[i:i+1].T
            recon_batch_adjusted[i:i+1] = alpha * recon_batch[i:i+1] + beta
        else:
            beta = np.mean(noisy_batch[i:i+1] - recon_batch[i:i+1])
            recon_batch_adjusted[i:i+1] = recon_batch[i:i+1] + beta
    return recon_batch_adjusted

def evaluate(model, test_loader, shots, device, fid=False, lse=False, foldername="", filename=""):

    if (model is None):
        print("CHECK")

    metric_names = ["psnr", "rmse", "pcc", "scc"]
    metric_names = metric_names + [m + "_model" for m in metric_names]
    values = [[] for i in range(len(metric_names))]
    metrics = dict(zip(metric_names, values))

    crop_start = 1075
    crop_stop = 2048
    
    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            
            if shots > 1:
                output = 0
                for i in range(shots):
                    if model is not None:
                        if type(model) == DDPM or type(model) == WaveGrad:
                            output+=model.denoising(noisy_batch)
                        elif type(model) == DnResUNet or type(model) == AFT_RACUNet:
                            model.eval()
                            output+=model(noisy_batch)
                    else:
                        output = noisy_batch
                output /= shots
            else:
                if model is not None:
                    if type(model) == DDPM or type(model) == WaveGrad:
                        output = model.denoising(noisy_batch) #B,1,L
                    elif type(model) == DnResUNet or type(model) == AFT_RACUNet:
                        output = model(noisy_batch)
                else:
                    output = noisy_batch

            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            
            if lse:
                out_numpy = lse_adjust(out_numpy, noisy_numpy)

            # keeping only the real part
            if fid:
                if np.iscomplexobj(noisy_numpy): # complex dtype
                    noisy_spectra = np.zeros_like(noisy_numpy)
                    for i in range(noisy_numpy.shape[0]):
                        noisy_spectra[i,:,0] = fftshift(fft(noisy_numpy[i,:,0]))
                else:
                    assert noisy_numpy.shape[-1] == 2, "The number of channels is not equal to 2." 
                    noisy_fid = np.apply_along_axis(lambda args: [complex(*args)], 2, noisy_numpy)
                    noisy_spectra = np.zeros_like(noisy_fid)
                    for i in range(noisy_numpy.shape[0]):
                        noisy_spectra[i,:,0] = fftshift(fft(noisy_fid[i,:,0]))

                clean_numpy_real = np.real( clean_numpy[...,0:1] )
                noisy_numpy_real = np.real( noisy_spectra[...,0:1] )
                out_numpy_real = np.real( out_numpy[...,0:1] )
            else:
                clean_numpy_real = np.real( clean_numpy[...,0:1] )
                noisy_numpy_real = np.real( noisy_numpy[...,0:1] )
                out_numpy_real = np.real( out_numpy[...,0:1] )
                
            out_noisy = compute_metrics(clean_numpy_real[crop_start:crop_stop], \
                                        noisy_numpy_real[crop_start:crop_stop])
            out_model = compute_metrics(clean_numpy_real[crop_start:crop_stop], \
                                        out_numpy_real[crop_start:crop_stop])
            out = out_noisy + out_model
            
            for i in range(len(metrics)):
                metrics[metric_names[i]] += out[i]

            restored_sig.append(out_numpy)

            avg_values = list(map(lambda x: np.mean(x), metrics.values()))
            metrics_avg = dict(zip(metrics.keys(), avg_values))

            it.set_postfix(
                ordered_dict=metrics_avg,
                refresh=True,
            )
    
    restored_sig = np.concatenate(restored_sig)

    for name, value in metrics_avg.items():
        print(name + ": ", value)

    with open(foldername + "/" + filename, "wb") as file:
        pickle.dump(metrics, file)

    list_metric = metrics_avg.values()

    return list_metric