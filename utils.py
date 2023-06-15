import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import metrics
from main_model import EMA
import scipy
import skimage
from dnresunet import DnResUNet
from main_model import DDPM

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=1, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    
    #ema = EMA(0.9)
    #ema.register(model)
    
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

                elif type(model) == DnResUNet:
                    recon_batch = model(noisy_batch)
                    loss_fn = torch.nn.L1Loss()
                    loss = loss_fn(recon_batch, clean_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
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
                        elif type(model) == DnResUNet:
                            recon_batch = model(noisy_batch)
                            loss_fn = torch.nn.L1Loss()
                            loss = loss_fn(recon_batch, clean_batch)
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
    clean = clean[..., 0:1] # considering real channel only
    noisy = noisy[..., 0:1] 
    data_range = np.max(np.concatenate((clean.flatten(), noisy.flatten())))
    pcc = scipy.stats.pearsonr(clean.flatten(), noisy.flatten())
    pcc = pcc.statistic
    scc = scipy.stats.spearmanr(clean.flatten(), noisy.flatten())
    scc = scc.statistic
    psnr = skimage.metrics.peak_signal_noise_ratio(clean.flatten(), noisy.flatten(), data_range=data_range)
    ssim = skimage.metrics.structural_similarity(clean.flatten(), noisy.flatten(), data_range=data_range)
    return psnr, ssim, pcc, scc

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

def evaluate(model, test_loader, shots, device, lse=False, foldername="", filename=""):

    metric_names = ["psnr", "ssim", "pcc", "scc"]
    metric_names = metric_names + [m + "_model" for m in metric_names]
    values = [[] for i in range(len(metric_names))]
    metrics = dict(zip(metric_names, values))


    # psnr_total = 0
    # ssim_total = 0
    # pcc_total = 0
    # scc_total = 0
    # eval_points = 0

    # psnr_model_total = 0
    # ssim_model_total = 0
    # pcc_model_total = 0
    # scc_model_total = 0

    eval_points = 0
    
    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            
            if shots > 1:
                output = 0
                for i in range(shots):
                    if type(model) == DDPM:
                        output+=model.denoising(noisy_batch)
                    elif type(model) == DnResUNet:
                        model.eval()
                        output+=model(noisy_batch)
                output /= shots
            else:
                if type(model) == DDPM:
                    output = model.denoising(noisy_batch) #B,1,L
                elif type(model) == DnResUNet:
                    output = model(noisy_batch)
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            
            if lse:
                out_numpy = lse_adjust(out_numpy, noisy_numpy)

            eval_points += 1 # len(output)
            out_noisy = compute_metrics(clean_numpy, noisy_numpy)
            out_model = compute_metrics(clean_numpy, out_numpy)
            out = out_noisy + out_model
            
            for i in range(len(metrics)):
                metrics[metric_names[i]].append(out[i])

            # psnr_total += psnr
            # ssim_total += ssim
            # pcc_total += pcc
            # scc_total += scc

            # psnr_model, ssim_model, pcc_model, scc_model = compute_metrics(clean_numpy, out_numpy)
            # psnr_model_total += psnr_model
            # ssim_model_total += ssim_model
            # pcc_model_total += pcc_model
            # scc_model_total += scc_model

            restored_sig.append(out_numpy)

            avg_values = list(map(lambda x: np.mean(x), metrics.values()))
            metrics_avg = dict(zip(metrics.keys(), avg_values))

            it.set_postfix(
                ordered_dict=metrics_avg,
                # {
                #     "psnr": psnr_total/eval_points,
                #     "ssim": ssim_total/eval_points,
                #     "pcc": pcc_total/eval_points,
                #     "scc": scc_total/eval_points,
                #     "psnr_model": psnr_model_total/eval_points,
                #     "ssim_model": ssim_model_total/eval_points,
                #     "pcc_model": pcc_model_total/eval_points,
                #     "scc_model": scc_model_total/eval_points
                # },
                refresh=True,
            )
    
    restored_sig = np.concatenate(restored_sig)

    for name, value in metrics_avg.items():
        print(name + ": ", value)

    # print("psnr_total: ", psnr_total/eval_points)
    # print("ssim_total: ", ssim_total/eval_points)
    # print("pcc_total: ", pcc_total/eval_points)
    # print("scc_total: ", scc_total/eval_points)
    # print("psnr_model_total: ", psnr_model_total/eval_points)
    # print("ssim_model_total: ", ssim_model_total/eval_points)
    # print("pcc_model_total: ", pcc_model_total/eval_points)
    # print("scc_model_total: ", scc_model_total/eval_points)

    # list_metric = [psnr_total, ssim_total, pcc_total, scc_total, psnr_model_total, ssim_model_total, \
    #                pcc_model_total, scc_model_total]
    
    # list_metric = [i / eval_points for i in list_metric]

    with open(filename, "wb") as file:
        pickle.dump(metrics, file)
        
    list_metric = metrics_avg.values()

    return list_metric
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    