import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
# from Cmodules import complex_to_polar, polar_to_complex
from torch.optim.lr_scheduler import CosineAnnealingLR
import hdf5storage
import pandas as pd
import scipy.io as sio
from torchsummary import summary

import argparse
from thop import profile
from API import *
from scipy.stats import spearmanr
from progressive_model import Restormer
from beam_utils import DownPrecoding, EqChannelGain, DataRate

# model parameters
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_shape', default=[4, 2, 72, 32], type=int,
                        nargs='*')  # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--hid_S', default=32, type=int)   # hid_S 32 for Cmodel
    parser.add_argument('--hid_T', default=128, type=int)   # hid_T 128 for Cmodel
    parser.add_argument('--N_S', default=6, type=int)
    parser.add_argument('--N_T', default=2, type=int)
    parser.add_argument('--groups', default=4, type=int)
    parser.add_argument('--reduction', default=32, type=int)  # Feedback Reduction
    return parser


args = create_parser().parse_args()
config = args.__dict__

num_epochs = 10
num_epochs_step1 = 0
batch_size = 25
Spearman_weight = 0  # upper from 0.5
sigma2_UE = 0.05

####################
## Adjust
# P_alpha = 2
# P_beta = 3
P_alpha = 1.5
P_beta = 4

####################
# Device Definition
device = torch.device("cuda:1")

# # Load Data
# # Training  Dataset
# PN_EF_H_file = '/home/hzl/ChannelP/ConvLSTM_Demo/dataset_60/200/Raw/H60_200_Serier_A4.mat'
# PN_EF_H = hdf5storage.loadmat(PN_EF_H_file)
# PN_EF_H = PN_EF_H['H_serier']
# file_path = 'H60_200_Serier_A4.npy'
# np.save(file_path, PN_EF_H)
#
# PN_EQ_H_file = '/home/hzl/ChannelP/ConvLSTM_Demo/dataset_60/200/Raw/HPN60_200_Serier_A4.mat'
# PN_EQ_H = hdf5storage.loadmat(PN_EQ_H_file)
# PN_EQ_H = PN_EQ_H['H_serier']
# file_path = 'HPN60_200_Serier_A4.npy'
# np.save(file_path, PN_EQ_H)
# print(Check)

Raw_Serier_file = '/home/hzl/ChannelP/ConvLSTM_Demo/dataset_30/20/HPN30_20_Serier_A8.npy'
Raw_Serier = np.expand_dims(np.load(Raw_Serier_file), axis=-1)
Raw_Serier = np.concatenate((np.real(Raw_Serier), np.imag(Raw_Serier)), axis=-1)+0.5
Check_Serier_file = '/home/hzl/ChannelP/ConvLSTM_Demo/dataset_30/20/H30_20_Serier_A8.npy'
Check_Serier = np.expand_dims(np.load(Check_Serier_file), axis=-1)
Check_Serier = np.concatenate((np.real(Check_Serier), np.imag(Check_Serier)), axis=-1)+0.5

random_indices = np.random.permutation(len(Raw_Serier))
Raw_Serier_S = Raw_Serier[random_indices]
Check_Serier_S = Check_Serier[random_indices]

train_data_PN = Raw_Serier[1000:15000]
val_data_PN = Raw_Serier[15000:18000]

val_data_PN_mean = np.mean(val_data_PN)
val_data_PN_std = np.std(val_data_PN)

train_data_tensor_PN = torch.from_numpy(train_data_PN)
val_data_tensor_PN = torch.from_numpy(val_data_PN)

train_data_PNfree = Check_Serier[1000:15000]
val_data_PNfree = Check_Serier[15000:18000]
train_data_tensor_PNfree = torch.from_numpy(train_data_PNfree)
val_data_tensor_PNfree = torch.from_numpy(val_data_PNfree)


#   PIDHIS Dataset=--
class SerierDataset_DePN(Dataset):
    def __init__(self, tensor_PN, tensor_PNFree):
        self.CSI_Serier_Meta_PN = tensor_PN
        self.CSI_Serier_Meta_PNfree = tensor_PNFree

    def __len__(self):
        return len(self.CSI_Serier_Meta_PN)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pn_x = self.CSI_Serier_Meta_PN[idx].to(torch.float32)
        pn_x = pn_x[0:4, :, :, :]  # Short and Long Correlation
        pn_x = torch.permute(pn_x, (0, 3, 1, 2))
        pnfree_y = self.CSI_Serier_Meta_PNfree[idx].to(torch.float32)
        pnfree_y = torch.cat([pnfree_y[3:4, :, :, :], pnfree_y[4:5, :, :, :]], 0)   # For the Denoise Purpose
        pnfree_y_one = pnfree_y[0:1, :, :, :]
        pnfree_y = torch.permute(pnfree_y, (0, 3, 1, 2))
        pnfree_y_one = torch.permute(pnfree_y_one, (0, 3, 1, 2))

        # pnfree_x = self.CSI_Serier_Meta_PNfree[idx].to(torch.float32)   # This is Perfect Channel Setting
        # pn_x = pnfree_x[0:4, :, :, :]  # Short and Long Correlation
        # pn_x = torch.permute(pn_x, (0, 3, 1, 2))

        return pn_x, pnfree_y, pnfree_y_one


Train_set = SerierDataset_DePN(train_data_tensor_PN, train_data_tensor_PNfree)
Val_set = SerierDataset_DePN(val_data_tensor_PN, val_data_tensor_PNfree)
print(f"trainset len {len(Train_set)} valset len {len(Val_set)}")


# Training Data Loader
train_loader = DataLoader(Train_set, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(Val_set, shuffle=False, batch_size=batch_size)
# check_loader = DataLoader(Check_set, shuffle=False, batch_size=batch_size)


model = Restormer(device).to(device)
# summary(model, (4, 2, 72, 32))
# 计算FLOPs
input_tensor = torch.randn(1, 4, 2, 72, 32).to(device)
flops, params = profile(model, inputs=(input_tensor,))
print(f"Total parameters: {params}")
print(f"Total FLOPs: {flops}")

optim = Adam(model.parameters(), lr=1e-3)
schedular = CosineAnnealingLR(optim, T_max=num_epochs, eta_min=5e-5)

criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss().to(device)


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(output_feature, data):
    with torch.no_grad():
        # De-Centralize
        output_feature = output_feature - 0.5
        data = data - 0.5
        # output_feature = output_feature
        # data = data

        # Calculate the NMSE
        power_gt = data[:, :, 0, :, :] ** 2 + data[:, :, 1, :, :] ** 2
        difference = data - output_feature
        mse = difference[:, :, 0, :, :] ** 2 + difference[:, :, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

    return nmse


class CustomLoss(nn.Module):
    def __init__(self, alpha_fn):
        super(CustomLoss, self).__init__()
        self.alpha_fn = alpha_fn

    def forward(self, loss1, loss2, t):
        alpha = self.alpha_fn(t-2)
        beta = 1-self.alpha_fn(t+2)
        total_loss = alpha * loss1 + beta * loss2
        return total_loss


def linear_alpha(t):
    # Custom Progressive Scheduler
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    a = P_alpha
    b = P_beta

    result = sigmoid(-a * (t - b))

    return result


loss_fn = CustomLoss(alpha_fn=linear_alpha)


for epoch in range(1, num_epochs+1):

    size = len(train_loader.dataset)
    train_loss = 0
    model.train()
    print("Training Phase")
    for batch_num, meta in enumerate(train_loader, 0):
        input, target, target_one = meta
        input, target, target_one = input.to(device), target.to(device), target_one.to(device)

        target_one_C = torch.permute(target_one, (0, 1, 3, 4, 2)).contiguous()
        target_one_C = torch.view_as_complex(target_one_C)
        _, target_one_phase = complex_to_polar(target_one_C)
        target_one_phase = target_one_phase.cpu().detach().numpy()
        target_one_phase = np.reshape(target_one_phase, (target_one_phase.shape[0], -1))

        output = model(input)
        # output_predict = output[:, 1, :, :, :].unsqueeze(1)
        # output_denoise = output[:, 0, :, :, :].unsqueeze(1)
        target_predict = target[:, 1, :, :, :].unsqueeze(1)
        target_denoise = target[:, 0, :, :, :].unsqueeze(1)

        # output_C = torch.permute(output, (0, 1, 3, 4, 2)).contiguous()
        # output_C = torch.view_as_complex(output_C)
        # _, output_phase = complex_to_polar(output_C)
        # output_phase = output_phase.cpu().detach().numpy()
        # output_phase = np.reshape(output_phase, (output_phase.shape[0], -1))

        loss_denoise = criterion(output.flatten(), target_denoise.flatten())
        loss_pid = criterion(output.flatten(), target_predict.flatten())
        # loss_recon = criterion(output.flatten(), target.flatten())
        # rho, _ = spearmanr(target_one_phase.flatten(), output_phase.flatten(), axis=0)
        # rho = torch.tensor(rho, dtype=torch.float32, device=device)
        # rho_loss = 1.0 - torch.abs(rho)
        # loss = loss_fn(loss_denoise, loss_pid, epoch)
        loss = loss_pid

        if batch_num % 20 == 0:
            loss_report, current = loss.item(), (batch_num+1)*len(input)
            print(f"loss: {loss_report:>7f}  [{current:>5d}/{size:>5d}]")

        # BP
        loss.backward()
        optim.step()
        # schedular.step()
        optim.zero_grad()

        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)

    val_size = len(valid_loader.dataset)
    val_loss = 0
    model.eval()
    print("Validation Phase")
    iter_nmse = AverageMeter('Iter nmse')
    iter_nmse_pid = AverageMeter('Iter nmse for Pid')
    iter_nmse_den = AverageMeter('Iter nmse for DeN')
    input_lst, truespid_lst, truesden_lst, preds_lst, check_lst, denoise_lst = [], [], [], [], [], []
    latent_check_lst, PN_latent_lst = [], []
    with torch.no_grad():
        for i, val_meta in enumerate(valid_loader, 0):
            input, target, _ = val_meta
            input, target = input.to(device), target.to(device)
            output = model(input)
            # output_predict = output[:, 1, :, :, :].unsqueeze(1)
            # output_denoise = output[:, 0, :, :, :].unsqueeze(1)
            target_predict = target[:, 1, :, :, :].unsqueeze(1)
            target_denoise = target[:, 0, :, :, :].unsqueeze(1)

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                input, target_predict, target_denoise, output, output], [input_lst, truespid_lst, truesden_lst, preds_lst, denoise_lst]))

            loss_denoise = criterion(output.flatten(), target_denoise.flatten())
            loss_pid = criterion(output.flatten(), target_predict.flatten())
            loss = loss_denoise + loss_pid

            # nmse = evaluator(output, target)
            nmse_pid = evaluator(output, target_predict)
            nmse_den = evaluator(output, target_denoise)
            if i % 20 == 0:
                loss_report_val, current_val = loss.item(), (i + 1) * len(input)
                print(f"val_loss: {loss_report_val:>7f}  [{current_val:>5d}/{val_size:>5d}]")

            val_loss += loss.item()
            prediction = output.cpu().numpy()
            target_check = target_predict.cpu().numpy()

        inputs, target_pid, target_denoise, preds, denoise = map(lambda data: np.concatenate(
            data, axis=0), [input_lst, truespid_lst, truesden_lst, preds_lst, denoise_lst])
        iter_nmse_pid.update(nmse_pid)
        iter_nmse_den.update(nmse_den)
        epoch_NMSE_pid = iter_nmse_pid.avg
        epoch_NMSE_den = iter_nmse_den.avg

    val_loss /= len(valid_loader.dataset)
    inputs = np.concatenate(input_lst, axis=0)
    preds = np.concatenate(preds_lst, axis=0)
    denoise = np.concatenate(denoise_lst, axis=0)
    target_pid = np.concatenate(truespid_lst, axis=0)
    target_denoise = np.concatenate(truesden_lst, axis=0)
    pn_H = np.transpose(inputs, (0, 1, 4, 3, 2))
    pn_H = np.expand_dims(pn_H[:, 3, :, :, :], 1)
    pid_H = np.transpose(preds, (0, 1, 4, 3, 2))
    den_H = np.transpose(denoise, (0, 1, 4, 3, 2))
    H = np.transpose(target_pid, (0, 1, 4, 3, 2))

    pne_precoding = DownPrecoding(pn_H)
    pid_precoding = DownPrecoding(pid_H)
    den_precoding = DownPrecoding(den_H)
    H_precoding = DownPrecoding(H)

    pne_channelGain = EqChannelGain(H, pne_precoding)
    pid_channelGain = EqChannelGain(H, pid_precoding)
    den_channelGain = EqChannelGain(H, den_precoding)
    H_channelGain = EqChannelGain(H, H_precoding)

    pne_dataRate = DataRate(pne_channelGain, sigma2_UE)
    pid_dataRate = DataRate(pid_channelGain, sigma2_UE)
    den_dataRate = DataRate(den_channelGain, sigma2_UE)
    H_dataRate = DataRate(H_channelGain, sigma2_UE)
    print('The Average Spectrum Efficiency of PN Effective Channel is %f bps/Hz' % pne_dataRate)
    print('The Average Spectrum Efficiency of predicted Channel is %f bps/Hz' % pid_dataRate)
    print('The Average Spectrum Efficiency of denoise Channel is %f bps/Hz' % den_dataRate)
    print('The Average Spectrum Efficiency of Perfect realtime Channel is %f bps/Hz' % H_dataRate)

    mse_pid, mae_pid, ssim_pid, psnr_pid, rho_pid = metric(preds-0.5, target_pid-0.5, val_data_PN_mean, val_data_PN_std, True)
    print('PID: mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}, Rho:{:.4f}'.format(mse_pid, mae_pid, ssim_pid, psnr_pid, rho_pid))
    print('Prediction NMSE:{:.4f}'.format(epoch_NMSE_pid))
    mse_den, mae_den, ssim_den, psnr_den, rho_den = metric(denoise-0.5, target_denoise-0.5, val_data_PN_mean, val_data_PN_std, True)
    print('DEN: mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}, Rho:{:.4f}'.format(mse_den, mae_den, ssim_den, psnr_den, rho_den))
    print('Denoise NMSE:{:.4f}'.format(epoch_NMSE_den))

    pid_check = {"Pid_H": preds}
    sio.savemat("Predicted.mat", pid_check)
    val_check = {"Test_H": target_pid}
    sio.savemat("Target.mat", val_check)
    # input_check = {"PN_H": inputs}
    # sio.savemat("Input_XdataOR.mat", input_check)
    # latentFeature_check = {"Latent_Feature": latent}
    # sio.savemat("LatentFeature.mat", latentFeature_check)
    # PNlatent_check = {"Test_H": PN_pattern}
    # sio.savemat("PhaseNoisePattern.mat", PNlatent_check)
    print("Epoch:{} Training Loss:{:.8f} Validation Loss:{:.8f}\n".format(
        epoch, train_loss, val_loss))