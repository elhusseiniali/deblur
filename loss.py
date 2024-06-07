import torch.nn as nn
import torch


def psnr(prediction, label):
    # Constant for numerical stability
    EPS = 1e-8
    mse = nn.MSELoss()(prediction, label)
    return -10 * torch.log10(mse + EPS)


losses = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'PSNR': psnr
}
