import torch.nn as nn
import torch


def psnr(prediction, label):
    # Constant for numerical stability
    EPS = 1e-8
    mse = nn.MSELoss()(prediction, label)
    return -10 * torch.log10(mse + EPS)


SUPPORTED_LOSS = {
    'L1': nn.L1Loss(),
    'MSE': nn.MSELoss(),
    'PSNR': psnr
}


def get_loss(loss_id):
    if not isinstance(loss_id, str):
        raise TypeError(
            f'loss_id should be a string. Got {type(loss_id)} instead.')
    loss_id = loss_id.upper()
    if loss_id not in SUPPORTED_LOSS.keys():
        raise ValueError(
            f'Unsupported loss_id. Try one of: {list(SUPPORTED_LOSS.keys())}')
    return SUPPORTED_LOSS[loss_id]
