from torchvision.models import vit_b_32 , ViT_B_32_Weights
from torchvision.models import vit_l_32 , ViT_L_32_Weights
from torchvision.models import vit_h_14 , ViT_H_14_Weights

import torch.nn as nn


SUPPORTED_VIT = {
    'vit_b32' : vit_b_32(weights=ViT_B_32_Weights.DEFAULT),
    'vit_l32' : vit_l_32(weights= ViT_L_32_Weights.DEFAULT),
    'vit_h14' : vit_h_14(weights=ViT_H_14_Weights.DEFAULT)
}


def get_vit(config):
    model_id = config.model_id
    img_size = config.img_size

    if model_id not in SUPPORTED_VIT.keys():
        raise ValueError('Unsupported model_id.')
    model = SUPPORTED_VIT[model_id]

    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(
        num_features,
        (3 * img_size * img_size)
    )
    model.to(config.device)
    return model
