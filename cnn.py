from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

import torch.nn as nn


SUPPORTED_CNN = {
    'resnet50': resnet50(weights=ResNet50_Weights.DEFAULT),
    'resnet101': resnet101(weights=ResNet101_Weights.DEFAULT),
    'resnet152': resnet152(weights=ResNet152_Weights.DEFAULT)
}


def get_cnn(config):
    model_id = config.model_id
    img_size = config.img_size

    if model_id not in SUPPORTED_CNN.keys():
        raise ValueError('Unsupported model_id.')
    model = SUPPORTED_CNN[model_id]

    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features,
        (3 * img_size * img_size)
    )
    model.to(config.device)
    return model
