from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

from torchvision.models import inception_v3, Inception_V3_Weights

import torch.nn as nn


SUPPORTED_MODELS = {
    'resnet50': resnet50(weights=ResNet50_Weights.DEFAULT),
    'resnet101': resnet101(weights=ResNet101_Weights.DEFAULT),
    'resnet152': resnet152(weights=ResNet152_Weights.DEFAULT),
    'inception_v3': inception_v3(weights=Inception_V3_Weights.DEFAULT)
}


def get_model(config):
    model_id = config.model_id
    img_size = config.img_size

    if model_id not in SUPPORTED_MODELS.keys():
        raise ValueError('Unsupported model_id.')
    model = SUPPORTED_MODELS[model_id]

    num_features = model.fc.in_features
    model.fc = nn.Linear(
        num_features,
        (3 * img_size * img_size)
    )
    model.to(config.device)
    return model
