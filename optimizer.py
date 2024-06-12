import torch


SUPPORTED_OPTIMIZERS = {
    "ADAM": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def get_optimizer(optimizer_id, model, learning_rate):
    if not isinstance(optimizer_id, str):
        raise TypeError(
            f"optimizer_id should be a string. Got {type(optimizer_id)} instead.")
    optimizer_id = optimizer_id.upper()
    if optimizer_id not in SUPPORTED_OPTIMIZERS.keys():
        raise ValueError(
            f"Unsupported optimizer_id. Try one of {list(SUPPORTED_OPTIMIZERS.keys())}"
        )
    return SUPPORTED_OPTIMIZERS[optimizer_id](model.parameters(), lr=learning_rate)
