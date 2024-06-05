import torch


class Config:
    def __init__(
            self,
            model_id,
            img_size,
            train_path, test_path, val_path,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ):
        self.model_id = model_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        self.mean = mean
        self.std = std
