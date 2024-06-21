from pathlib import Path

from dataset import get_loader
from config import Config

from vit import get_vit, SUPPORTED_VIT

from trainer import Trainer

from loss import get_loss, SUPPORTED_LOSS
from optimizer import get_optimizer, SUPPORTED_OPTIMIZERS


celeb_path = Path('./data/CelebA')
train_path = celeb_path / 'train'
test_path = celeb_path / 'test'
val_path = celeb_path / 'validation'

batch_size = 1024
image_limit = None

learning_rates = [1e-6, 1e-4, 1e-2]
num_epochs = 50

debug = False
debug_step = 15
debug_image_count = 2

workers = 48

AUGMENT_DATA = True

print(f'Supported vision transformers: {list(SUPPORTED_VIT.keys())}')
print(f'Supported losses: {list(SUPPORTED_LOSS.keys())}')
print(f'Supported optimizers: {list(SUPPORTED_OPTIMIZERS.keys())}')

models = ['vit_l32', 'vit_h14']
for model_id in models:
    assert model_id in SUPPORTED_VIT.keys()

for model_id in models:
    print(f'Using model: {model_id}')
    if model_id == 'vit_l32':
        device = 'cuda:0'
    elif model_id == 'vit_h14':
        device = 'cuda:1'

    config = Config(
        model_id=model_id,
        img_size=224,
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        device=device
    )

    train_loader = get_loader(
        data_path=train_path,
        config=config,
        batch_size=batch_size,
        image_limit=image_limit,
        augment=AUGMENT_DATA,
        workers=workers
    )
    val_loader = get_loader(
        data_path=val_path,
        config=config,
        batch_size=batch_size,
        image_limit=image_limit,
        augment=AUGMENT_DATA,
        workers=workers
    )

    model = get_vit(config=config)
    for criterion_id in SUPPORTED_LOSS:
        criterion = get_loss(criterion_id)
        for optimizer_id in SUPPORTED_OPTIMIZERS:
            for learning_rate in learning_rates:
                optimizer = get_optimizer(
                    optimizer_id=optimizer_id,
                    model=model,
                    learning_rate=learning_rate
                )
                experiment_name = (
                    f'{model_id}_lr-{learning_rate}_'
                    f'loss-{criterion_id}_optim-{optimizer_id}')
                print(f'Setting learning rate to: {learning_rate}')
                print('##############################')
                print(
                    f'Using {optimizer_id} with {criterion_id} on {model_id}')
                print('##############################')
                trainer = Trainer(
                    model_id=model_id,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    config=config,
                    experiment_name=experiment_name,
                    debug=debug,
                    debug_step=debug_step,
                    debug_image_count=debug_image_count
                )
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=num_epochs
                )
