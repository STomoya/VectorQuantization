import logging
import os

import torch
import torchvision.transforms.v2 as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torchutils
from vq.data import imagefolder
from vqgan.transformer import ImageGPT, sample
from vqvae.model import VQVAE


def train():
    config = OmegaConf.load('./vqgan/config.prior.yaml')

    torchutils.set_seeds(**config.repr)

    device = torchutils.get_device()

    # Folder to save stuff.
    checkpoint_folder = os.path.join(config.run.folder, config.run.name)
    torchutils.makedirs0(checkpoint_folder, exist_ok=True)

    # Save config
    OmegaConf.save(config, os.path.join(checkpoint_folder, 'config.yaml'), resolve=True)

    # Create logger.
    logger = torchutils.get_logger(
        f'ImageGPT [{torchutils.get_rank()}]',
        filename=os.path.join(checkpoint_folder, config.logging.filename) if torchutils.is_primary() else None,
    )

    # Dataset.
    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Resize(config.data.image_size),
            T.CenterCrop(config.data.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = imagefolder(config.data.dataset_dir, transforms=transforms)
    dl_kwargs = torchutils.get_dataloader_kwargs()
    dataset = torchutils.create_dataloader(
        dataset,
        config.data.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        **dl_kwargs,
    )

    # Create model.
    # VQVAE
    vqvae = VQVAE.from_config(config.vqvae.config)
    vqvae.load_state_dict(torch.load(config.vqvae.pretrained, map_location='cpu', weights_only=True))
    vqvae.to(device)
    torchutils.freeze(vqvae)
    # Prior model
    model = ImageGPT(**OmegaConf.to_object(config.model))
    model.save_config(os.path.join(checkpoint_folder, 'model.json'))
    _, cmodel = torchutils.wrap_module(model, strategy=config.env.strategy, compile=config.env.compile)

    # Create optimizer.
    optimizer = torch.optim.Adam(cmodel.parameters(), **OmegaConf.to_object(config.optimizer))

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Load checkpoint if exists.
    consts = torchutils.load_checkpoint(
        os.path.join(checkpoint_folder, 'bins'),
        allow_empty=True,
        model=model,
        optimizer=optimizer,
        others={'batches_done': 0},
    )
    batches_done = consts.get('batches_done', 0)

    train_loop(
        config=config,
        dataset=dataset,
        vqvae=vqvae,
        model=cmodel,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        checkpoint_folder=checkpoint_folder,
        batches_done=batches_done,
    )


def train_loop(
    config: DictConfig,
    dataset: DataLoader,
    vqvae: VQVAE,
    model: ImageGPT,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_folder: str,
    batches_done: int | None = None,
):
    log_cfg = config.logging
    batches_done = batches_done or 0
    epoch = batches_done // len(dataset)
    save_image0 = torchutils.only_on_primary(save_image)

    while batches_done < config.train.num_iterations:
        if hasattr(dataset.sampler, 'set_epoch'):
            dataset.sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataset:
            image = batch.get('image').to(device)

            with torch.no_grad():
                label = vqvae.encode(image, return_indices=True)

            # Forward.
            logits = model(label)

            # Loss.
            batch_loss = criterion(logits, label)

            # Backward and step.
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batches_done += 1

            # Logging.
            if (
                batches_done % log_cfg.interval == 0
                or (batches_done <= log_cfg.frequent_until and batches_done % log_cfg.frequent_interval == 0)
                or batches_done in (1, config.train.num_iterations)
            ):
                progress_p = batches_done / config.train.num_iterations * 100
                message = f'Progress: {progress_p:5.2f}% | Loss: {batch_loss.item():.5f}'
                logger.info(message)

            # Save snapshop.
            if batches_done % config.train.save_every == 0:
                kbatches = f'{batches_done/1000:.2f}k'
                torchutils.save_model(checkpoint_folder, model, f'{kbatches}')
                with torch.no_grad():
                    indices = sample(model, label.size(), device)
                    images = vqvae.decode(indices, return_loss=False, input_is_indices=True)
                images = torchutils.gather(images)
                save_image0(
                    images,
                    os.path.join(checkpoint_folder, f'snapshot-{kbatches}.png'),
                    normalize=True,
                    value_range=(-1, 1),
                )

            if batches_done >= config.train.num_iterations:
                break

        # Checkpoint for resuming.
        torchutils.save_checkpoint(
            os.path.join(checkpoint_folder, 'bins'),
            model=model,
            optimizer=optimizer,
            others={'batches_done': batches_done},
        )

    # Save last model.
    torchutils.save_model(checkpoint_folder, model, 'last-model')


def main():
    train()


if __name__ == '__main__':
    main()
