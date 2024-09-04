import logging
import os

import torch
import torchvision.transforms.v2 as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torchutils
from vq.data import imagefolder
from vqvae.model import VQVAE


def train():
    config = OmegaConf.load('./vqvae/config.vae.yaml')

    torchutils.set_seeds(**config.repr)

    device = torchutils.get_device()

    # Folder to save stuff.
    checkpoint_folder = os.path.join(config.run.folder, config.run.name)
    torchutils.makedirs0(checkpoint_folder, exist_ok=True)

    # Save config
    OmegaConf.save(config, os.path.join(checkpoint_folder, 'config.yaml'), resolve=True)

    # Create logger.
    logger = torchutils.get_logger(
        f'VQVAE [{torchutils.get_rank()}]',
        filename=os.path.join(checkpoint_folder, config.logging.filename) if torchutils.is_primary() else None,
    )

    # Dataset.
    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.RandomResizedCrop((config.data.image_size, config.data.image_size)),
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
    model = VQVAE(**OmegaConf.to_object(config.model))
    model.save_config(os.path.join(checkpoint_folder, 'model.json'))
    _, cmodel = torchutils.wrap_module(model, strategy=config.env.strategy, compile=config.env.compile)

    # Create optimizer.
    optimizer = torch.optim.Adam(cmodel.parameters(), **OmegaConf.to_object(config.optimizer))

    # Create loss.
    criterion = torch.nn.MSELoss()

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
        model=cmodel,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        logger=logger,
        checkpoint_folder=checkpoint_folder,
        batches_done=batches_done,
    )


def train_loop(
    config: DictConfig,
    dataset: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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

            # Forward.
            recon, commit_loss = model(image, return_loss=True)

            # Loss.
            recon_loss = criterion(recon, image)
            batch_loss = recon_loss + commit_loss

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

            # Save current results for checkpoint progress.
            if batches_done % config.train.running == 0:
                recon = torchutils.gather(recon)
                save_image0(recon, os.path.join(checkpoint_folder, 'running.png'), normalize=True, value_range=(-1, 1))

            # Save snapshop.
            if batches_done % config.train.save_every == 0:
                kbatches = f'{batches_done/1000:.2f}k'
                torchutils.save_model(checkpoint_folder, model, f'{kbatches}')
                image = torchutils.gather(image)
                if image.size(0) != recon.size(0):  # recon might be already gathered.
                    recon = torchutils.gather(recon)
                save_image0(
                    torch.cat([image, recon]),
                    os.path.join(checkpoint_folder, f'snapshot-{kbatches}.png'),
                    normalize=True,
                    value_range=(-1, 1),
                )

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
