import logging
import os

import torch
import torchvision.transforms.v2 as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torchutils
from vq.data import imagefolder
from vqgan.discriminator import Discriminator
from vqgan.loss import NonSaturatingGANLoss
from vqvae.model import VQVAE


def train():
    config = OmegaConf.load('./vqgan/config.vae.yaml')
    OmegaConf.resolve(config)

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
    disc = Discriminator(**OmegaConf.to_object(config.discriminator))
    disc.save_config(os.path.join(checkpoint_folder, 'model.disc.json'))
    _, cdisc = torchutils.wrap_module(disc, strategy=config.env.strategy, compile=config.env.compile)

    # Create optimizer.
    optimizer = torch.optim.Adam(cmodel.parameters(), **OmegaConf.to_object(config.optimizer))
    disc_optim = torch.optim.Adam(cdisc.parameters(), **OmegaConf.to_object(config.optimizer))

    # Create loss.
    criterion = torch.nn.MSELoss()
    gan_loss_fn = NonSaturatingGANLoss(decoder_last_layer=model.decoder.output[-1])
    gan_loss_fn.to(device)

    # Load checkpoint if exists.
    consts = torchutils.load_checkpoint(
        os.path.join(checkpoint_folder, 'bins'),
        allow_empty=True,
        model=[model, disc],
        optimizer=[optimizer, disc_optim],
        others={'batches_done': 0},
    )
    batches_done = consts.get('batches_done', 0)

    train_loop(
        config=config,
        dataset=dataset,
        model=cmodel,
        disc=cdisc,
        criterion=criterion,
        gan_loss_fn=gan_loss_fn,
        optimizer=optimizer,
        disc_optim=disc_optim,
        device=device,
        logger=logger,
        checkpoint_folder=checkpoint_folder,
        batches_done=batches_done,
    )


def train_loop(
    config: DictConfig,
    dataset: DataLoader,
    model: torch.nn.Module,
    disc: torch.nn.Module,
    criterion: torch.nn.Module,
    gan_loss_fn: NonSaturatingGANLoss,
    optimizer: torch.optim.Optimizer,
    disc_optim: torch.optim.Optimizer,
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

            disc_optim.zero_grad()

            # UPDATE D
            if config.train.gp_every > 0 and batches_done % config.train.gp_every == 0:
                image.requires_grad_(True)

            # Forward.
            recon, commit_loss = model(image, return_loss=True)

            # D forward.
            real_logits = disc(image)
            fake_logits = disc(recon.detach())

            # D Loss.
            gan_d_loss = gan_loss_fn.d_loss(real_logits, fake_logits)
            d_loss = gan_d_loss

            # Backward and step.
            d_loss.backward()
            disc_optim.step()

            optimizer.zero_grad()

            # D forward w/ grad
            fake_logits = disc(recon)

            # Loss.
            gan_g_loss = gan_loss_fn.g_loss(fake_logits, image, recon) * config.train.gan_lambda
            recon_loss = criterion(recon, image)
            g_loss = gan_g_loss + recon_loss + commit_loss

            # Backward and step.
            g_loss.backward()
            optimizer.step()

            batches_done += 1

            # Logging.
            if (
                batches_done % log_cfg.interval == 0
                or (batches_done <= log_cfg.frequent_until and batches_done % log_cfg.frequent_interval == 0)
                or batches_done in (1, config.train.num_iterations)
            ):
                progress_p = batches_done / config.train.num_iterations * 100
                message = (
                    f'Progress: {progress_p:5.2f}% | Loss: {g_loss.item():8.5f} | G: {gan_g_loss.item():8.5f} | '
                    f'MSE: {recon_loss.item():8.5f} | VQ: {commit_loss.item():8.5f} | D: {gan_d_loss.item():8.5f}'
                )
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

            if batches_done >= config.train.num_iterations:
                break

        # Checkpoint for resuming.
        torchutils.save_checkpoint(
            os.path.join(checkpoint_folder, 'bins'),
            model=[model, disc],
            optimizer=[optimizer, disc_optim],
            others={'batches_done': batches_done},
        )

    # Save last model.
    torchutils.save_model(checkpoint_folder, model, 'last-model')


def main():
    train()


if __name__ == '__main__':
    main()
