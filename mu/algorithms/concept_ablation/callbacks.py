from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import torchvision
import pytorch_lightning as pl
import os 
from omegaconf import OmegaConf
import torch
from PIL import Image
import time 
from pytorch_lightning.utilities import rank_zero_info,rank_zero_only
import numpy as np


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(
                        self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass



class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, save_freq=100, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.save_freq = save_freq
        # self.logger_log_images = {
        #     pl.loggers.TensorBoardLogger: self._tb,
        #     pl.loggers.WandbLogger: self._wandb,
        # }

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, logger, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            im = Image.fromarray(grid)
            im.save(path)
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    f'{split}_{k}', torchvision.utils.make_grid(images[k], nrow=4), global_step)
            else:
                logger.log_image(
                    key=f'{split}_{k}', images=[path], step=global_step)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                if isinstance(batch, list):
                    images = pl_module.log_images(batch[0], split=split, **self.log_images_kwargs)
                    images1 = pl_module.log_images(batch[1], split=split, **self.log_images_kwargs)
                else:
                    images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            if isinstance(batch, list):
                for k in images1:
                    N = min(images1[k].shape[0], self.max_images)
                    images1[k] = images1[k][:N]
                    if isinstance(images1[k], torch.Tensor):
                        images1[k] = images1[k].detach().cpu()
                        if self.clamp:
                            images1[k] = torch.clamp(images1[k], -1., 1.)

            self.log_local(pl_module.logger, pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if isinstance(batch, list):
                self.log_local(pl_module.logger, pl_module.logger.save_dir, split + "_reg", images1,
                               pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        #     self.log_img(pl_module, batch, batch_idx, split="train")
        # if self.save_freq is not None:
        #     global_step = trainer.global_step
        #     if global_step % self.save_freq == 0:
        #         filename = f'step_{global_step}.ckpt'
        #         ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        #         trainer.save_checkpoint(ckpt_path)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        if pl_module.freeze_model == 'none':
            # Get the index for tokens that we want to zero the grads for
            grads_text_encoder = pl_module.cond_stage_model.transformer.get_input_embeddings().weight.grad
            index_grads_to_zero = torch.arange(len(pl_module.cond_stage_model.tokenizer)) != \
                pl_module.cond_stage_model.modifier_token_id[0]
            for i in range(len(pl_module.cond_stage_model.modifier_token_id[1:])):
                index_grads_to_zero = index_grads_to_zero & (torch.arange(len(pl_module.cond_stage_model.tokenizer)) !=
                                                             pl_module.cond_stage_model.modifier_token_id[i])
            grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

