"""Lightning training plan for FAVA."""

from __future__ import annotations

import lightning as L
import torch
from torch import optim

from .models import FAVAModule


class FAVATrainingPlan(L.LightningModule):
    """Lightning wrapper around the legacy FAVA VAE."""

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        n_latents: int,
        learning_rate: float = 0.001,
        clipnorm: float = 0.001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.module = FAVAModule(
            n_obs=n_obs,
            n_hidden=n_hidden,
            n_latents=n_latents,
        )
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        _, loss_output = self.module(x)
        self.log('train_loss', loss_output.loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss_output.loss

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        x, gene_idx = batch
        inference = self.module.inference(x)
        return {'gene_idx': gene_idx, 'z_mean': inference.z_mean}

    def configure_optimizers(self):
        return optim.Adam(
            self.module.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        del gradient_clip_val, gradient_clip_algorithm
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    torch.nn.utils.clip_grad_norm_([parameter], self.clipnorm)
