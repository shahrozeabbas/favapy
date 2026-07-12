"""PyTorch VAE module for FAVA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InferenceOutput:
    z_mean: torch.Tensor
    z_log_sigma: torch.Tensor
    z: torch.Tensor


@dataclass
class GenerativeOutput:
    reconstruction: torch.Tensor


@dataclass
class LossOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class FAVAModule(nn.Module):
    """Legacy FAVA variational autoencoder."""

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        n_latents: int,
        sampling_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latents = n_latents
        self.sampling_std = sampling_std

        self.encoder_hidden = nn.Linear(n_obs, n_hidden)
        self.z_mean_layer = nn.Linear(n_hidden, n_latents)
        self.z_log_sigma_layer = nn.Linear(n_hidden, n_latents)

        self.decoder_hidden = nn.Linear(n_latents, n_hidden)
        self.decoder_output = nn.Linear(n_hidden, n_obs)

        for layer in (
            self.encoder_hidden,
            self.z_mean_layer,
            self.z_log_sigma_layer,
            self.decoder_hidden,
            self.decoder_output,
        ):
            _init_linear(layer)

    def inference(
        self,
        x: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> InferenceOutput:
        hidden = torch.relu(self.encoder_hidden(x))
        z_mean = self.z_mean_layer(hidden)
        z_log_sigma = self.z_log_sigma_layer(hidden)

        if epsilon is None:
            epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(z_log_sigma) * (self.sampling_std * epsilon)
        return InferenceOutput(z_mean=z_mean, z_log_sigma=z_log_sigma, z=z)

    def generative(self, z: torch.Tensor) -> GenerativeOutput:
        hidden = torch.relu(self.decoder_hidden(z))
        reconstruction = torch.sigmoid(self.decoder_output(hidden))
        return GenerativeOutput(reconstruction=reconstruction)

    def loss(self, x: torch.Tensor, inference: InferenceOutput, generative: GenerativeOutput) -> LossOutput:
        reconstruction_loss = (x - generative.reconstruction).square().mean(dim=-1)
        reconstruction_loss = reconstruction_loss * self.n_obs

        kl = 1 + inference.z_log_sigma - inference.z_mean.square() - inference.z_log_sigma.exp()
        kl_loss = -0.5 * kl.sum(dim=-1)

        total = (0.9 * reconstruction_loss + 0.1 * kl_loss).mean()
        return LossOutput(
            loss=total,
            reconstruction_loss=reconstruction_loss.mean(),
            kl_loss=kl_loss.mean(),
        )

    def forward(
        self,
        x: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> tuple[GenerativeOutput, LossOutput]:
        inference = self.inference(x, epsilon=epsilon)
        generative = self.generative(inference.z)
        loss = self.loss(x, inference, generative)
        return generative, loss
