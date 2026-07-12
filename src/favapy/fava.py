"""High-level FAVA model API."""

from __future__ import annotations

from typing import Any, Optional, Union

import anndata
import lightning as L
import numpy as np
import pandas as pd

from .data import FAVADataManager, FAVADataModule
from .network import AssociationNetworkBuilder
from .training import FAVATrainingPlan


class FAVA:
    """Functional Associations using Variational Autoencoders."""

    def __init__(
        self,
        data: Union[anndata.AnnData, pd.DataFrame],
        n_hidden: Optional[int] = None,
        n_latents: Optional[int] = None,
        layer: Optional[str] = None,
        log2_normalization: bool = True,
    ) -> None:
        self.data_manager = FAVADataManager.from_input(data, layer=layer)
        if log2_normalization:
            self.data_manager.preprocess()

        self.n_hidden = n_hidden or self.data_manager.resolve_n_hidden()
        self.n_latents = n_latents or self.data_manager.resolve_n_latents(self.n_hidden)

        self._batch_size = 32
        self._training_plan: FAVATrainingPlan | None = None
        self._z_mean: np.ndarray | None = None
        self._is_cooked = False

    def cook(
        self,
        max_epochs: int = 50,
        batch_size: int = 32,
        accelerator: str = 'auto',
        devices: Union[str, int, list[int]] = 'auto',
        deterministic: Optional[bool] = None,
        **trainer_kwargs: Any,
    ) -> 'FAVA':
        """Train the VAE and return self."""
        self._batch_size = batch_size
        datamodule = FAVADataModule(self.data_manager, batch_size=batch_size)
        self._training_plan = FAVATrainingPlan(
            n_obs=self.data_manager.n_obs,
            n_hidden=self.n_hidden,
            n_latents=self.n_latents,
        )

        trainer_kwargs.setdefault('logger', False)
        trainer_kwargs.setdefault('enable_checkpointing', False)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            deterministic=deterministic,
            **trainer_kwargs,
        )
        trainer.fit(self._training_plan, datamodule=datamodule)
        self._is_cooked = True
        self._z_mean = None
        return self

    def get_latent_representation(self) -> np.ndarray:
        """Return ordered z_mean embeddings with shape (n_genes, n_latents)."""
        self._require_cooked()
        if self._z_mean is None:
            datamodule = FAVADataModule(self.data_manager, batch_size=self._batch_size)
            trainer = L.Trainer(
                logger=False,
                enable_checkpointing=False,
                accelerator='auto',
                devices='auto',
            )
            predictions = trainer.predict(self._training_plan, datamodule=datamodule)
            self._z_mean = self._assemble_z_mean(predictions)
        return self._z_mean

    def get_association_network(
        self,
        metric: str = 'pearson',
        interaction_count: Optional[int] = 100_000,
        cc_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        """Build a protein association network from z_mean embeddings."""
        self._require_cooked()
        z_mean = self.get_latent_representation()
        return AssociationNetworkBuilder().build(
            z_mean=z_mean,
            var_names=self.data_manager.var_names,
            metric=metric,
            interaction_count=interaction_count,
            cc_cutoff=cc_cutoff,
        )

    def _require_cooked(self) -> None:
        if not self._is_cooked or self._training_plan is None:
            raise RuntimeError('Model must be cooked before inference.')

    @staticmethod
    def _assemble_z_mean(predictions: list[dict[str, Any]]) -> np.ndarray:
        if not predictions:
            raise RuntimeError('Prediction returned no batches.')

        sample_batch = predictions[0]['z_mean']
        n_latents = sample_batch.shape[1]
        n_genes = sum(batch['gene_idx'].shape[0] for batch in predictions)
        z_mean = np.zeros((n_genes, n_latents), dtype=np.float32)

        for batch in predictions:
            gene_idx = batch['gene_idx'].cpu().numpy()
            values = batch['z_mean'].detach().cpu().numpy()
            z_mean[gene_idx] = values

        return z_mean
