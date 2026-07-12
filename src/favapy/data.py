"""Data ingestion and Lightning datamodule for FAVA."""

from __future__ import annotations

import logging
from typing import Optional, Union

import anndata
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def resolve_n_hidden(n_obs: int) -> int:
    if n_obs >= 2000:
        return 1000
    if n_obs >= 500:
        return 500
    return max(50, n_obs // 2)


def resolve_n_latents(n_hidden: int) -> int:
    if n_hidden >= 1000:
        return 100
    if n_hidden >= 500:
        return 50
    return max(5, n_hidden // 10)


def preprocess_expression(x: np.ndarray) -> np.ndarray:
    if np.any(x < 0):
        logger.warning('Negative values detected, skipping log2 normalization.')
    else:
        x = np.log2(1 + x)

    constant = 1e-8
    row_min = np.min(x, axis=1, keepdims=True)
    row_max = np.max(x, axis=1, keepdims=True)
    x = (x - row_min) / (row_max - row_min + constant)
    return np.nan_to_num(x)


class FAVADataManager:
    """Validate, orient, and preprocess omics matrices."""

    def __init__(self, x: np.ndarray, var_names: list[str]) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.var_names = var_names
        if self.x.ndim != 2:
            raise ValueError(f'Expected 2D matrix, got shape {self.x.shape}.')
        if self.x.shape[0] == 0 or self.x.shape[1] == 0:
            raise ValueError(f'Empty data matrix with shape {self.x.shape}.')

    @property
    def n_genes(self) -> int:
        return self.x.shape[0]

    @property
    def n_obs(self) -> int:
        return self.x.shape[1]

    @classmethod
    def from_input(
        cls,
        data: Union[anndata.AnnData, pd.DataFrame],
        layer: Optional[str] = None,
    ) -> 'FAVADataManager':
        if isinstance(data, anndata.AnnData):
            if layer is not None:
                if layer not in data.layers:
                    raise ValueError(
                        f"Layer '{layer}' not found in AnnData object. "
                        f'Available layers: {list(data.layers.keys())}'
                    )
                x_matrix = data.layers[layer]
            else:
                x_matrix = data.X

            if hasattr(x_matrix, 'toarray'):
                x = x_matrix.toarray()
            else:
                x = np.asarray(x_matrix)
            x = x.T
            var_names = data.var.index.tolist()
        elif isinstance(data, pd.DataFrame):
            x = data.values
            var_names = data.index.tolist()
        else:
            raise ValueError(
                f'Unsupported input type: {type(data).__name__}. '
                'Expected anndata.AnnData or pd.DataFrame.'
            )

        return cls(x, var_names)

    def preprocess(self) -> None:
        self.x = preprocess_expression(self.x).astype(np.float32)

    def resolve_n_hidden(self) -> int:
        return resolve_n_hidden(self.n_obs)

    def resolve_n_latents(self, n_hidden: int) -> int:
        return resolve_n_latents(n_hidden)


class _GeneDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], torch.tensor(idx, dtype=torch.long)


class FAVADataModule(L.LightningDataModule):
    """Lightning datamodule over gene rows."""

    def __init__(self, data_manager: FAVADataManager, batch_size: int = 32) -> None:
        super().__init__()
        self.data_manager = data_manager
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            _GeneDataset(self.data_manager.x),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            _GeneDataset(self.data_manager.x),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
