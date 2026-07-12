import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from favapy import FAVA


@pytest.fixture
def data_dir() -> Path:
    data_dir = os.environ.get('DATA_DIR')
    if data_dir is None:
        raise ValueError('DATA_DIR environment variable not set')
    return Path(data_dir)


@pytest.fixture
def test_dataset(data_dir) -> pd.DataFrame:
    data_file_path = data_dir / 'Example_dataset_GSE75748_sc_cell_type_ec.tsv'
    return pd.read_csv(data_file_path, sep='\t', index_col=0).iloc[:100, :100]


def test_cook_and_network_schema(test_dataset):
    model = FAVA(test_dataset, n_hidden=50, n_latents=5)
    network = model.cook(max_epochs=2, batch_size=32, enable_progress_bar=False).get_association_network(
        metric='pearson',
        interaction_count=10,
    )
    assert list(network.columns) == ['Protein_1', 'Protein_2', 'Score']
    assert len(network) == 10


def test_z_mean_is_deterministic(test_dataset):
    model = FAVA(test_dataset, n_hidden=50, n_latents=5)
    model.cook(max_epochs=2, batch_size=32, enable_progress_bar=False)
    first = model.get_latent_representation()
    second = model.get_latent_representation()
    assert np.allclose(first, second)


def test_cook_requires_inference_after_training(test_dataset):
    model = FAVA(test_dataset, n_hidden=50, n_latents=5)
    with pytest.raises(RuntimeError):
        model.get_latent_representation()


def test_anndata_dense_input():
    x = np.random.default_rng(0).random((20, 10)).astype(np.float32)
    adata = anndata.AnnData(X=x)
    adata.var_names = [f'gene_{i}' for i in range(20)]
    adata.obs_names = [f'cell_{i}' for i in range(10)]
    model = FAVA(adata, n_hidden=10, n_latents=3)
    network = model.cook(max_epochs=1, batch_size=8, enable_progress_bar=False).get_association_network(
        interaction_count=5,
    )
    assert len(network) == 5


def test_anndata_sparse_input():
    x = scipy.sparse.random(15, 8, density=0.5, random_state=0).astype(np.float32)
    adata = anndata.AnnData(X=x)
    adata.var_names = [f'gene_{i}' for i in range(15)]
    adata.obs_names = [f'cell_{i}' for i in range(8)]
    model = FAVA(adata, n_hidden=8, n_latents=2)
    z_mean = model.cook(max_epochs=1, batch_size=4, enable_progress_bar=False).get_latent_representation()
    assert z_mean.shape == (15, 2)
