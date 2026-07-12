import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from favapy import FAVA


@pytest.fixture
def test_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    values = rng.random((12, 8)).astype(np.float32)
    return pd.DataFrame(
        values,
        index=[f'gene_{i}' for i in range(12)],
        columns=[f'cell_{i}' for i in range(8)],
    )


def _cook_kwargs() -> dict:
    return {
        'max_epochs': 1,
        'batch_size': 4,
        'accelerator': 'cpu',
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }


def test_cook_and_network_schema(test_dataset):
    model = FAVA(test_dataset, n_hidden=8, n_latents=2)
    network = model.cook(**_cook_kwargs()).get_association_network(
        metric='pearson',
        interaction_count=10,
    )
    assert list(network.columns) == ['Protein_1', 'Protein_2', 'Score']
    assert len(network) == 10


def test_z_mean_is_deterministic(test_dataset):
    model = FAVA(test_dataset, n_hidden=8, n_latents=2)
    model.cook(**_cook_kwargs())
    first = model.get_latent_representation()
    second = model.get_latent_representation()
    assert np.allclose(first, second)


def test_cook_requires_inference_after_training(test_dataset):
    model = FAVA(test_dataset, n_hidden=8, n_latents=2)
    with pytest.raises(RuntimeError):
        model.get_latent_representation()


def test_anndata_dense_input():
    x = np.random.default_rng(0).random((10, 20)).astype(np.float32)
    adata = anndata.AnnData(X=x)
    adata.var_names = [f'gene_{i}' for i in range(20)]
    adata.obs_names = [f'cell_{i}' for i in range(10)]
    model = FAVA(adata, n_hidden=10, n_latents=3)
    network = model.cook(**_cook_kwargs()).get_association_network(interaction_count=5)
    assert len(network) == 5


def test_anndata_sparse_input():
    x = scipy.sparse.random(8, 15, density=0.5, random_state=0).astype(np.float32)
    adata = anndata.AnnData(X=x)
    adata.var_names = [f'gene_{i}' for i in range(15)]
    adata.obs_names = [f'cell_{i}' for i in range(8)]
    model = FAVA(adata, n_hidden=8, n_latents=2)
    z_mean = model.cook(**_cook_kwargs()).get_latent_representation()
    assert z_mean.shape == (15, 2)
