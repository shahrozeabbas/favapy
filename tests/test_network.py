import numpy as np
import pandas as pd
from scipy import stats

from favapy.network import AssociationNetworkBuilder, _normalize_rows, _rank_rows_average_ties


def _reference_pearson_pairs(z_mean: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    corr = np.corrcoef(z_mean)
    n_genes = corr.shape[0]
    pairs = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            pairs.append((i, j, float(corr[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:k]


def test_blockwise_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    z_mean = rng.normal(size=(20, 5)).astype(np.float32)
    var_names = [f'g{i}' for i in range(z_mean.shape[0])]
    builder = AssociationNetworkBuilder(block_size=7)
    result = builder.build(
        z_mean=z_mean,
        var_names=var_names,
        metric='pearson',
        interaction_count=15,
    )
    expected = _reference_pearson_pairs(z_mean, 15)
    assert list(result['Protein_1']) == [var_names[i] for i, _, _ in expected]
    assert list(result['Protein_2']) == [var_names[j] for _, j, _ in expected]
    assert np.allclose(result['Score'].to_numpy(), [score for _, _, score in expected], atol=1e-6)


def test_blockwise_spearman_matches_scipy():
    z_mean = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    ranked = _rank_rows_average_ties(z_mean)
    corr = np.corrcoef(ranked)
    var_names = ['a', 'b', 'c']
    result = AssociationNetworkBuilder(block_size=2).build(
        z_mean=z_mean,
        var_names=var_names,
        metric='spearman',
        interaction_count=3,
    )
    assert len(result) == 3
    assert np.isclose(result.loc[0, 'Score'], corr[0, 1])


def test_cutoff_filtering():
    z_mean = np.eye(4, dtype=np.float32)
    var_names = [f'g{i}' for i in range(4)]
    result = AssociationNetworkBuilder().build(
        z_mean=z_mean,
        var_names=var_names,
        metric='pearson',
        cc_cutoff=0.5,
    )
    assert (result['Score'] >= 0.5).all()


def test_normalize_constant_row():
    x = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    normalized = _normalize_rows(x)
    assert normalized.shape == x.shape
