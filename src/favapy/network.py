"""Association network construction from latent embeddings."""

from __future__ import annotations

import heapq
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _top_k_sorted_indices(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.intp)
    if k >= len(arr):
        return np.argsort(arr)[::-1]
    top = np.argpartition(arr, -k)[-k:]
    return top[np.argsort(arr[top])[::-1]]


def _rank_rows_average_ties(x: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(x, dtype=np.float64)
    for row_idx in range(x.shape[0]):
        ranked[row_idx] = stats.rankdata(x[row_idx], method='average')
    return ranked


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return centered / norms


class AssociationNetworkBuilder:
    """Build protein association networks from z_mean embeddings."""

    def __init__(self, block_size: int = 512) -> None:
        self.block_size = block_size

    def build(
        self,
        z_mean: np.ndarray,
        var_names: list[str],
        metric: str = 'pearson',
        interaction_count: Optional[int] = 100_000,
        cc_cutoff: Optional[float] = None,
    ) -> pd.DataFrame:
        start_time = time.time()
        if metric not in {'pearson', 'spearman'}:
            raise ValueError(
                f"Invalid metric: '{metric}'. Expected 'pearson' or 'spearman'."
            )

        features = z_mean.astype(np.float64, copy=False)
        if metric == 'spearman':
            features = _rank_rows_average_ties(features)
        normalized = _normalize_rows(features)

        n_genes = normalized.shape[0]
        if cc_cutoff is not None:
            pairs = self._collect_cutoff_pairs(normalized, cc_cutoff)
        else:
            max_pairs = n_genes * (n_genes - 1) // 2
            k = min(interaction_count, max_pairs) if interaction_count is not None else max_pairs
            if interaction_count is not None:
                logger.warning(f' The number of interactions in the output file is {k}.')
            pairs = self._collect_top_k_pairs(normalized, k)

        pairs.sort(key=lambda item: item[2], reverse=True)
        pairs_df = pd.DataFrame(
            {
                'Protein_1': [var_names[i] for i, _, _ in pairs],
                'Protein_2': [var_names[j] for _, j, _ in pairs],
                'Score': [score for _, _, score in pairs],
            }
        )
        logger.info(f'Total time taken: {time.time() - start_time:.2f} seconds')
        return pairs_df

    def _collect_cutoff_pairs(
        self,
        normalized: np.ndarray,
        cc_cutoff: float,
    ) -> list[tuple[int, int, float]]:
        logger.info(f' A cut-off of {cc_cutoff} is applied.')
        n_genes = normalized.shape[0]
        pairs: list[tuple[int, int, float]] = []
        for i0 in range(0, n_genes, self.block_size):
            i1 = min(i0 + self.block_size, n_genes)
            block = normalized[i0:i1]
            corr_block = block @ normalized.T
            for local_i, global_i in enumerate(range(i0, i1)):
                scores = corr_block[local_i, global_i + 1 :]
                valid = np.where(scores >= cc_cutoff)[0]
                for offset in valid:
                    pairs.append((global_i, global_i + 1 + offset, float(scores[offset])))
        return pairs

    def _collect_top_k_pairs(
        self,
        normalized: np.ndarray,
        k: int,
    ) -> list[tuple[int, int, float]]:
        if k <= 0:
            return []

        n_genes = normalized.shape[0]
        heap: list[tuple[float, int, int]] = []

        for i0 in range(0, n_genes, self.block_size):
            i1 = min(i0 + self.block_size, n_genes)
            block = normalized[i0:i1]
            corr_block = block @ normalized.T
            for local_i, global_i in enumerate(range(i0, i1)):
                for global_j in range(global_i + 1, n_genes):
                    score = float(corr_block[local_i, global_j])
                    if len(heap) < k:
                        heapq.heappush(heap, (score, global_i, global_j))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, global_i, global_j))

        return [(i, j, score) for score, i, j in sorted(heap, reverse=True)]
