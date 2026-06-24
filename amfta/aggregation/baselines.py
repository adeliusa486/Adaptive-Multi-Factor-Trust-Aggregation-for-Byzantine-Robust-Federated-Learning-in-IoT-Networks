"""
Baseline Aggregation Methods
==============================

Implements four baseline Byzantine-robust aggregation methods for comparison
against AMFTA:

  1. FedAvg        — Standard federated averaging (McMahan et al., 2017)
  2. Krum           — Nearest-neighbour selection (Blanchard et al., 2017)
  3. Trimmed Mean   — Coordinate-wise trimmed mean (Yin et al., 2018)
  4. FLTrust        — Server reference trust scoring (Cao et al., 2022)

All methods expose the same interface:
    aggregator.aggregate(global_model, updates, **kwargs) → update_dict
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

UpdateDict = Dict[int, Dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAggregator(ABC):
    """Common interface for all aggregation methods."""

    def reset(self) -> None:
        """Reset any stateful components between experiments."""

    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flatten_updates(updates: UpdateDict) -> Dict[int, torch.Tensor]:
    """Flatten each client update to a 1-D float tensor."""
    return {
        cid: torch.cat([v.float().flatten() for v in u.values()])
        for cid, u in updates.items()
    }


def _uniform_mean(updates: UpdateDict) -> Dict[str, torch.Tensor]:
    """Compute the uniform mean of all updates (FedAvg with equal weights)."""
    n = len(updates)
    result: Dict[str, torch.Tensor] = {}
    for k in next(iter(updates.values())).keys():
        result[k] = torch.stack([updates[cid][k].float() for cid in updates]).mean(0)
    return result


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging.

    Uses uniform weighting by default.  Optionally accepts per-client dataset
    sizes for data-proportional weighting as in the original FedAvg paper.

    Reference: McMahan et al., "Communication-Efficient Learning of Deep
    Networks from Decentralized Data", AISTATS 2017.
    """

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        dataset_sizes: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates.

        Parameters
        ----------
        dataset_sizes : dict {client_id: num_samples} | None
            If provided, weights proportionally by dataset size (FedAvg proper).
            If None, uses uniform weights.
        """
        if dataset_sizes is not None:
            total = sum(dataset_sizes[cid] for cid in updates)
            weights = {cid: dataset_sizes[cid] / total for cid in updates}
        else:
            n = len(updates)
            weights = {cid: 1.0 / n for cid in updates}

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            result[k] = sum(
                weights[cid] * updates[cid][k].float() for cid in updates
            )
        return result


# ---------------------------------------------------------------------------
# Krum / Multi-Krum
# ---------------------------------------------------------------------------

class KrumAggregator(BaseAggregator):
    """Krum (Multi-Krum) Byzantine-robust aggregation.

    Selects the single update whose sum of squared distances to its
    k = N − f − 2 nearest neighbours is minimised, where f is the assumed
    number of Byzantine clients.

    Limitation: Selects ONE update, losing information from all other benign
    clients.  Particularly hurt by non-IID data where legitimate clients with
    unusual distributions may be rejected.

    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent", NeurIPS 2017.

    Parameters
    ----------
    num_byzantine : int | None
        Assumed number of Byzantine clients.  If None, defaults to
        ``floor(0.3 * num_clients)`` (30% assumption).
    multi_krum : bool
        If True, returns the mean of the top-m Krum-selected updates
        rather than the single best.  Default False (standard Krum).
    m : int | None
        Number of updates to select for Multi-Krum.  Defaults to
        N − f − 2 when multi_krum=True.
    """

    def __init__(
        self,
        num_byzantine: Optional[int] = None,
        multi_krum: bool = False,
        m: Optional[int] = None,
    ) -> None:
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        self.m = m

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        client_ids = list(updates.keys())
        n = len(client_ids)

        f = self.num_byzantine if self.num_byzantine is not None else max(1, int(0.3 * n))
        k = max(1, n - f - 2)

        flat = _flatten_updates(updates)

        # Compute pairwise distances and Krum scores
        krum_scores: Dict[int, float] = {}
        for cid in client_ids:
            dists = sorted(
                torch.dist(flat[cid], flat[other]).item()
                for other in client_ids if other != cid
            )
            krum_scores[cid] = sum(dists[:k])

        if self.multi_krum:
            m = self.m or k
            selected = sorted(krum_scores, key=krum_scores.get)[:m]
            logger.debug("Multi-Krum selected %d clients: %s", len(selected), selected)
            return _uniform_mean({cid: updates[cid] for cid in selected})
        else:
            best = min(krum_scores, key=krum_scores.get)
            logger.debug("Krum selected client %d (score=%.4f)", best, krum_scores[best])
            return updates[best]

    def __repr__(self) -> str:
        return f"KrumAggregator(f={self.num_byzantine}, multi_krum={self.multi_krum})"


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMeanAggregator(BaseAggregator):
    """Coordinate-wise Trimmed Mean.

    For each parameter coordinate, removes the top and bottom
    ``trim_fraction`` of values and averages the remainder.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
    Optimal Statistical Rates", ICML 2018.

    Parameters
    ----------
    trim_fraction : float
        Fraction of values to trim from each end.  Default 0.1 (10% each side).
        Should be set ≥ Byzantine fraction for full protection, but the
        Byzantine fraction is not always known in practice.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError(f"trim_fraction must be in [0, 0.5); got {trim_fraction}")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        trim_k = max(1, int(n * self.trim_fraction))

        if trim_k * 2 >= n:
            logger.warning(
                "trim_k=%d trims all updates (n=%d); falling back to FedAvg.", trim_k, n
            )
            return _uniform_mean(updates)

        result: Dict[str, torch.Tensor] = {}
        for k in next(iter(updates.values())).keys():
            stacked = torch.stack([updates[cid][k].float() for cid in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k : n - trim_k]
            result[k] = trimmed.mean(dim=0)

        return result

    def __repr__(self) -> str:
        return f"TrimmedMeanAggregator(trim_fraction={self.trim_fraction})"


# ---------------------------------------------------------------------------
# FLTrust
# ---------------------------------------------------------------------------

class FLTrustAggregator(BaseAggregator):
    """FLTrust — Server Reference Trust Score Aggregation.

    Assigns each client a trust score = ReLU(cosine_similarity(g_i, g_root))
    where g_root is computed by the server from a small clean root dataset.
    Each client's update is normalised to the magnitude of the root update
    before weighted aggregation.

    IMPORTANT: FLTrust REQUIRES a clean labelled root dataset on the server.
    This is the key assumption that AMFTA eliminates.  In simulation we use
    50-200 samples from the held-out validation set; in a real deployment this
    must come from a verified clean source.

    Reference: Cao et al., "FLTrust: Byzantine-robust Federated Learning via
    Trust Bootstrapping", NDSS 2022.

    Parameters
    ----------
    root_dataset_size : int
        Number of samples in the server root dataset (default 200 per paper).
    """

    def __init__(self, root_dataset_size: int = 200) -> None:
        self.root_dataset_size = root_dataset_size
        self._root_update: Optional[Dict[str, torch.Tensor]] = None

    def set_root_update(self, root_update: Dict[str, torch.Tensor]) -> None:
        """Provide the root update computed from the server's clean dataset."""
        self._root_update = root_update

    def aggregate(
        self,
        global_model: nn.Module,
        updates: UpdateDict,
        root_update: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        root_update : dict | None
            The server-computed reference update.  If not provided here, uses
            the last value set via set_root_update().  Falls back to FedAvg
            if neither is available.
        """
        ref = root_update or self._root_update
        if ref is None:
            logger.warning("FLTrust: no root update available; using FedAvg fallback.")
            return _uniform_mean(updates)

        ref_flat = torch.cat([v.float().flatten() for v in ref.values()])
        ref_norm = ref_flat.norm() + 1e-8

        trust: Dict[int, float] = {}
        for cid, update in updates.items():
            flat = torch.cat([v.float().flatten() for v in update.values()])
            cos = F.cosine_similarity(
                flat.unsqueeze(0), ref_flat.unsqueeze(0)
            ).item()
            trust[cid] = max(0.0, cos)  # ReLU

        total = sum(trust.values())
        if total <= 0.0:
            logger.warning("FLTrust: all trust scores are zero; using FedAvg fallback.")
            return _uniform_mean(updates)

        result: Dict[str, torch.Tensor] = {}
        for k in ref.keys():
            weighted = torch.zeros_like(ref[k].float())
            for cid in updates:
                client_flat = torch.cat([updates[cid][v].float().flatten() for v in updates[cid]])
                client_norm = client_flat.norm() + 1e-8
                # Normalise client update to root magnitude
                scale = ref_norm / client_norm
                weighted += trust[cid] * updates[cid][k].float() * scale
            result[k] = weighted / total

        return result

    def __repr__(self) -> str:
        return f"FLTrustAggregator(root_dataset_size={self.root_dataset_size})"


# ---------------------------------------------------------------------------
# Aggregator factory
# ---------------------------------------------------------------------------

AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
    "krum": KrumAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
    "fltrust": FLTrustAggregator,
}


def build_aggregator(method: str, config: Optional[dict] = None) -> BaseAggregator:
    """Build a baseline aggregator by name.

    Parameters
    ----------
    method : str
        One of 'fedavg', 'krum', 'trimmed_mean', 'fltrust'.
    config : dict | None
        Constructor keyword arguments.

    Returns
    -------
    BaseAggregator instance.
    """
    cfg = config or {}
    method_lower = method.lower().replace("-", "_")

    if method_lower not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )
    return AGGREGATOR_REGISTRY[method_lower](**cfg)
