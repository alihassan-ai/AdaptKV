"""Communication-aware KV cache entry placement / eviction scoring.

Effective score for entry i on GPU j (local_rank = local):

    effective_score(i, j) = importance(i) - λ × comm_cost(j, local)

where:
    importance(i) = cumulative attention score (normalized to [0, 1])
    comm_cost(j, local) = 0.0 if j == local, else 1.0
    λ = config['distributed']['comm_penalty_lambda']

This biases eviction towards remote entries first (they are more expensive
to fetch), and towards less-important entries overall.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CommAwarePlacement:
    """Scores cached entries combining importance and communication cost.

    Used by the DistributedCacheManager to decide which entries to evict or
    migrate when cache pressure is high.
    """

    def __init__(self, config: Dict, local_rank: int = 0, num_gpus: int = 1):
        self.local_rank = local_rank
        self.num_gpus = num_gpus
        self.lambda_ = config.get("distributed", {}).get("comm_penalty_lambda", 0.1)
        self.migration_threshold = config.get("distributed", {}).get("migration_threshold", 0.3)

    def comm_cost(self, gpu_id: int) -> float:
        """Return communication cost for accessing an entry on gpu_id."""
        return 0.0 if gpu_id == self.local_rank else 1.0

    def effective_score(
        self,
        importance: float,
        gpu_id: int,
    ) -> float:
        """Compute effective score for eviction ranking (higher = more valuable)."""
        return importance - self.lambda_ * self.comm_cost(gpu_id)

    def rank_entries_for_eviction(
        self,
        entry_importance: torch.Tensor,       # [N] normalized importance scores
        entry_gpu_ids: List[int],             # [N] which GPU each entry lives on
    ) -> torch.Tensor:
        """Return indices sorted from least valuable to most valuable (evict first).

        Args:
            entry_importance: [N] float tensor of importance scores in [0, 1].
            entry_gpu_ids:    list of length N, GPU id per entry.

        Returns:
            sorted_indices: [N] long tensor, ascending by effective score.
        """
        N = entry_importance.shape[0]
        comm_costs = torch.tensor(
            [self.comm_cost(g) for g in entry_gpu_ids],
            dtype=entry_importance.dtype,
            device=entry_importance.device,
        )
        eff_scores = entry_importance - self.lambda_ * comm_costs

        # Ascending sort: lowest effective score → evict first
        return eff_scores.argsort(descending=False)

    def should_migrate(self, importance: float, gpu_id: int) -> bool:
        """Return True if an entry is important enough to migrate to local GPU."""
        if gpu_id == self.local_rank:
            return False
        # Migrate if importance is high but communication cost would penalize it
        return importance > self.migration_threshold

    def select_entries_to_keep(
        self,
        entry_importance: torch.Tensor,    # [N]
        entry_gpu_ids: List[int],          # [N]
        budget: int,
    ) -> torch.Tensor:
        """Return the indices of the top-`budget` entries to keep.

        Entries are ranked by effective score; top-budget are retained.

        Returns:
            keep_indices: [budget] sorted long tensor.
        """
        evict_order = self.rank_entries_for_eviction(entry_importance, entry_gpu_ids)
        # The last `budget` in evict_order are the most valuable
        keep_indices = evict_order[-budget:]
        keep_indices, _ = keep_indices.sort()
        return keep_indices

    def compute_placement_plan(
        self,
        entry_importance: torch.Tensor,    # [N]
        current_gpu_ids: List[int],        # [N]
        budget: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute which entries to keep and their optimal GPU placement.

        Returns:
            keep_indices:    [<=budget] indices of entries to retain.
            target_gpu_ids:  GPU ID each kept entry should live on.
        """
        keep_indices = self.select_entries_to_keep(
            entry_importance, current_gpu_ids, budget
        )

        target_gpu_ids = []
        for idx in keep_indices.tolist():
            imp = entry_importance[idx].item()
            cur_gpu = current_gpu_ids[idx]
            if self.should_migrate(imp, cur_gpu):
                target_gpu_ids.append(self.local_rank)
            else:
                target_gpu_ids.append(cur_gpu)

        return keep_indices, target_gpu_ids

    def estimate_communication_volume(
        self,
        entry_importance: torch.Tensor,
        current_gpu_ids: List[int],
        budget: int,
        entry_bytes: int,
    ) -> Dict:
        """Estimate total bytes that would be transferred for a given plan.

        Args:
            entry_bytes: bytes per KV entry (key + value, all heads).

        Returns:
            dict with 'fetch_bytes', 'migration_bytes', 'total_bytes'.
        """
        keep_indices, target_gpus = self.compute_placement_plan(
            entry_importance, current_gpu_ids, budget
        )

        fetch_bytes = 0
        migration_bytes = 0

        for idx, tgt_gpu in zip(keep_indices.tolist(), target_gpus):
            cur_gpu = current_gpu_ids[idx]
            if cur_gpu != self.local_rank:
                fetch_bytes += entry_bytes
            if tgt_gpu != cur_gpu:
                migration_bytes += entry_bytes

        return {
            "fetch_bytes": fetch_bytes,
            "migration_bytes": migration_bytes,
            "total_bytes": fetch_bytes + migration_bytes,
        }
