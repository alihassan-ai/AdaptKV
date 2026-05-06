"""Simulate KV cache sharding across N virtual GPUs.

Tracks which virtual GPU each token lives on (ring-buffer assignment) and
counts bytes that would need cross-GPU transfer for a given kept-token set.
Works on a single physical GPU — no NCCL required.
"""

from typing import Dict, List

import torch


class VirtualShardedCache:
    """Simulates multi-GPU KV cache distribution on a single machine.

    Each token is assigned to a virtual GPU (vgpu) in round-robin order.
    Communication volume is computed analytically based on which tokens need
    to be fetched for attention computation by the local vgpu (rank 0).
    """

    def __init__(
        self,
        num_virtual_gpus: int = 4,
        local_rank: int = 0,
        num_heads: int = 16,
        head_dim: int = 64,
        dtype_bytes: int = 2,   # FP16
    ):
        self.n_gpus      = num_virtual_gpus
        self.local_rank  = local_rank
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.dtype_bytes = dtype_bytes

        # bytes for one token's KV pair (keys + values across all heads)
        self.bytes_per_token = 2 * num_heads * head_dim * dtype_bytes

        self._token_gpu: Dict[int, int] = {}
        self._next_id: int = 0

    def add_tokens(self, n: int) -> List[int]:
        """Assign `n` new tokens round-robin and return their token IDs."""
        ids = []
        for _ in range(n):
            tid = self._next_id
            self._token_gpu[tid] = tid % self.n_gpus
            ids.append(tid)
            self._next_id += 1
        return ids

    def assign_comm_aware(
        self,
        importance: torch.Tensor,
        lambda_: float = 0.1,
    ) -> torch.Tensor:
        """Return importance scores penalised by communication cost.

        effective_score(i) = importance(i) - λ × is_remote(i)
        """
        seq_len = importance.shape[0]
        is_remote = torch.tensor(
            [0.0 if self._token_gpu.get(i, i % self.n_gpus) == self.local_rank
             else 1.0
             for i in range(seq_len)],
            dtype=importance.dtype,
        )
        return importance - lambda_ * is_remote

    def comm_volume_bytes(self, kept_token_ids: List[int]) -> int:
        """Count bytes that would be fetched from remote vGPUs for attention."""
        remote = sum(
            1 for tid in kept_token_ids
            if self._token_gpu.get(tid, tid % self.n_gpus) != self.local_rank
        )
        return remote * self.bytes_per_token

    def reset(self) -> None:
        self._token_gpu.clear()
        self._next_id = 0


def compute_comm_volume(
    importance: torch.Tensor,
    budget_k: int,
    num_virtual_gpus: int,
    lambda_comm: float,
    num_heads: int,
    head_dim: int,
    strategy: str = "naive",
) -> Dict:
    """Compute simulated communication volume for naive vs comm-aware placement.

    Args:
        importance:       [seq_len] per-token importance scores
        budget_k:         number of tokens to keep
        num_virtual_gpus: simulated GPU count
        lambda_comm:      communication penalty weight (comm-aware only)
        strategy:         "naive" | "comm_aware"

    Returns:
        dict with comm_bytes, comm_mb, kept_tokens, remote_tokens
    """
    seq_len = importance.shape[0]
    cache   = VirtualShardedCache(
        num_virtual_gpus, num_heads=num_heads, head_dim=head_dim
    )
    cache.add_tokens(seq_len)

    if strategy == "comm_aware":
        scores = cache.assign_comm_aware(importance, lambda_comm)
    else:
        scores = importance.clone()

    _, top_idx  = scores.topk(min(budget_k, seq_len))
    kept        = top_idx.tolist()
    comm_bytes  = cache.comm_volume_bytes(kept)
    remote_count = sum(
        1 for tid in kept
        if cache._token_gpu.get(tid, tid % num_virtual_gpus) != cache.local_rank
    )

    return {
        "strategy":      strategy,
        "kept_tokens":   len(kept),
        "remote_tokens": remote_count,
        "comm_bytes":    comm_bytes,
        "comm_mb":       comm_bytes / 1024**2,
    }
