"""Distributed KV cache manager with ring-buffer sharding across GPUs.

In GPU mode: uses torch.distributed (NCCL) for cross-GPU communication.
In CPU simulation mode: uses in-process SimulatedGPU objects; all data
stays in one process but we track byte-transfer volumes for benchmarking.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from .utils import (
    get_rank, get_world_size, is_distributed,
    send_tensor, recv_tensor, compute_comm_bytes,
)

logger = logging.getLogger(__name__)


# ── CPU simulation objects ────────────────────────────────────────────────────

class SimulatedGPU:
    """Simulates a GPU's local KV cache partition in single-process CPU mode."""

    def __init__(self, gpu_id: int, budget_per_gpu: int):
        self.gpu_id = gpu_id
        self.budget = budget_per_gpu
        # {layer_idx: (k_tensor [H, CL, D], v_tensor [H, CL, D])}
        self.store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.bytes_sent = 0
        self.bytes_recv = 0

    def store_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.store[layer_idx] = (k, v)

    def fetch_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.store.get(layer_idx, None)

    def memory_used(self) -> int:
        total = 0
        for k, v in self.store.values():
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
        return total


# ── Main manager ──────────────────────────────────────────────────────────────

class DistributedCacheManager:
    """Shards the KV cache across multiple GPUs with ring-buffer assignment.

    Assignment rule: token at global position `i` is assigned to GPU `i % num_gpus`.
    Tokens can be migrated to balance load or reduce communication cost.
    """

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        dist_cfg = config.get("distributed", {})
        self.simulate = dist_cfg.get("simulate", True) or not torch.cuda.is_available()
        self.num_gpus = (dist_cfg.get("num_simulated_gpus", 4)
                         if self.simulate
                         else get_world_size())
        self.local_rank = 0 if self.simulate else get_rank()

        max_ctx = max(config.get("evaluation", {}).get("context_lengths", [2048]))
        budget_ratio = config.get("cache", {}).get("budget_ratio", 0.1)
        total_budget = max(1, int(budget_ratio * max_ctx))
        self.budget_per_gpu = max(1, total_budget // self.num_gpus)

        self.migration_threshold = dist_cfg.get("migration_threshold", 0.3)

        # Simulated GPU objects (only in simulate mode)
        self._sim_gpus: Optional[List[SimulatedGPU]] = None
        if self.simulate:
            self._sim_gpus = [
                SimulatedGPU(i, self.budget_per_gpu) for i in range(self.num_gpus)
            ]

        # Tracking: global_token_id → gpu_id assignment
        self._token_gpu_map: Dict[int, int] = {}
        self._next_token_id: int = 0

        # Communication volume tracking (bytes)
        self._total_comm_bytes: int = 0

        logger.info(
            f"DistributedCacheManager init: simulate={self.simulate}, "
            f"num_gpus={self.num_gpus}, budget_per_gpu={self.budget_per_gpu}"
        )

    # ------------------------------------------------------------------ #
    # Token placement                                                       #
    # ------------------------------------------------------------------ #

    def assign_gpu(self, token_idx: int) -> int:
        """Ring-buffer assignment: token i → GPU i % num_gpus."""
        return token_idx % self.num_gpus

    def store_kv_distributed(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Distribute new KV entries to their assigned GPUs.

        key_states: [batch, num_heads, seq_len, head_dim]
        """
        k = key_states.squeeze(0)   # [H, S, D]
        v = value_states.squeeze(0)
        S = k.shape[1]

        for pos in range(S):
            tok_id = self._next_token_id + pos
            gpu_id = self.assign_gpu(tok_id)
            self._token_gpu_map[tok_id] = gpu_id

            k_entry = k[:, pos:pos+1, :]   # [H, 1, D]
            v_entry = v[:, pos:pos+1, :]

            if self.simulate:
                self._sim_store(gpu_id, layer_idx, k_entry, v_entry)
            else:
                self._real_store(gpu_id, layer_idx, k_entry, v_entry)

        self._next_token_id += S

    def gather_remote_kv(
        self,
        layer_idx: int,
        token_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch KV entries (possibly from remote GPUs) for the given token indices.

        Returns:
            (keys, values): [num_heads, len(token_indices), head_dim]
        """
        H, D = self.num_heads, self.head_dim
        keys_out   = torch.zeros(H, len(token_indices), D, device=self.device)
        values_out = torch.zeros(H, len(token_indices), D, device=self.device)

        for col, tok_id in enumerate(token_indices):
            gpu_id = self._token_gpu_map.get(tok_id, tok_id % self.num_gpus)
            k, v = self._fetch_entry(gpu_id, layer_idx, tok_id)
            if k is not None:
                keys_out[:, col, :]   = k[:, 0, :]
                values_out[:, col, :] = v[:, 0, :]

        return keys_out, values_out

    # ------------------------------------------------------------------ #
    # Migration                                                            #
    # ------------------------------------------------------------------ #

    def migrate_entry(self, token_id: int, layer_idx: int,
                      from_gpu: int, to_gpu: int) -> bool:
        """Move a KV entry from one GPU to another.

        Returns True if migration succeeded.
        """
        if from_gpu == to_gpu:
            return True

        if self.simulate:
            src = self._sim_gpus[from_gpu]
            dst = self._sim_gpus[to_gpu]

            kv = src.fetch_kv(layer_idx)
            if kv is None:
                return False
            k, v = kv
            # In simulation, track bytes as if we sent them
            n_bytes = compute_comm_bytes(k) + compute_comm_bytes(v)
            src.bytes_sent += n_bytes
            dst.bytes_recv += n_bytes
            self._total_comm_bytes += n_bytes

            # Copy entry to destination
            dst.store_kv(layer_idx, k, v)
            self._token_gpu_map[token_id] = to_gpu
            return True
        else:
            # Real migration via P2P NCCL
            logger.warning("Real GPU migration not yet implemented; skipping.")
            return False

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        stats = {
            "num_gpus": self.num_gpus,
            "simulate": self.simulate,
            "total_comm_bytes": self._total_comm_bytes,
            "num_tracked_tokens": len(self._token_gpu_map),
        }
        if self.simulate and self._sim_gpus:
            stats["per_gpu_memory_bytes"] = [
                g.memory_used() for g in self._sim_gpus
            ]
            stats["per_gpu_bytes_sent"] = [g.bytes_sent for g in self._sim_gpus]
        return stats

    def reset(self) -> None:
        self._token_gpu_map.clear()
        self._next_token_id = 0
        self._total_comm_bytes = 0
        if self._sim_gpus:
            for g in self._sim_gpus:
                g.store.clear()
                g.bytes_sent = 0
                g.bytes_recv = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _sim_store(self, gpu_id: int, layer_idx: int,
                   k: torch.Tensor, v: torch.Tensor) -> None:
        """Store a single-token KV on the simulated GPU, appending to existing."""
        gpu = self._sim_gpus[gpu_id]
        existing = gpu.fetch_kv(layer_idx)
        if existing is None:
            gpu.store_kv(layer_idx, k, v)
        else:
            k_prev, v_prev = existing
            gpu.store_kv(layer_idx,
                         torch.cat([k_prev, k], dim=1),
                         torch.cat([v_prev, v], dim=1))

        # Check budget
        if gpu.store.get(layer_idx, (None,))[0] is not None:
            if gpu.store[layer_idx][0].shape[1] > gpu.budget:
                # Trim oldest non-sink tokens (simple FIFO)
                k_g, v_g = gpu.store[layer_idx]
                gpu.store_kv(layer_idx, k_g[:, -gpu.budget:, :],
                             v_g[:, -gpu.budget:, :])

    def _real_store(self, gpu_id: int, layer_idx: int,
                    k: torch.Tensor, v: torch.Tensor) -> None:
        """Send KV entry to a remote GPU via NCCL (real distributed mode)."""
        if gpu_id == self.local_rank:
            self._sim_store(gpu_id, layer_idx, k, v)
        else:
            n_bytes = compute_comm_bytes(k) + compute_comm_bytes(v)
            self._total_comm_bytes += n_bytes
            send_tensor(k.contiguous(), dst=gpu_id)
            send_tensor(v.contiguous(), dst=gpu_id)

    def _fetch_entry(
        self, gpu_id: int, layer_idx: int, token_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve a single-token KV entry (may trigger remote fetch)."""
        if self.simulate:
            kv = self._sim_gpus[gpu_id].fetch_kv(layer_idx)
            if kv is None:
                return None, None
            k, v = kv
            # Approximate: return the first entry (in practice we'd index properly)
            col = token_id % max(k.shape[1], 1)
            col = min(col, k.shape[1] - 1)
            n_bytes = k[:, 0, :].nelement() * k.element_size() * 2
            if gpu_id != self.local_rank:
                self._sim_gpus[gpu_id].bytes_sent += n_bytes
                self._total_comm_bytes += n_bytes
            return k[:, col:col+1, :], v[:, col:col+1, :]
        else:
            if gpu_id == self.local_rank:
                return None, None  # would look up local store
            H, D = self.num_heads, self.head_dim
            k_buf = torch.zeros(H, 1, D, device=self.device)
            v_buf = torch.zeros(H, 1, D, device=self.device)
            recv_tensor(k_buf, src=gpu_id)
            recv_tensor(v_buf, src=gpu_id)
            self._total_comm_bytes += compute_comm_bytes(k_buf) + compute_comm_bytes(v_buf)
            return k_buf, v_buf
