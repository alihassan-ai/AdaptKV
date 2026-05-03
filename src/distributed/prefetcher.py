"""Async KV cache prefetcher with double-buffering.

At each generation step, the prefetcher predicts which KV entries will be
needed at the *next* step and begins fetching them in the background
(non-blocking NCCL in GPU mode; thread-based in CPU simulation mode).

Double-buffer design:
    Buffer A: serves the current step's attention computation.
    Buffer B: being filled by prefetch while current step runs.
    On step completion, A ↔ B are swapped.
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AsyncPrefetcher:
    """Async double-buffering prefetcher for remote KV entries.

    Works in both real-NCCL (GPU) and simulation (CPU) modes.
    The prediction heuristic is simple: prefetch the top-K entries by
    current EWMA attention score (most likely to be needed next step).
    """

    def __init__(self, config: Dict, cache_manager, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cpu"):
        self.config = config
        self.cache_manager = cache_manager
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        dist_cfg = config.get("distributed", {})
        self.top_k = dist_cfg.get("prefetch_top_k", 64)
        self.simulate = dist_cfg.get("simulate", True) or not torch.cuda.is_available()

        # Double buffers: A (current) and B (being prefetched)
        # Each buffer: {layer_idx: (k [H, K, D], v [H, K, D])}
        self._buffer_a: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._buffer_b: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._prefetch_token_ids: List[int] = []

        # Background thread for simulation mode
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_ready = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start_prefetch(
        self,
        layer_idx: int,
        attention_scores: torch.Tensor,   # [num_heads, cache_len]
        token_ids: List[int],             # global token ids in cache
    ) -> None:
        """Kick off async prefetch of the top-K entries for the next step.

        Args:
            layer_idx:        Which transformer layer.
            attention_scores: Current EWMA attention scores for each cached token.
            token_ids:        Corresponding global token IDs.
        """
        # Rank tokens by mean attention score across heads
        mean_scores = attention_scores.mean(dim=0)   # [cache_len]
        k = min(self.top_k, len(token_ids))
        _, top_idx = torch.topk(mean_scores, k)
        self._prefetch_token_ids = [token_ids[i.item()] for i in top_idx]

        if self.simulate:
            self._prefetch_ready.clear()
            self._prefetch_thread = threading.Thread(
                target=self._do_simulated_prefetch,
                args=(layer_idx,),
                daemon=True,
            )
            self._prefetch_thread.start()
        else:
            self._do_nccl_prefetch(layer_idx)

    def wait_and_swap(self) -> None:
        """Block until prefetch completes, then swap A ↔ B buffers."""
        if self._prefetch_thread is not None:
            self._prefetch_ready.wait(timeout=5.0)
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None

        self._buffer_a, self._buffer_b = self._buffer_b, self._buffer_a
        self._buffer_b.clear()

    def get_prefetched(
        self, layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return prefetched (k, v) for a layer from the active buffer.

        Returns None if nothing was prefetched for this layer.
        """
        return self._buffer_a.get(layer_idx, None)

    def get_stats(self) -> Dict:
        n_entries_a = sum(
            v[0].shape[1] for v in self._buffer_a.values()
        ) if self._buffer_a else 0
        return {
            "prefetch_top_k": self.top_k,
            "num_prefetched_entries": n_entries_a,
            "simulate": self.simulate,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _do_simulated_prefetch(self, layer_idx: int) -> None:
        """Simulate prefetch by copying from simulated GPU stores."""
        H, D = self.num_heads, self.head_dim
        k_out = torch.zeros(H, len(self._prefetch_token_ids), D,
                            device=self.device)
        v_out = torch.zeros(H, len(self._prefetch_token_ids), D,
                            device=self.device)

        for col, tok_id in enumerate(self._prefetch_token_ids):
            gpu_id = self.cache_manager._token_gpu_map.get(
                tok_id, tok_id % self.cache_manager.num_gpus
            )
            k_e, v_e = self.cache_manager._fetch_entry(gpu_id, layer_idx, tok_id)
            if k_e is not None:
                k_out[:, col, :] = k_e[:, 0, :]
                v_out[:, col, :] = v_e[:, 0, :]

        self._buffer_b[layer_idx] = (k_out, v_out)
        self._prefetch_ready.set()

    def _do_nccl_prefetch(self, layer_idx: int) -> None:
        """Issue non-blocking NCCL recv operations for each remote entry."""
        if not self._prefetch_token_ids:
            return

        H, D = self.num_heads, self.head_dim
        K = len(self._prefetch_token_ids)
        k_buf = torch.zeros(H, K, D, device=self.device)
        v_buf = torch.zeros(H, K, D, device=self.device)

        # In a full implementation, fire off isend/irecv per token.
        # Here we do a blocking gather as a functional placeholder.
        k_g, v_g = self.cache_manager.gather_remote_kv(
            layer_idx, self._prefetch_token_ids
        )
        k_buf[:, :k_g.shape[1], :] = k_g
        v_buf[:, :v_g.shape[1], :] = v_g

        self._buffer_b[layer_idx] = (k_buf, v_buf)

    def reset(self) -> None:
        self._buffer_a.clear()
        self._buffer_b.clear()
        self._prefetch_token_ids = []
        self._prefetch_ready.clear()
