"""Abstract base class for all KV cache compression strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


class BaseKVCache(ABC):
    """Abstract base class defining the interface for all KV cache strategies.

    Subclasses implement update() and evict() to provide different eviction
    or compression policies (H2O, SnapKV, StreamingLLM, AdaptKV, etc.).
    """

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Derive absolute budget from ratio × max_context_length
        max_ctx = max(config.get("evaluation", {}).get("context_lengths", [2048]))
        ratio = config.get("cache", {}).get("budget_ratio", 0.1)
        self.budget = max(1, int(ratio * max_ctx))

        # One (k_cache, v_cache) tuple per layer; None until first update.
        # Shapes: [num_heads, current_len, head_dim]
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
        self.metadata: Dict = {}

    @abstractmethod
    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states and apply the eviction/compression policy.

        Args:
            layer_idx: Transformer layer index.
            key_states:       [batch, num_heads, seq_len, head_dim]
            value_states:     [batch, num_heads, seq_len, head_dim]
            attention_weights:[batch, num_heads, seq_len, cache_len]  (optional)

        Returns:
            (key_cache, value_cache) both [batch, num_heads, cache_len, head_dim]
        """

    @abstractmethod
    def evict(self, layer_idx: int) -> None:
        """Trim the cache for layer_idx down to at most self.budget entries."""

    def reset(self) -> None:
        """Clear all cached state (call between independent sequences)."""
        self.cache = [None] * self.num_layers
        self.metadata.clear()

    def get_cache_size(self) -> int:
        """Return current total cache storage in bytes."""
        total = 0
        for layer_cache in self.cache:
            if layer_cache is not None:
                k, v = layer_cache
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        return total

    def get_current_len(self, layer_idx: int) -> int:
        """Return the number of cached tokens for a specific layer."""
        if self.cache[layer_idx] is None:
            return 0
        return self.cache[layer_idx][0].shape[1]

    def get_stats(self) -> Dict:
        """Return a dict of cache statistics for benchmarking."""
        total_entries = sum(
            self.cache[i][0].shape[1] if self.cache[i] is not None else 0
            for i in range(self.num_layers)
        )
        return {
            "cache_size_bytes": self.get_cache_size(),
            "total_entries": total_entries,
            "budget": self.budget,
            "num_layers": self.num_layers,
            "method": self.__class__.__name__,
        }
