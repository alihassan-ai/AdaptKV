"""StreamingLLM: attention sink + sliding window KV cache.

Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
ICLR 2024.

Always keep the first `sink_tokens` tokens (attention sinks that stabilize
softmax), plus a sliding window of the most recently seen tokens.
"""

from typing import Dict, Optional, Tuple

import torch

from .base_cache import BaseKVCache


class StreamingLLMCache(BaseKVCache):
    """Attention sink + sliding window eviction policy."""

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        super().__init__(config, num_layers, num_heads, head_dim, device)
        cache_cfg = config.get("cache", {})
        self.sink_tokens = cache_cfg.get("sink_tokens", 4)
        # Window size = budget minus sink slots (at minimum 1)
        self.window_size = max(1, self.budget - self.sink_tokens)

        # Track how many sink tokens have been stored per layer
        self._sinks_stored: list = [0] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV, then trim to sink + window budget.

        Args:
            layer_idx:         Transformer layer index.
            key_states:        [batch, num_heads, seq_len, head_dim]
            value_states:      [batch, num_heads, seq_len, head_dim]
            attention_weights: ignored (StreamingLLM uses positional policy).

        Returns:
            (key_cache, val_cache) [batch, num_heads, cache_len, head_dim]
        """
        new_k = key_states.squeeze(0)   # [H, S, D]
        new_v = value_states.squeeze(0)
        new_len = new_k.shape[1]

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
            # The first tokens entering are the sinks
            self._sinks_stored[layer_idx] = min(new_len, self.sink_tokens)
        else:
            k_prev, v_prev = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([k_prev, new_k], dim=1),
                torch.cat([v_prev, new_v], dim=1),
            )

        # Apply sliding-window eviction if over budget
        if self.cache[layer_idx][0].shape[1] > self.budget:
            self.evict(layer_idx)

        k_out, v_out = self.cache[layer_idx]
        return k_out.unsqueeze(0), v_out.unsqueeze(0)

    def evict(self, layer_idx: int) -> None:
        """Keep first `sink_tokens` + last `window_size` tokens; drop middle."""
        k_cache, v_cache = self.cache[layer_idx]
        current_len = k_cache.shape[1]

        if current_len <= self.budget:
            return

        n_sinks = self._sinks_stored[layer_idx]
        # Keep sinks from the front, window from the end
        sink_k = k_cache[:, :n_sinks, :]
        sink_v = v_cache[:, :n_sinks, :]

        window_k = k_cache[:, -self.window_size:, :]
        window_v = v_cache[:, -self.window_size:, :]

        self.cache[layer_idx] = (
            torch.cat([sink_k, window_k], dim=1),
            torch.cat([sink_v, window_v], dim=1),
        )

    def reset(self) -> None:
        super().reset()
        self._sinks_stored = [0] * self.num_layers
