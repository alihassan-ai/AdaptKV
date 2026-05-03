"""Full KV cache baseline — no eviction or compression."""

from typing import Dict, Optional, Tuple

import torch

from .base_cache import BaseKVCache


class FullCache(BaseKVCache):
    """Stores the entire KV history without any eviction.

    This is the oracle baseline: highest quality, highest memory cost.
    Used to measure the quality ceiling and to generate teacher labels for
    policy distillation.
    """

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values without any eviction.

        Args:
            layer_idx: Transformer layer index.
            key_states:   [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            attention_weights: ignored (full cache does no eviction).

        Returns:
            (key_cache, val_cache) [batch, num_heads, total_len, head_dim]
        """
        # Squeeze batch dimension: store as [num_heads, total_len, head_dim]
        new_k = key_states.squeeze(0)
        new_v = value_states.squeeze(0)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
        else:
            k_prev, v_prev = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([k_prev, new_k], dim=1),
                torch.cat([v_prev, new_v], dim=1),
            )

        k_out, v_out = self.cache[layer_idx]
        return k_out.unsqueeze(0), v_out.unsqueeze(0)

    def evict(self, layer_idx: int) -> None:
        """No-op: full cache never evicts."""
