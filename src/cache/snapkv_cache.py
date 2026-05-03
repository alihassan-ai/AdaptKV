"""SnapKV: observation-window guided KV cache compression.

Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation",
arXiv 2024.

During prefill (long prompt), SnapKV examines the attention patterns of the
*last* `observation_window` tokens to identify which prior positions are
important per attention head.  Those positions are retained; everything else
is evicted.  During decode the selected cache is fixed (no further eviction).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base_cache import BaseKVCache


class SnapKVCache(BaseKVCache):
    """Observation-window driven prefill compression with frozen decode cache."""

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        super().__init__(config, num_layers, num_heads, head_dim, device)
        cache_cfg = config.get("cache", {})
        self.observation_window = cache_cfg.get("recent_window", 64)

        # How many positions to keep per head (excluding observation window itself)
        #   budget = selected_positions + observation_window
        self.max_selected = max(1, self.budget - self.observation_window)

        # Track whether prefill compression has already been applied per layer
        self._compressed: List[bool] = [False] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV; apply SnapKV compression on first large prefill.

        SnapKV applies compression once when the cache first exceeds the budget
        (i.e. at the end of prompt prefill).  Subsequent decode steps just
        append without any eviction.
        """
        new_k = key_states.squeeze(0)   # [H, S, D]
        new_v = value_states.squeeze(0)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
        else:
            k_prev, v_prev = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([k_prev, new_k], dim=1),
                torch.cat([v_prev, new_v], dim=1),
            )

        # Apply SnapKV compression once when cache exceeds budget after prefill
        current_len = self.cache[layer_idx][0].shape[1]
        if (not self._compressed[layer_idx]
                and current_len > self.budget
                and attention_weights is not None):
            self._apply_snapkv_compression(layer_idx, attention_weights)
            self._compressed[layer_idx] = True

        k_out, v_out = self.cache[layer_idx]
        return k_out.unsqueeze(0), v_out.unsqueeze(0)

    def _apply_snapkv_compression(
        self, layer_idx: int, attention_weights: torch.Tensor
    ) -> None:
        """Select important positions using the observation window's attention.

        attention_weights: [1, num_heads, obs_win_len, total_cache_len]
        """
        k_cache, v_cache = self.cache[layer_idx]
        total_len = k_cache.shape[1]
        obs_win = min(self.observation_window, total_len)

        # Use attention weights from the observation window queries only
        # attn shape: [1, H, obs_win, total_len]  → squeeze batch
        attn = attention_weights.squeeze(0)  # [H, obs_win, total_len]

        # Average pool attention scores across the observation window queries
        pooled = attn.mean(dim=1)  # [H, total_len]

        # The observation window itself is always retained
        obs_start = total_len - obs_win
        num_heads = k_cache.shape[0]

        new_k_list, new_v_list = [], []
        for h in range(num_heads):
            head_scores = pooled[h]  # [total_len]

            # Score only the non-observation-window portion
            pre_scores = head_scores[:obs_start]
            n_select = min(self.max_selected, obs_start)

            if obs_start > 0 and n_select > 0:
                _, selected_idx = torch.topk(pre_scores, n_select)
                selected_idx, _ = torch.sort(selected_idx)
            else:
                selected_idx = torch.empty(0, dtype=torch.long, device=self.device)

            # Observation window indices
            obs_idx = torch.arange(obs_start, total_len, device=self.device)

            keep_idx = torch.cat([selected_idx, obs_idx])
            new_k_list.append(k_cache[h, keep_idx, :])
            new_v_list.append(v_cache[h, keep_idx, :])

        self.cache[layer_idx] = (torch.stack(new_k_list), torch.stack(new_v_list))

    def evict(self, layer_idx: int) -> None:
        """SnapKV does not evict incrementally during decode — no-op."""

    def reset(self) -> None:
        super().reset()
        self._compressed = [False] * self.num_layers
