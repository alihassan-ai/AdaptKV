"""H2O (Heavy Hitter Oracle) KV cache eviction.

Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models", NeurIPS 2023.

Strategy: rank cached tokens by their cumulative attention score (sum of all
attention weights received across decoding steps). Keep the top-k heavy
hitters plus the most-recent `recent_window` tokens.
"""

from typing import Dict, Optional, Tuple

import torch

from .base_cache import BaseKVCache


class H2OCache(BaseKVCache):
    """Heavy-Hitter Oracle eviction policy."""

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        super().__init__(config, num_layers, num_heads, head_dim, device)
        self.recent_window = config.get("cache", {}).get("recent_window", 128)
        self.heavy_hitter_ratio = config.get("cache", {}).get("heavy_hitter_ratio", 0.5)

        # cumulative_scores[l]: [num_heads, current_len]  running attention sum
        self.cumulative_scores: list = [None] * num_layers

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append KV states, update cumulative scores, evict if over budget.

        attention_weights shape: [batch, num_heads, new_seq_len, total_cache_len]
        The attention weights over the *new* query tokens tell us how much
        attention each *cached* token received in this step — that contribution
        is summed and added to the running cumulative score.
        """
        new_k = key_states.squeeze(0)   # [H, S, D]
        new_v = value_states.squeeze(0)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
            if attention_weights is not None:
                # Sum over the query dimension to get per-key score
                scores = attention_weights.squeeze(0).sum(dim=-2)  # [H, S]
                self.cumulative_scores[layer_idx] = scores
            else:
                seq_len = new_k.shape[1]
                self.cumulative_scores[layer_idx] = torch.zeros(
                    self.num_heads, seq_len, device=self.device
                )
        else:
            k_prev, v_prev = self.cache[layer_idx]
            prev_len = k_prev.shape[1]

            # Extend cache
            combined_k = torch.cat([k_prev, new_k], dim=1)
            combined_v = torch.cat([v_prev, new_v], dim=1)
            self.cache[layer_idx] = (combined_k, combined_v)

            # Update scores
            if attention_weights is not None:
                # attention_weights: [1, H, new_S, prev_len + new_S]
                step_scores = attention_weights.squeeze(0).sum(dim=-2)  # [H, total]
                old_contrib = step_scores[:, :prev_len]
                new_contrib = step_scores[:, prev_len:]
                updated = torch.cat([
                    self.cumulative_scores[layer_idx] + old_contrib,
                    new_contrib,
                ], dim=1)
                self.cumulative_scores[layer_idx] = updated
            else:
                new_zeros = torch.zeros(
                    self.num_heads, new_k.shape[1], device=self.device
                )
                self.cumulative_scores[layer_idx] = torch.cat(
                    [self.cumulative_scores[layer_idx], new_zeros], dim=1
                )

        # Evict if we exceed the budget
        if self.cache[layer_idx][0].shape[1] > self.budget:
            self.evict(layer_idx)

        k_out, v_out = self.cache[layer_idx]
        return k_out.unsqueeze(0), v_out.unsqueeze(0)

    def evict(self, layer_idx: int) -> None:
        """Keep top heavy-hitters + the most recent tokens; evict the rest."""
        k_cache, v_cache = self.cache[layer_idx]
        scores = self.cumulative_scores[layer_idx]
        current_len = k_cache.shape[1]

        if current_len <= self.budget:
            return

        num_heads = k_cache.shape[0]
        recent_k = min(self.recent_window, self.budget)
        heavy_k = self.budget - recent_k

        kept_k_list, kept_v_list, kept_s_list = [], [], []

        for h in range(num_heads):
            # Indices of the most recent tokens (always preserved)
            recent_start = current_len - recent_k
            recent_idx = torch.arange(recent_start, current_len, device=self.device)

            # From non-recent tokens, pick the top heavy_k by cumulative score
            non_recent_len = recent_start
            if non_recent_len > 0 and heavy_k > 0:
                non_recent_scores = scores[h, :non_recent_len]
                topk = min(heavy_k, non_recent_len)
                _, hh_idx = torch.topk(non_recent_scores, topk)
                hh_idx, _ = torch.sort(hh_idx)
                all_idx = torch.cat([hh_idx, recent_idx])
            else:
                all_idx = recent_idx

            all_idx, _ = torch.sort(all_idx)
            kept_k_list.append(k_cache[h, all_idx, :])
            kept_v_list.append(v_cache[h, all_idx, :])
            kept_s_list.append(scores[h, all_idx])

        self.cache[layer_idx] = (torch.stack(kept_k_list), torch.stack(kept_v_list))
        self.cumulative_scores[layer_idx] = torch.stack(kept_s_list)

    def reset(self) -> None:
        super().reset()
        self.cumulative_scores = [None] * self.num_layers


# ── Functional API used by experiments ────────────────────────────────────────

def h2o_select_tokens(
    importance: torch.Tensor,
    budget_k: int,
    recent_k: int = 0,
) -> list:
    """Select token indices using H2O's heavy-hitter + recent-window rule.

    Args:
        importance: [S] per-token importance (cumulative attention score)
        budget_k:   total tokens to keep
        recent_k:   how many of the most-recent positions are always kept

    Returns:
        Sorted list of kept token indices.
    """
    S = importance.shape[0]
    recent_k = min(recent_k, S)
    heavy_k  = max(0, budget_k - recent_k)

    recent_idx = set(range(S - recent_k, S))

    non_recent = torch.tensor(
        [i for i in range(S) if i not in recent_idx],
        dtype=torch.long,
    )
    if heavy_k > 0 and non_recent.numel() > 0:
        topk = min(heavy_k, non_recent.numel())
        scores = importance[non_recent]
        _, top_pos = torch.topk(scores, topk)
        heavy_idx = set(non_recent[top_pos].tolist())
    else:
        heavy_idx = set()

    kept = sorted(heavy_idx | recent_idx)
    return kept
