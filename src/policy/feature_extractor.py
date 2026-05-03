"""Extract the 5 input features for the AdaptKV policy network.

Features per cached KV entry:
    0: rolling attention momentum  (EWMA, alpha=0.1, window=attention_window)
    1: attention variance          (var over the window)
    2: key embedding redundancy    (max cosine similarity with other keys)
    3: relative position           (position_index / total_len)
    4: head type id                (0=local, 1=global, 2=sink)  ← returned separately
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class FeatureExtractor:
    """Tracks rolling attention statistics and computes per-entry features."""

    HEAD_LOCAL  = 0
    HEAD_GLOBAL = 1
    HEAD_SINK   = 2

    def __init__(self, config: Dict, num_layers: int, num_heads: int, device: str = "cpu"):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        policy_cfg = config.get("policy", {})
        self.alpha = 0.1                             # EWMA decay
        self.window = policy_cfg.get("attention_window", 32)

        # Per-layer, per-head EWMA momentum: list of [num_heads, cache_len] tensors
        self._ewma: List[Optional[torch.Tensor]] = [None] * num_layers
        # Sliding window of raw attention scores for variance computation
        # Stored as list[layer][list of [num_heads, 1, cache_len] per step]
        self._attn_history: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

        # Head-type assignments (set via classify_head_types)
        self._head_types: Optional[torch.Tensor] = None  # [num_heads]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def classify_head_types(
        self,
        calibration_attentions: List[torch.Tensor],
    ) -> torch.Tensor:
        """Assign each attention head to local / global / sink category.

        Classification is based on the entropy and concentration of attention
        weights measured on a small calibration set.

        Args:
            calibration_attentions: list of tensors [num_heads, seq_len, seq_len]
                from a few calibration forward passes.

        Returns:
            head_types: [num_heads] int tensor (0=local, 1=global, 2=sink)
        """
        # Average attention matrix over calibration samples
        avg_attn = torch.stack(calibration_attentions).mean(0)  # [H, T, T]
        H, T, _ = avg_attn.shape

        # Per-head attention entropy (high = global, low = local or sink)
        entropy = -(avg_attn * (avg_attn + 1e-9).log()).sum(dim=-1).mean(dim=-1)  # [H]
        max_entropy = torch.log(torch.tensor(float(T), device=entropy.device))

        # Attention mass on first tokens (sink indicator)
        sink_mass = avg_attn[:, :, :4].sum(dim=-1).mean(dim=-1)  # [H] mean over queries

        head_types = torch.zeros(H, dtype=torch.long, device=self.device)

        entropy_thresh_high = 0.6 * max_entropy
        sink_thresh = 0.4

        for h in range(H):
            if entropy[h] > entropy_thresh_high:
                head_types[h] = self.HEAD_GLOBAL
            elif sink_mass[h] > sink_thresh:
                head_types[h] = self.HEAD_SINK
            else:
                head_types[h] = self.HEAD_LOCAL

        self._head_types = head_types
        return head_types

    def update_attention_stats(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
    ) -> None:
        """Record attention weights for a single generation step.

        Args:
            attention_weights: [1, num_heads, 1, cache_len]  (single decode step)
        """
        # Squeeze to [num_heads, cache_len]; handle both prefill and decode shapes
        attn_sq = attention_weights.squeeze(0)   # [H, Q, CL]
        attn = attn_sq.mean(dim=-2)              # [H, CL]  — avg over query positions
        cache_len = attn.shape[1]

        prev = self._ewma[layer_idx]
        if prev is None:
            self._ewma[layer_idx] = attn.clone()
        elif prev.shape[1] == cache_len:
            self._ewma[layer_idx] = self.alpha * attn + (1 - self.alpha) * prev
        elif cache_len > prev.shape[1]:
            # Cache grew (new tokens appended) — update existing, init new entries
            old_len = prev.shape[1]
            updated_old = self.alpha * attn[:, :old_len] + (1 - self.alpha) * prev
            new_entries = attn[:, old_len:].clone()
            self._ewma[layer_idx] = torch.cat([updated_old, new_entries], dim=1)
        else:
            # Cache shrank (eviction occurred) — reinitialize at new length
            self._ewma[layer_idx] = attn.clone()

        # Maintain a rolling window for variance (only store if shape is stable)
        self._attn_history[layer_idx].append(attn.unsqueeze(1))  # [H, 1, CL]
        if len(self._attn_history[layer_idx]) > self.window:
            self._attn_history[layer_idx].pop(0)

    def extract_features(
        self,
        layer_idx: int,
        key_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the 4 continuous features + head-type ids for all cached entries.

        Args:
            layer_idx: Layer index.
            key_cache: [num_heads, cache_len, head_dim] — current FP16 key cache.

        Returns:
            features:      [1, num_heads * cache_len, 4]  float features
            head_type_ids: [1, num_heads * cache_len]     long head-type ids
        """
        H, cache_len, D = key_cache.shape

        # Feature 0 & 1: momentum and variance from attention history
        momentum = self._compute_momentum(layer_idx, cache_len, H)   # [H, CL]
        variance = self._compute_variance(layer_idx, cache_len, H)   # [H, CL]

        # Feature 2: key redundancy (max cosine sim with any other key)
        redundancy = self._compute_key_redundancy(key_cache)          # [H, CL]

        # Feature 3: relative position
        positions = torch.arange(cache_len, device=self.device, dtype=torch.float32)
        rel_pos = positions / max(cache_len - 1, 1)                   # [CL]
        rel_pos = rel_pos.unsqueeze(0).expand(H, -1)                  # [H, CL]

        # Stack: [H, CL, 4]
        feat = torch.stack([momentum, variance, redundancy, rel_pos], dim=-1)

        # Flatten heads × positions into a single "entries" dimension
        feat_flat = feat.view(1, H * cache_len, 4)                    # [1, H*CL, 4]

        # Head-type ids: repeat cache_len times for each head
        if self._head_types is not None:
            ht = self._head_types[:H]
        else:
            ht = torch.zeros(H, dtype=torch.long, device=self.device)
        ht_expanded = ht.unsqueeze(1).expand(H, cache_len).reshape(1, -1)  # [1, H*CL]

        return feat_flat, ht_expanded

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_momentum(self, layer_idx: int, cache_len: int, H: int) -> torch.Tensor:
        """Return EWMA attention momentum per head per position."""
        if (self._ewma[layer_idx] is not None
                and self._ewma[layer_idx].shape[1] >= cache_len):
            return self._ewma[layer_idx][:H, :cache_len].to(self.device)
        return torch.zeros(H, cache_len, device=self.device)

    def _compute_variance(self, layer_idx: int, cache_len: int, H: int) -> torch.Tensor:
        """Return attention variance per head per position over the window."""
        history = self._attn_history[layer_idx]
        if len(history) < 2:
            return torch.zeros(H, cache_len, device=self.device)

        # Stack history: [W, H, CL]  (some steps may have smaller CL if expanding)
        valid = [s[:H, :, :cache_len] for s in history if s.shape[0] >= H]
        valid = [s for s in valid if s.shape[2] == cache_len]
        if len(valid) < 2:
            return torch.zeros(H, cache_len, device=self.device)

        stacked = torch.cat(valid, dim=1)     # [H, W, CL]
        return stacked.var(dim=1).clamp(min=0.0)  # [H, CL]

    def _compute_key_redundancy(self, key_cache: torch.Tensor) -> torch.Tensor:
        """For each key, compute max cosine similarity with all other keys in the same head.

        Uses batched matrix multiplication for efficiency.

        Args:
            key_cache: [H, CL, D]

        Returns:
            [H, CL] max cosine similarity (excluding self)
        """
        H, CL, D = key_cache.shape
        if CL <= 1:
            return torch.zeros(H, CL, device=self.device)

        # Normalize keys: [H, CL, D]
        normed = F.normalize(key_cache.float(), p=2, dim=-1)
        # Cosine similarity matrix: [H, CL, CL]
        sim = torch.bmm(normed, normed.transpose(1, 2))

        # Mask diagonal (self-similarity = 1.0)
        eye = torch.eye(CL, device=self.device, dtype=torch.bool).unsqueeze(0)
        sim.masked_fill_(eye, -1.0)

        # Max similarity with any other key
        max_sim, _ = sim.max(dim=-1)   # [H, CL]
        return max_sim.clamp(0.0, 1.0).to(key_cache.dtype)

    def reset_layer(self, layer_idx: int) -> None:
        self._ewma[layer_idx] = None
        self._attn_history[layer_idx] = []

    def reset(self) -> None:
        for i in range(self.num_layers):
            self.reset_layer(i)
