"""AdaptKV: learned three-tier adaptive KV cache (our method).

Instead of the binary keep-or-evict decision used by H2O / SnapKV / StreamingLLM,
AdaptKV uses a small policy MLP to assign each cached entry to one of three tiers:

    Tier 0 — KEEP      : stored at full FP16 precision
    Tier 1 — COMPRESS  : quantized to 4-bit NF4 (4× smaller) and kept
    Tier 2 — EVICT     : discarded

The policy runs once per generation step and operates per-head per-layer.
Total retained entries (FP16 + compressed) never exceeds self.budget.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from .base_cache import BaseKVCache
from .quantization import quantize_to_nf4, dequantize_from_nf4

logger = logging.getLogger(__name__)


class _CompressedEntry:
    """Lightweight wrapper for a single NF4-compressed (key, value) pair."""

    __slots__ = ("k_packed", "k_scales", "k_cb", "k_shape", "k_dtype",
                 "v_packed", "v_scales", "v_cb", "v_shape", "v_dtype")

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        # k, v: [head_dim]  (single entry, single head — already indexed)
        self.k_shape, self.k_dtype = k.shape, k.dtype
        self.v_shape, self.v_dtype = v.shape, v.dtype
        self.k_packed, self.k_scales, self.k_cb = quantize_to_nf4(k)
        self.v_packed, self.v_scales, self.v_cb = quantize_to_nf4(v)

    def decompress(self) -> Tuple[torch.Tensor, torch.Tensor]:
        k = dequantize_from_nf4(self.k_packed, self.k_scales, self.k_cb,
                                 self.k_shape, self.k_dtype)
        v = dequantize_from_nf4(self.v_packed, self.v_scales, self.v_cb,
                                 self.v_shape, self.v_dtype)
        return k, v


class AdaptKVCache(BaseKVCache):
    """Three-tier adaptive KV cache driven by a learned policy MLP.

    The policy network must be set via :meth:`set_policy` before the first
    update call, otherwise the cache falls back to H2O-style eviction using
    cumulative attention scores.
    """

    def __init__(self, config: Dict, num_layers: int, num_heads: int,
                 head_dim: int, device: str = "cpu"):
        super().__init__(config, num_layers, num_heads, head_dim, device)

        policy_cfg = config.get("policy", {})
        self.compress_ratio = 0.3          # Fraction of budget for compressed tier

        # Policy network (set externally via set_policy)
        self._policy = None
        self._feature_extractor = None

        # Per-layer compressed tier: list[layer] → list of _CompressedEntry per head
        # Stored as dict: {token_idx: _CompressedEntry}  per head per layer
        self._compressed: List[Optional[List[Dict[int, _CompressedEntry]]]] = [
            None
        ] * num_layers

        # Cumulative attention scores for fallback H2O-style ranking
        self._cumulative_scores: List[Optional[torch.Tensor]] = [None] * num_layers

        # Count total entries (FP16 + compressed) per layer
        self._compressed_count: List[int] = [0] * num_layers

    def set_policy(self, policy, feature_extractor) -> None:
        """Attach the trained policy MLP and feature extractor."""
        self._policy = policy
        self._feature_extractor = feature_extractor
        logger.info("AdaptKV policy network attached.")

    # ------------------------------------------------------------------ #
    # BaseKVCache interface                                                #
    # ------------------------------------------------------------------ #

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new KV states and run the tier-assignment policy.

        Returns the FP16 cache + on-the-fly dequantized compressed entries.
        """
        new_k = key_states.squeeze(0)   # [H, S, D]
        new_v = value_states.squeeze(0)

        # --- Append to FP16 cache ---
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
            self._compressed[layer_idx] = [{} for _ in range(self.num_heads)]
            h = new_k.shape[0]
            self._cumulative_scores[layer_idx] = torch.zeros(
                h, new_k.shape[1], device=self.device
            )
        else:
            k_prev, v_prev = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([k_prev, new_k], dim=1),
                torch.cat([v_prev, new_v], dim=1),
            )
            new_zeros = torch.zeros(
                new_k.shape[0], new_k.shape[1], device=self.device
            )
            self._cumulative_scores[layer_idx] = torch.cat(
                [self._cumulative_scores[layer_idx], new_zeros], dim=1
            )

        # --- Update attention scores ---
        if attention_weights is not None:
            step_scores = attention_weights.squeeze(0).sum(dim=-2)  # [H, total]
            prev_len = self._cumulative_scores[layer_idx].shape[1] - new_k.shape[1]
            self._cumulative_scores[layer_idx][:, :prev_len] += step_scores[:, :prev_len]

            if self._feature_extractor is not None:
                self._feature_extractor.update_attention_stats(layer_idx, attention_weights)

        # --- Run eviction if over budget ---
        fp16_len = self.cache[layer_idx][0].shape[1]
        comp_count = self._compressed_count[layer_idx]
        if fp16_len + comp_count > self.budget:
            self.evict(layer_idx)

        # --- Build combined cache for attention ---
        return self._build_combined_cache(layer_idx)

    def evict(self, layer_idx: int) -> None:
        """Apply three-tier policy to trim cache to budget."""
        k_cache, v_cache = self.cache[layer_idx]
        H, current_len, D = k_cache.shape
        comp_count = self._compressed_count[layer_idx]
        total = current_len + comp_count

        if total <= self.budget:
            return

        if self._policy is not None and self._feature_extractor is not None:
            self._policy_evict(layer_idx, k_cache, v_cache)
        else:
            self._fallback_evict(layer_idx, k_cache, v_cache)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _policy_evict(self, layer_idx: int, k_cache: torch.Tensor,
                      v_cache: torch.Tensor) -> None:
        """Use the policy MLP to decide keep / compress / evict."""
        features, head_type_ids = self._feature_extractor.extract_features(
            layer_idx, k_cache
        )
        features = features.to(self.device)
        head_type_ids = head_type_ids.to(self.device)

        decisions = self._policy.decide_with_budget_constraint(
            features, head_type_ids,
            budget=self.budget,
            compress_ratio=self.compress_ratio,
        )  # [1, H * current_len]

        H, current_len, D = k_cache.shape
        decisions_2d = decisions.view(H, current_len)  # [H, current_len]

        self._apply_tier_decisions(layer_idx, k_cache, v_cache, decisions_2d)

    def _fallback_evict(self, layer_idx: int, k_cache: torch.Tensor,
                        v_cache: torch.Tensor) -> None:
        """H2O-style fallback when no policy is loaded."""
        H, CL, D = k_cache.shape
        scores = self._cumulative_scores[layer_idx]  # [H, CL]

        n_keep = max(1, int(self.budget * (1 - self.compress_ratio)))
        n_compress = max(0, self.budget - n_keep)

        new_k_list, new_v_list = [], []
        new_comp = [{} for _ in range(H)]

        for h in range(H):
            s = scores[h]
            sorted_idx = s.argsort(descending=True)
            keep_idx  = sorted_idx[:n_keep]
            comp_idx  = sorted_idx[n_keep: n_keep + n_compress]

            keep_idx_sorted, _ = keep_idx.sort()
            new_k_list.append(k_cache[h, keep_idx_sorted, :])
            new_v_list.append(v_cache[h, keep_idx_sorted, :])

            for i, ci in enumerate(comp_idx.tolist()):
                entry = _CompressedEntry(k_cache[h, ci, :], v_cache[h, ci, :])
                new_comp[h][i] = entry

        self.cache[layer_idx] = (torch.stack(new_k_list), torch.stack(new_v_list))
        self._compressed[layer_idx] = new_comp
        self._compressed_count[layer_idx] = n_compress
        self._cumulative_scores[layer_idx] = scores[:, sorted_idx[:n_keep].sort().values]

    def _apply_tier_decisions(
        self,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        decisions: torch.Tensor,
    ) -> None:
        """Execute 0=keep / 1=compress / 2=evict decisions.

        decisions: [H, current_len]
        """
        H, CL, D = k_cache.shape
        KEEP, COMPRESS, EVICT = 0, 1, 2

        new_k_list, new_v_list = [], []
        new_comp = [{} for _ in range(H)]
        total_compressed = 0

        for h in range(H):
            keep_mask    = decisions[h] == KEEP
            compress_mask = decisions[h] == COMPRESS

            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            comp_idx = compress_mask.nonzero(as_tuple=True)[0]

            new_k_list.append(k_cache[h, keep_idx, :])
            new_v_list.append(v_cache[h, keep_idx, :])

            for i, ci in enumerate(comp_idx.tolist()):
                entry = _CompressedEntry(k_cache[h, ci, :], v_cache[h, ci, :])
                new_comp[h][i] = entry
            total_compressed = max(total_compressed, len(comp_idx))

        # Pad keep lists to same length so we can stack
        max_keep = max(t.shape[0] for t in new_k_list) if new_k_list else 0
        padded_k = []
        padded_v = []
        for t_k, t_v in zip(new_k_list, new_v_list):
            pad_len = max_keep - t_k.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, D, device=self.device, dtype=t_k.dtype)
                t_k = torch.cat([t_k, pad], dim=0)
                t_v = torch.cat([t_v, pad], dim=0)
            padded_k.append(t_k)
            padded_v.append(t_v)

        if padded_k:
            self.cache[layer_idx] = (torch.stack(padded_k), torch.stack(padded_v))
        else:
            empty = torch.zeros(H, 0, D, device=self.device)
            self.cache[layer_idx] = (empty, empty)

        self._compressed[layer_idx] = new_comp
        self._compressed_count[layer_idx] = total_compressed

    def _build_combined_cache(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize compressed entries and concatenate with FP16 cache."""
        k_fp16, v_fp16 = self.cache[layer_idx]     # [H, fp16_len, D]
        comp_per_head = self._compressed[layer_idx]
        H, _, D = k_fp16.shape

        if not any(comp_per_head):
            return k_fp16.unsqueeze(0), v_fp16.unsqueeze(0)

        comp_count = max(len(c) for c in comp_per_head)
        if comp_count == 0:
            return k_fp16.unsqueeze(0), v_fp16.unsqueeze(0)

        # Dequantize compressed entries: [H, comp_count, D]
        all_ck = torch.zeros(H, comp_count, D, device=self.device, dtype=k_fp16.dtype)
        all_cv = torch.zeros(H, comp_count, D, device=self.device, dtype=v_fp16.dtype)

        for h in range(H):
            for i, entry in comp_per_head[h].items():
                ck, cv = entry.decompress()
                all_ck[h, i, :] = ck.to(self.device)
                all_cv[h, i, :] = cv.to(self.device)

        combined_k = torch.cat([k_fp16, all_ck], dim=1)
        combined_v = torch.cat([v_fp16, all_cv], dim=1)
        return combined_k.unsqueeze(0), combined_v.unsqueeze(0)

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats["compressed_entries"] = sum(self._compressed_count)
        fp16_entries = sum(
            self.cache[i][0].shape[1] if self.cache[i] is not None else 0
            for i in range(self.num_layers)
        )
        stats["fp16_entries"] = fp16_entries
        stats["total_retained"] = fp16_entries + stats["compressed_entries"]
        return stats

    def reset(self) -> None:
        super().reset()
        self._compressed = [None] * self.num_layers
        self._cumulative_scores = [None] * self.num_layers
        self._compressed_count = [0] * self.num_layers


# ── Functional API used by experiments ────────────────────────────────────────

# Memory-equivalence constants (budget_ratio=0.10):
#   H2O:     0.10 × S × 2B         = 0.20 × S bytes
#   AdaptKV: 0.04 × S × 2B
#          + 0.24 × S × 0.5B       = 0.08 + 0.12 = 0.20 × S bytes  ✓
FP16_RATIO      = 0.4   # fraction of budget stored at FP16
COMPRESS_RATIO  = 2.4   # fraction of budget stored at INT4/compressed
COMPRESS_QUALITY = 0.95  # approximate attention-mass retention after compression


def adaptkv_select_tokens(
    importance: torch.Tensor,
    budget_k: int,
    fp16_ratio: float = FP16_RATIO,
    compress_ratio: float = COMPRESS_RATIO,
) -> tuple:
    """Select token indices for FP16 and compressed tiers.

    Args:
        importance: [S] per-token importance scores (higher = more important)
        budget_k:   total number of budget slots
        fp16_ratio: fraction of budget_k kept at FP16
        compress_ratio: fraction of budget_k kept compressed (INT4)

    Returns:
        (fp16_indices, compressed_indices)  — lists of int token indices
    """
    fp16_k = max(1, int(budget_k * fp16_ratio))
    int4_k = max(1, int(budget_k * compress_ratio))

    sorted_idx = torch.argsort(importance, descending=True)
    fp16_idx  = sorted_idx[:fp16_k].tolist()
    comp_idx  = sorted_idx[fp16_k: fp16_k + int4_k].tolist()
    return fp16_idx, comp_idx
