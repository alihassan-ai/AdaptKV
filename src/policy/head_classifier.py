"""Classify transformer attention heads into local / global / sink categories
based on empirical attention entropy on a calibration set.

Head types:
    HEAD_LOCAL  (0) — attend densely to nearby tokens (low entropy)
    HEAD_GLOBAL (1) — attend broadly across the sequence (high entropy)
    HEAD_SINK   (2) — attend heavily to the first 1-4 tokens (attention sink)
"""

from typing import Dict, List, Tuple

import torch

HEAD_LOCAL  = 0
HEAD_GLOBAL = 1
HEAD_SINK   = 2

GLOBAL_ENTROPY_FRAC = 0.60   # head qualifies as global if entropy > 60% of max
SINK_FIRST_ATTN_THR = 0.30   # head qualifies as sink if first-4 mass > 30%


def classify_heads(
    calibration_attns: List[torch.Tensor],
    entropy_global_frac: float = GLOBAL_ENTROPY_FRAC,
    sink_threshold: float = SINK_FIRST_ATTN_THR,
) -> Tuple[torch.Tensor, dict]:
    """Classify every (layer, head) into local/global/sink.

    Args:
        calibration_attns: list of [num_heads, seq_len, seq_len] attention
            matrices, one per layer, averaged across calibration prompts.

    Returns:
        head_types: LongTensor [num_layers, num_heads]  (values 0/1/2)
        stats:      dict with per-type counts and diagnostics
    """
    num_layers = len(calibration_attns)
    num_heads  = calibration_attns[0].shape[0]
    head_types = torch.zeros(num_layers, num_heads, dtype=torch.long)

    all_entropies: List[List[float]] = []
    all_sink_mass: List[List[float]] = []

    for l, attn in enumerate(calibration_attns):
        S = attn.shape[-1]

        # Entropy averaged across query positions: [H]
        log_attn = (attn + 1e-10).log()
        entropy  = -(attn * log_attn).sum(dim=-1).mean(dim=-1)
        max_ent  = torch.tensor(float(S)).log().clamp(min=1e-6)

        # Fraction of attention mass on first 4 tokens (sink signal): [H]
        sink_mass = attn[:, :, :min(4, S)].sum(dim=-1).mean(dim=-1)

        for h in range(num_heads):
            e_frac = (entropy[h] / max_ent).item()
            s_mass = sink_mass[h].item()
            if e_frac > entropy_global_frac:
                head_types[l, h] = HEAD_GLOBAL
            elif s_mass > sink_threshold:
                head_types[l, h] = HEAD_SINK
            else:
                head_types[l, h] = HEAD_LOCAL

        all_entropies.append(entropy.tolist())
        all_sink_mass.append(sink_mass.tolist())

    flat = head_types.reshape(-1).tolist()
    counts: Dict = {
        "local":      flat.count(HEAD_LOCAL),
        "global":     flat.count(HEAD_GLOBAL),
        "sink":       flat.count(HEAD_SINK),
        "total":      len(flat),
    }
    counts["local_pct"]  = 100 * counts["local"]  / max(counts["total"], 1)
    counts["global_pct"] = 100 * counts["global"] / max(counts["total"], 1)
    counts["sink_pct"]   = 100 * counts["sink"]   / max(counts["total"], 1)

    return head_types, {
        "counts":     counts,
        "entropies":  all_entropies,
        "sink_mass":  all_sink_mass,
    }


class HeadClassifier:
    """Stateful head classifier with per-head-type AdaptKV budget parameters."""

    # Each head type must retain more total tokens than the uniform baseline
    # (uniform: fp16=0.4 + compress=2.5 = 2.9× budget) to prove per-head wins.
    TYPE_PARAMS = {
        HEAD_LOCAL:  {"recent_frac": 0.80, "fp16_ratio": 0.55, "compress_ratio": 2.5},
        HEAD_GLOBAL: {"recent_frac": 0.20, "fp16_ratio": 0.35, "compress_ratio": 3.0},
        HEAD_SINK:   {"recent_frac": 0.10, "fp16_ratio": 0.40, "compress_ratio": 3.0,
                      "always_keep_sink": 4},
    }

    def __init__(self):
        self.head_types: torch.Tensor = None
        self.stats: dict = {}

    def fit(self, calibration_attns: List[torch.Tensor]) -> "HeadClassifier":
        self.head_types, self.stats = classify_heads(calibration_attns)
        return self

    def get_type(self, layer_idx: int, head_idx: int) -> int:
        if self.head_types is None:
            return HEAD_LOCAL
        return int(self.head_types[layer_idx, head_idx].item())

    def get_params(self, layer_idx: int, head_idx: int) -> dict:
        return self.TYPE_PARAMS[self.get_type(layer_idx, head_idx)]
