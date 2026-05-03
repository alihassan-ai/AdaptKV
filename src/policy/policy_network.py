"""AdaptKV policy MLP: per-head adaptive three-tier KV cache decision network."""

import torch
import torch.nn as nn
from typing import Tuple


class AdaptKVPolicy(nn.Module):
    """Per-head adaptive KV cache policy network.

    Input:  5-dimensional feature vector per cached KV entry
    Output: 3-class logits → keep at FP16 (0), compress to 4-bit (1), evict (2)

    Feature dimensions (4 continuous + 1 discrete):
        0: Rolling attention momentum  (EWMA over last `attention_window` steps)
        1: Attention variance          (variance over the same window)
        2: Key embedding redundancy    (max cosine similarity to other keys)
        3: Relative token position     (position_index / total_cached_length)
        --- head_type fed as int index, embedded separately ---
    """

    HEAD_KEEP = 0
    HEAD_COMPRESS = 1
    HEAD_EVICT = 2

    def __init__(
        self,
        num_features: int = 5,
        hidden_dim: int = 128,
        num_tiers: int = 3,
        num_head_types: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_continuous = num_features - 1  # 4 continuous features
        embed_dim = 8

        # Learned embedding for head type (local=0, global=1, sink=2)
        self.head_type_embedding = nn.Embedding(num_head_types, embed_dim)

        in_dim = self.num_continuous + embed_dim
        self.policy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tiers),
        )

        # Weight init: small values → initially uncertain policy
        for m in self.policy_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        head_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for all cached entries in a batch.

        Args:
            features:      [batch, num_entries, 4]  continuous features
            head_type_ids: [batch, num_entries]      integer head-type indices

        Returns:
            logits: [batch, num_entries, 3]
        """
        head_embeds = self.head_type_embedding(head_type_ids)  # [B, N, 8]
        combined = torch.cat([features, head_embeds], dim=-1)   # [B, N, 12]
        return self.policy_net(combined)                          # [B, N, 3]

    def decide(
        self,
        features: torch.Tensor,
        head_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return hard tier decisions (argmax) without gradient tracking.

        Returns:
            decisions: [batch, num_entries]  — 0=keep, 1=compress, 2=evict
        """
        with torch.no_grad():
            logits = self.forward(features, head_type_ids)
            return logits.argmax(dim=-1)

    def decide_with_budget_constraint(
        self,
        features: torch.Tensor,
        head_type_ids: torch.Tensor,
        budget: int,
        compress_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Make decisions while respecting an absolute token budget.

        Tokens are first sorted by keep-probability (descending).
        The top `budget * (1 - compress_ratio)` get KEEP,
        the next `budget * compress_ratio` get COMPRESS,
        the rest get EVICT.

        Args:
            budget:          Maximum total tokens to retain.
            compress_ratio:  Fraction of budget allocated to compressed tier.

        Returns:
            decisions: [batch, num_entries]
        """
        with torch.no_grad():
            logits = self.forward(features, head_type_ids)
            probs = logits.softmax(dim=-1)                    # [B, N, 3]
            importance = probs[..., 0] + 0.5 * probs[..., 1] # keep + half-weight compress

            B, N = importance.shape
            decisions = torch.full((B, N), self.HEAD_EVICT,
                                   dtype=torch.long, device=features.device)

            n_keep = max(1, int(budget * (1.0 - compress_ratio)))
            n_compress = max(0, budget - n_keep)

            for b in range(B):
                sorted_idx = importance[b].argsort(descending=True)
                keep_idx = sorted_idx[:n_keep]
                comp_idx = sorted_idx[n_keep: n_keep + n_compress]
                decisions[b, keep_idx] = self.HEAD_KEEP
                decisions[b, comp_idx] = self.HEAD_COMPRESS

        return decisions
