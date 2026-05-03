"""Train the AdaptKV policy via knowledge distillation from a full-cache model.

Training procedure:
1. Load model with full cache (teacher).
2. Run generation on training prompts; at each step record:
   - Per-head attention weights for all cached tokens
   - Key embeddings for redundancy computation
   - Teacher output logit distribution
3. For each cached token compute an oracle tier label:
   - GPU mode:   run 3 forward passes (keep FP16, compress 4-bit, evict)
                 pick action with lowest KL divergence from full-cache output.
   - CPU mode:   approximate with attention-score thresholds (fast, no extra passes).
4. Train policy MLP with cross-entropy loss against oracle labels.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .policy_network import AdaptKVPolicy
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class PolicyTrainer:
    """Distillation trainer for the AdaptKV tier-assignment policy."""

    # Attention-score thresholds for the fast CPU-mode oracle
    _KEEP_THRESHOLD     = 0.15   # EWMA score above this → KEEP
    _COMPRESS_THRESHOLD = 0.05   # above this but below KEEP → COMPRESS
    # below COMPRESS_THRESHOLD → EVICT

    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.policy_cfg = config.get("policy", {})

        self.policy = AdaptKVPolicy(
            num_features=self.policy_cfg.get("num_features", 5),
            hidden_dim=self.policy_cfg.get("hidden_dim", 128),
            num_tiers=self.policy_cfg.get("num_tiers", 3),
        ).to(device)

        self.feature_extractor: Optional[FeatureExtractor] = None
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.policy_cfg.get("learning_rate", 1e-3),
        )
        self.criterion = nn.CrossEntropyLoss()

        # Training data buffers
        self._features_buf: List[torch.Tensor] = []
        self._head_ids_buf: List[torch.Tensor] = []
        self._labels_buf:   List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def collect_training_data(
        self,
        model,
        tokenizer,
        prompts: List[str],
        num_samples: int,
    ) -> None:
        """Run the teacher (full-cache) model and collect oracle-labelled data.

        Args:
            model:       HuggingFace CausalLM in eval mode.
            tokenizer:   Corresponding tokenizer.
            prompts:     List of training text prompts.
            num_samples: Maximum number of (features, label) pairs to collect.
        """
        model.eval()
        collected = 0
        max_new_tokens = self.config.get("evaluation", {}).get("max_new_tokens", 32)
        use_cpu_oracle = not torch.cuda.is_available()

        for prompt in tqdm(prompts, desc="Collecting training data"):
            if collected >= num_samples:
                break

            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                try:
                    out = model(input_ids, output_attentions=True, use_cache=True)
                except Exception as e:
                    logger.warning(f"Forward pass failed: {e}")
                    continue

            # Extract attention from all layers
            if out.attentions is None:
                logger.warning("Model did not return attentions. Skipping sample.")
                continue

            all_attentions = out.attentions  # tuple of [1, H, S, S]
            past_kv = out.past_key_values    # tuple of (k, v) per layer

            if self.feature_extractor is None:
                num_layers = len(all_attentions)
                num_heads = all_attentions[0].shape[1]
                self.feature_extractor = FeatureExtractor(
                    self.config, num_layers, num_heads, self.device
                )
                # Classify head types using this sample's attention
                calib_attns = []
                for layer_attn in all_attentions:
                    calib_attns.append(layer_attn.squeeze(0))  # [H, S, S]
                self.feature_extractor.classify_head_types(calib_attns)

            # Update feature extractor with attention from all layers
            for layer_idx, layer_attn in enumerate(all_attentions):
                # last token's attention row: [1, H, 1, S]
                last_attn = layer_attn[:, :, -1:, :]
                self.feature_extractor.update_attention_stats(layer_idx, last_attn)

            # Extract features and compute oracle labels
            for layer_idx, (k, v) in enumerate(past_kv):
                # k: [1, H, S, D]
                key_cache = k.squeeze(0)  # [H, S, D]

                if key_cache.shape[1] == 0:
                    continue

                features, head_ids = self.feature_extractor.extract_features(
                    layer_idx, key_cache
                )

                if use_cpu_oracle:
                    labels = self._cpu_oracle_labels(layer_idx, features)
                else:
                    labels = self._gpu_oracle_labels(
                        model, layer_idx, key_cache, k, v, input_ids, out
                    )

                self._features_buf.append(features.squeeze(0))   # [H*S, 4]
                self._head_ids_buf.append(head_ids.squeeze(0))    # [H*S]
                self._labels_buf.append(labels.squeeze(0))        # [H*S]
                collected += features.shape[1]

            if collected % 1000 == 0:
                logger.info(f"Collected {collected}/{num_samples} training samples")

        logger.info(f"Total training samples collected: {collected}")

    def train(self, epochs: Optional[int] = None, save_path: Optional[str] = None) -> Dict:
        """Train the policy on collected data. Returns loss history."""
        if not self._features_buf:
            raise RuntimeError("No training data collected. Call collect_training_data first.")

        epochs = epochs or self.policy_cfg.get("training_epochs", 10)

        features_all = torch.cat(self._features_buf, dim=0)   # [N, 4]
        head_ids_all = torch.cat(self._head_ids_buf, dim=0)    # [N]
        labels_all   = torch.cat(self._labels_buf,   dim=0)    # [N]

        # Shuffle
        perm = torch.randperm(features_all.shape[0])
        features_all = features_all[perm]
        head_ids_all = head_ids_all[perm]
        labels_all   = labels_all[perm]

        dataset = TensorDataset(features_all, head_ids_all, labels_all)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        self.policy.train()
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for feat_batch, hid_batch, lbl_batch in loader:
                feat_batch = feat_batch.to(self.device)
                hid_batch  = hid_batch.to(self.device)
                lbl_batch  = lbl_batch.to(self.device)

                self.optimizer.zero_grad()
                # Policy expects [B, N, 4], [B, N] — add batch dim
                logits = self.policy(
                    feat_batch.unsqueeze(0),
                    hid_batch.unsqueeze(0),
                ).squeeze(0)  # [N, 3]

                loss = self.criterion(logits, lbl_batch.long())
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item() * lbl_batch.shape[0]
                correct += (logits.argmax(-1) == lbl_batch).sum().item()
                total += lbl_batch.shape[0]

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)
            logger.info(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")

        self.policy.eval()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save({
                "policy_state_dict": self.policy.state_dict(),
                "config": self.config,
                "history": history,
                "head_types": self.feature_extractor._head_types
                              if self.feature_extractor else None,
            }, save_path)
            logger.info(f"Policy saved to {save_path}")

        return history

    def load_policy(self, path: str) -> None:
        """Load a previously saved policy checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

        if self.feature_extractor is not None and ckpt.get("head_types") is not None:
            self.feature_extractor._head_types = ckpt["head_types"].to(self.device)
        logger.info(f"Policy loaded from {path}")

    # ------------------------------------------------------------------ #
    # Oracle labeling                                                      #
    # ------------------------------------------------------------------ #

    def _cpu_oracle_labels(
        self,
        layer_idx: int,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Fast CPU oracle: use EWMA momentum thresholds to assign tier labels.

        features: [1, H*CL, 4]  → feature[..., 0] is the momentum channel.
        Returns: [1, H*CL] long tensor with tier labels.
        """
        momentum = features[..., 0]   # [1, H*CL]
        labels = torch.full_like(momentum, 2, dtype=torch.long)  # default: EVICT
        labels[momentum > self._KEEP_THRESHOLD] = 0              # KEEP
        compress_mask = (momentum > self._COMPRESS_THRESHOLD) & (momentum <= self._KEEP_THRESHOLD)
        labels[compress_mask] = 1                                 # COMPRESS
        return labels

    @torch.no_grad()
    def _gpu_oracle_labels(
        self,
        model,
        layer_idx: int,
        key_cache: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        input_ids: torch.Tensor,
        full_output,
    ) -> torch.Tensor:
        """GPU oracle: test each tier via forward passes, pick minimum KL div.

        This is expensive (3 × num_cached_tokens forward passes per layer).
        Only runs on GPU where each pass is fast.

        Returns: [1, H*CL] long tensor with tier labels.
        """
        from .quantization import quantize_to_nf4, dequantize_from_nf4

        full_logits = full_output.logits[:, -1, :]       # [1, vocab]
        full_log_probs = full_logits.log_softmax(-1)

        H, CL, D = key_cache.shape
        labels = torch.full((H * CL,), 2, dtype=torch.long, device=self.device)

        # Sample a subset to make this tractable
        sample_rate = 0.1
        for flat_idx in range(H * CL):
            if torch.rand(1).item() > sample_rate:
                continue

            h = flat_idx // CL
            t = flat_idx % CL

            kl_scores = []
            for action in range(3):  # 0=keep, 1=compress, 2=evict
                modified_k = k_full.clone()   # [1, H, CL, D]
                modified_v = v_full.clone()

                if action == 1:  # compress
                    k_q, k_sc, k_cb = quantize_to_nf4(modified_k[0, h, t, :])
                    k_dq = dequantize_from_nf4(k_q, k_sc, k_cb,
                                               (D,), modified_k.dtype)
                    modified_k[0, h, t, :] = k_dq

                    v_q, v_sc, v_cb = quantize_to_nf4(modified_v[0, h, t, :])
                    v_dq = dequantize_from_nf4(v_q, v_sc, v_cb,
                                               (D,), modified_v.dtype)
                    modified_v[0, h, t, :] = v_dq

                elif action == 2:  # evict — mask the position
                    # Set key/value to zero (simulates removal)
                    modified_k[0, h, t, :] = 0.0
                    modified_v[0, h, t, :] = 0.0

                try:
                    past_kv = tuple(
                        (modified_k, modified_v) if i == layer_idx else pv
                        for i, pv in enumerate(full_output.past_key_values)
                    )
                    out = model(input_ids[:, -1:], past_key_values=past_kv,
                                use_cache=False)
                    logits = out.logits[:, -1, :]
                    log_probs = logits.log_softmax(-1)
                    kl = F.kl_div(log_probs, full_log_probs.exp(), reduction="sum")
                    kl_scores.append(kl.item())
                except Exception:
                    kl_scores.append(float("inf"))

            labels[flat_idx] = int(min(range(3), key=lambda i: kl_scores[i]))

        return labels.unsqueeze(0)
