"""Perplexity, accuracy, and memory tracking utilities."""

import logging
import math
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute perplexity of `text` under the model using a sliding window.

    Uses the overlapping-window method to handle sequences longer than the
    model's max context.

    Returns:
        Perplexity (float); lower is better.
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        chunk = input_ids[:, begin_loc:end_loc]
        target = chunk.clone()
        target[:, :-trg_len] = -100   # mask already-scored tokens

        with torch.no_grad():
            out = model(chunk, labels=target)
            nlls.append(out.loss.item() * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return math.exp(sum(nlls) / seq_len)


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Exact-match accuracy between predicted and reference strings."""
    if not references:
        return 0.0
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return correct / len(references)


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference strings."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens  = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Simplified ROUGE-L (LCS-based) without external dependencies."""
    pred_words = prediction.lower().split()
    ref_words  = reference.lower().split()

    if not pred_words or not ref_words:
        return 0.0

    m, n = len(ref_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == pred_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall    = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class MemoryTracker:
    """Tracks peak GPU memory usage across a context block."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._peak_bytes: int = 0
        self._enabled = torch.cuda.is_available() and device != "cpu"

    def reset(self) -> None:
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_bytes = 0

    def peak_bytes(self) -> int:
        if self._enabled:
            return torch.cuda.max_memory_allocated(self.device)
        return self._peak_bytes

    def peak_mb(self) -> float:
        return self.peak_bytes() / (1024 ** 2)

    @contextmanager
    def track(self):
        self.reset()
        try:
            yield self
        finally:
            self._peak_bytes = self.peak_bytes()

    def get_stats(self) -> Dict:
        stats: Dict = {"peak_memory_mb": self.peak_mb()}
        if self._enabled:
            stats["allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            stats["reserved_mb"]  = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        return stats
