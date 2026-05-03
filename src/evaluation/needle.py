"""Needle-in-a-Haystack evaluation.

Tests the model's ability to retrieve a specific fact ("needle") embedded at
various depths within a long distractor document ("haystack").

Evaluation grid:
    - Context lengths: from config['evaluation']['context_lengths']
    - Depth percentages: [10, 25, 50, 75, 90]
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .metrics import MemoryTracker
from .profiler import Profiler

logger = logging.getLogger(__name__)

# The secret needle fact
_NEEDLE = "The secret passcode is ADAPTIV-2024."
_QUESTION = "What is the secret passcode?"
_EXPECTED = "ADAPTIV-2024"

# Filler sentence pool for building haystacks
_FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Scientists have discovered a new species of deep-sea fish near the Mariana Trench.",
    "The annual economic summit concluded with agreements on trade and climate policy.",
    "Researchers published findings on the efficacy of a novel drug compound.",
    "A new bridge spanning two kilometers was inaugurated in the capital city.",
    "The championship match ended with a dramatic overtime goal in the final seconds.",
    "Students from across the country competed in the national science olympiad.",
    "The museum unveiled a collection of artifacts from an ancient civilization.",
    "Weather forecasters predict significant rainfall over the next several days.",
    "The new software update includes improvements to performance and security.",
]


class NeedleEvaluator:
    """Evaluates retrieval accuracy at various context lengths and needle depths."""

    DEPTH_PERCENTAGES = [10, 25, 50, 75, 90]

    def __init__(self, config: Dict, model_info: Dict, cache_strategy,
                 output_dir: str = "results"):
        self.config = config
        self.model     = model_info["model"]
        self.tokenizer = model_info["tokenizer"]
        self.device    = model_info["device"]
        self.cache     = cache_strategy
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.context_lengths = config.get("evaluation", {}).get(
            "context_lengths", [512, 1024, 2048]
        )
        self.max_new_tokens = min(
            64, config.get("evaluation", {}).get("max_new_tokens", 64)
        )
        random.seed(42)

    def run(self) -> Dict:
        """Run the full needle evaluation grid.

        Returns:
            results: {context_length: {depth_pct: {"score": float, ...}}}
        """
        results: Dict = {}
        profiler = Profiler(self.device)
        profiler.start_experiment()

        for ctx_len in self.context_lengths:
            results[ctx_len] = {}
            for depth_pct in self.DEPTH_PERCENTAGES:
                logger.info(f"Needle: ctx_len={ctx_len}, depth={depth_pct}%")
                score, metadata = self._eval_single(ctx_len, depth_pct)
                results[ctx_len][depth_pct] = {
                    "score": score,
                    **metadata,
                }
                logger.info(f"  score={score:.3f}")

        exp_stats = profiler.end_experiment()
        results["__profiler__"] = exp_stats

        # Aggregate
        all_scores = [
            results[c][d]["score"]
            for c in self.context_lengths
            for d in self.DEPTH_PERCENTAGES
        ]
        results["__average__"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

        out_path = os.path.join(
            self.output_dir,
            f"needle_{self.cache.__class__.__name__}.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Needle results saved to {out_path}")
        return results

    def _eval_single(self, ctx_len: int, depth_pct: int):
        """Build one haystack and test needle retrieval."""
        haystack = self._build_haystack(ctx_len, depth_pct)
        prompt = (
            f"{haystack}\n\n"
            f"Based on the text above, {_QUESTION} "
            f"Answer with the passcode only:"
        )
        prediction = self._generate(prompt)
        score = 1.0 if _EXPECTED.lower() in prediction.lower() else 0.0
        return score, {
            "prediction": prediction[:100],
            "expected": _EXPECTED,
            "context_length": ctx_len,
            "depth_pct": depth_pct,
        }

    def _build_haystack(self, target_token_len: int, depth_pct: int) -> str:
        """Build a text of approximately `target_token_len` tokens with the needle
        inserted at `depth_pct`% of the way through."""
        # Estimate tokens per sentence (~15 tokens average)
        n_sentences = max(10, target_token_len // 15)
        sentences = [
            random.choice(_FILLER_SENTENCES) for _ in range(n_sentences)
        ]

        # Insert needle at the target depth
        needle_pos = max(0, int(n_sentences * depth_pct / 100))
        sentences.insert(needle_pos, _NEEDLE)

        return " ".join(sentences)

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        self.model.eval()
        self.cache.reset()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        )
        input_ids = inputs["input_ids"].to(self.device)

        try:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = output_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return ""
