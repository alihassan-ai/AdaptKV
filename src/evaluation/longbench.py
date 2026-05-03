"""LongBench benchmark runner.

Runs the 16 LongBench subtasks from THUDM/LongBench and computes official
task-specific metrics for each evaluated cache strategy.

Reference: Bai et al., "LongBench: A Bilingual, Multitask Benchmark for Long
Context Understanding", ACL 2024.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .metrics import compute_f1, compute_rouge_l, compute_accuracy, MemoryTracker
from .profiler import Profiler

logger = logging.getLogger(__name__)

# Official LongBench metric assignments per task
_TASK_METRICS = {
    "narrativeqa":          "f1",
    "qasper":               "f1",
    "multifieldqa_en":      "f1",
    "hotpotqa":             "f1",
    "2wikimqa":             "f1",
    "musique":              "f1",
    "gov_report":           "rouge_l",
    "qmsum":                "rouge_l",
    "multi_news":           "rouge_l",
    "trec":                 "accuracy",
    "triviaqa":             "f1",
    "samsum":               "rouge_l",
    "passage_count":        "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc":                  "edit_sim",
    "repobench-p":          "edit_sim",
}

_ALL_TASKS = list(_TASK_METRICS.keys())


class LongBenchEvaluator:
    """Evaluates a KV cache strategy on the LongBench benchmark."""

    def __init__(self, config: Dict, model_info: Dict, cache_strategy,
                 output_dir: str = "results"):
        self.config = config
        self.model     = model_info["model"]
        self.tokenizer = model_info["tokenizer"]
        self.device    = model_info["device"]
        self.cache     = cache_strategy
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.max_new_tokens = config.get("evaluation", {}).get("max_new_tokens", 256)
        self.batch_size     = config.get("evaluation", {}).get("batch_size", 1)

    def run(
        self,
        tasks: Optional[List[str]] = None,
        max_samples_per_task: int = 200,
    ) -> Dict[str, Any]:
        """Run evaluation on specified tasks.

        Args:
            tasks:                List of task names; defaults to all 16.
            max_samples_per_task: Cap to keep runtime manageable.

        Returns:
            Dict of {task_name: {"score": float, "num_samples": int, ...}}
        """
        if tasks is None:
            tasks = _ALL_TASKS

        all_results: Dict[str, Any] = {}

        for task in tasks:
            logger.info(f"LongBench: evaluating task '{task}'")
            try:
                result = self._eval_task(task, max_samples_per_task)
                all_results[task] = result
                logger.info(f"  {task}: {result['metric']}={result['score']:.4f}")
            except Exception as e:
                logger.error(f"  {task} failed: {e}")
                all_results[task] = {"score": 0.0, "error": str(e)}

        # Save results
        out_path = os.path.join(
            self.output_dir,
            f"longbench_{self.cache.__class__.__name__}.json"
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"LongBench results saved to {out_path}")

        # Overall average
        scores = [v["score"] for v in all_results.values() if "score" in v]
        all_results["__average__"] = sum(scores) / len(scores) if scores else 0.0
        return all_results

    def _eval_task(self, task: str, max_samples: int) -> Dict[str, Any]:
        """Run one LongBench task and return score + metadata."""
        samples = self._load_samples(task, max_samples)
        if not samples:
            return {"score": 0.0, "num_samples": 0, "metric": _TASK_METRICS.get(task, "f1")}

        metric_name = _TASK_METRICS.get(task, "f1")
        predictions, references = [], []
        profiler = Profiler(self.device)
        mem_tracker = MemoryTracker(self.device)

        profiler.start_experiment()
        with mem_tracker.track():
            for sample in tqdm(samples, desc=task, leave=False):
                prompt = self._build_prompt(task, sample)
                prediction = self._generate(prompt)
                predictions.append(prediction)
                references.append(sample.get("answers", sample.get("answer", "")))

        exp_stats = profiler.end_experiment()

        score = self._compute_score(metric_name, predictions, references)
        return {
            "score": score,
            "num_samples": len(samples),
            "metric": metric_name,
            "profiler": exp_stats,
            "memory": mem_tracker.get_stats(),
            "cache_stats": self.cache.get_stats() if hasattr(self.cache, "get_stats") else {},
        }

    def _load_samples(self, task: str, max_samples: int) -> List[Dict]:
        """Load LongBench samples from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task, split="test")
            samples = list(ds)[:max_samples]
            return samples
        except Exception as e:
            logger.warning(f"Could not load LongBench/{task} from HuggingFace: {e}")
            logger.warning("Generating synthetic samples for CPU debug mode.")
            return self._synthetic_samples(task, min(max_samples, 5))

    def _synthetic_samples(self, task: str, n: int) -> List[Dict]:
        """Minimal synthetic samples for CPU/offline testing."""
        samples = []
        for i in range(n):
            samples.append({
                "input": f"Context: Alice went to the store. Question: Where did Alice go?",
                "context": f"Alice went to the store on day {i}.",
                "answers": "the store",
                "answer": "the store",
                "length": 100,
            })
        return samples

    def _build_prompt(self, task: str, sample: Dict) -> str:
        """Build a prompt string from a LongBench sample."""
        context = sample.get("context", sample.get("input", ""))
        question = sample.get("input", sample.get("question", ""))
        return f"{context}\n\nQuestion: {question}\nAnswer:"

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Run greedy generation and return the generated text."""
        self.model.eval()
        self.cache.reset()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = inputs["input_ids"].to(self.device)

        try:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = output_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            logger.debug(f"Generation failed: {e}")
            return ""

    def _compute_score(self, metric: str, preds: List[str],
                       refs: List[Any]) -> float:
        """Compute aggregate score for a task metric."""
        if not preds:
            return 0.0

        scores = []
        for pred, ref in zip(preds, refs):
            # refs can be list of acceptable answers or single string
            if isinstance(ref, list):
                # Take max score across all valid answers
                s = max(self._single_score(metric, pred, r) for r in ref) if ref else 0.0
            else:
                s = self._single_score(metric, pred, str(ref))
            scores.append(s)

        return sum(scores) / len(scores)

    def _single_score(self, metric: str, pred: str, ref: str) -> float:
        if metric == "f1":
            return compute_f1(pred, ref)
        if metric == "rouge_l":
            return compute_rouge_l(pred, ref)
        if metric == "accuracy":
            return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
        if metric == "edit_sim":
            return _edit_similarity(pred, ref)
        return 0.0


def _edit_similarity(a: str, b: str) -> float:
    """Normalized edit similarity (1 - edit_distance / max_len)."""
    if not a and not b:
        return 1.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return 1.0 - dp[m] / max(n, m)
