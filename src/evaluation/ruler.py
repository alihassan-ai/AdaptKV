"""RULER benchmark runner.

RULER (Ruler for Unified Long-context Evaluation and Ranking) evaluates models
on tasks requiring reasoning over long contexts: multi-hop QA, key-value
retrieval, and aggregation tasks.

Reference: Hsieh et al., "RULER: What's the Real Context Size of Your LLM?", 2024.

This implementation provides a self-contained version of the main RULER tasks
that can run offline (no external dataset download required).
"""

import json
import logging
import os
import random
import string
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .metrics import compute_f1, compute_accuracy, MemoryTracker
from .profiler import Profiler

logger = logging.getLogger(__name__)


class RulerEvaluator:
    """Evaluates cache strategies on RULER-style synthetic tasks."""

    # Task definitions: (name, description, generator, scorer)
    TASKS = ["niah_single", "niah_multi_key", "vt", "cwe", "fwe", "qa_1", "qa_2"]

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
        self.max_new_tokens = config.get("evaluation", {}).get("max_new_tokens", 64)
        self.n_samples = 10   # samples per (task, context_length) cell
        random.seed(42)

    def run(self, tasks: Optional[List[str]] = None) -> Dict:
        """Run the RULER evaluation for all tasks and context lengths."""
        if tasks is None:
            tasks = self.TASKS

        all_results: Dict = {}
        profiler = Profiler(self.device)
        profiler.start_experiment()

        for task in tasks:
            all_results[task] = {}
            for ctx_len in self.context_lengths:
                logger.info(f"RULER: task={task}, ctx_len={ctx_len}")
                score, meta = self._eval_cell(task, ctx_len)
                all_results[task][ctx_len] = {"score": score, **meta}
                logger.info(f"  score={score:.3f}")

        exp_stats = profiler.end_experiment()
        all_results["__profiler__"] = exp_stats

        # Average across all cells
        flat_scores = [
            all_results[t][c]["score"]
            for t in tasks for c in self.context_lengths
            if "score" in all_results.get(t, {}).get(c, {})
        ]
        all_results["__average__"] = sum(flat_scores) / len(flat_scores) if flat_scores else 0.0

        out_path = os.path.join(
            self.output_dir,
            f"ruler_{self.cache.__class__.__name__}.json"
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"RULER results saved to {out_path}")
        return all_results

    # ------------------------------------------------------------------ #
    # Task generators                                                      #
    # ------------------------------------------------------------------ #

    def _eval_cell(self, task: str, ctx_len: int) -> Tuple[float, Dict]:
        """Generate samples for one (task, ctx_len) cell and evaluate."""
        generator_map = {
            "niah_single":    self._gen_niah_single,
            "niah_multi_key": self._gen_niah_multi,
            "vt":             self._gen_variable_tracking,
            "cwe":            self._gen_common_words,
            "fwe":            self._gen_freq_words,
            "qa_1":           self._gen_qa,
            "qa_2":           self._gen_qa_multi,
        }
        gen = generator_map.get(task, self._gen_niah_single)

        preds, refs = [], []
        for _ in range(self.n_samples):
            prompt, answer = gen(ctx_len)
            prediction = self._generate(prompt)
            preds.append(prediction)
            refs.append(answer)

        score = sum(
            compute_f1(p, r) for p, r in zip(preds, refs)
        ) / max(len(preds), 1)

        return score, {"num_samples": len(preds), "ctx_len": ctx_len}

    def _gen_niah_single(self, ctx_len: int) -> Tuple[str, str]:
        """Single needle-in-a-haystack with random UUID key."""
        key = "".join(random.choices(string.ascii_uppercase, k=8))
        value = "".join(random.choices(string.digits, k=6))
        needle = f"The magic number for key {key} is {value}."

        filler = self._make_filler(ctx_len)
        insert_pos = random.randint(0, len(filler))
        filler.insert(insert_pos, needle)

        prompt = (
            " ".join(filler) +
            f"\n\nWhat is the magic number for key {key}? Answer:"
        )
        return prompt, value

    def _gen_niah_multi(self, ctx_len: int) -> Tuple[str, str]:
        """Multiple needles; retrieve one specific value."""
        needles = {}
        for _ in range(5):
            k = "".join(random.choices(string.ascii_uppercase, k=6))
            v = "".join(random.choices(string.digits, k=4))
            needles[k] = v

        filler = self._make_filler(ctx_len)
        for k, v in needles.items():
            pos = random.randint(0, len(filler))
            filler.insert(pos, f"Record {k}={v}.")

        target_key = random.choice(list(needles.keys()))
        prompt = (
            " ".join(filler) +
            f"\n\nWhat is the value of record {target_key}? Answer:"
        )
        return prompt, needles[target_key]

    def _gen_variable_tracking(self, ctx_len: int) -> Tuple[str, str]:
        """Track the final value of a variable through assignments."""
        var = "X"
        steps = []
        value = 0
        n_steps = min(20, ctx_len // 50)

        for _ in range(n_steps):
            value = random.randint(1, 999)
            steps.append(f"Let {var} = {value}.")

        filler = self._make_filler(ctx_len - len(steps) * 5)
        for i, step in enumerate(steps):
            pos = i * (len(filler) // max(n_steps, 1))
            filler.insert(min(pos, len(filler)), step)

        prompt = (
            " ".join(filler) +
            f"\n\nWhat is the final value of {var}? Answer:"
        )
        return prompt, str(value)

    def _gen_common_words(self, ctx_len: int) -> Tuple[str, str]:
        """Count the occurrences of a target word in a long text."""
        words = ["apple", "banana", "cherry", "dog", "elephant"]
        target = random.choice(words)
        count = random.randint(3, 10)

        filler_words = [random.choice(words) for _ in range(ctx_len // 3)]
        # Insert exactly `count` occurrences of target
        positions = random.sample(range(len(filler_words)), min(count, len(filler_words)))
        for pos in positions:
            filler_words[pos] = target

        prompt = (
            "Text: " + " ".join(filler_words) +
            f"\n\nHow many times does '{target}' appear? Answer with a number:"
        )
        return prompt, str(filler_words.count(target))

    def _gen_freq_words(self, ctx_len: int) -> Tuple[str, str]:
        """Find the most frequent word in a passage."""
        words = [f"word{i}" for i in range(10)]
        freq = {w: random.randint(1, 5) for w in words}
        most_freq = max(freq, key=freq.get)

        tokens = []
        for w, c in freq.items():
            tokens.extend([w] * c)
        random.shuffle(tokens)

        prompt = (
            "Text: " + " ".join(tokens) +
            "\n\nWhat is the most frequent word? Answer:"
        )
        return prompt, most_freq

    def _gen_qa(self, ctx_len: int) -> Tuple[str, str]:
        """Simple one-hop QA with a planted fact."""
        entity = "".join(random.choices(string.ascii_uppercase, k=5))
        attribute = random.choice(["color", "size", "age", "name"])
        value = "".join(random.choices(string.ascii_lowercase, k=6))

        fact = f"The {attribute} of {entity} is {value}."
        filler = self._make_filler(ctx_len)
        filler.insert(random.randint(0, len(filler)), fact)

        prompt = (
            " ".join(filler) +
            f"\n\nWhat is the {attribute} of {entity}? Answer:"
        )
        return prompt, value

    def _gen_qa_multi(self, ctx_len: int) -> Tuple[str, str]:
        """Two-hop QA requiring chaining two planted facts."""
        A = "".join(random.choices(string.ascii_uppercase, k=4))
        B = "".join(random.choices(string.ascii_uppercase, k=4))
        val = "".join(random.choices(string.ascii_lowercase, k=5))

        fact1 = f"{A} is connected to {B}."
        fact2 = f"The code of {B} is {val}."
        filler = self._make_filler(ctx_len)

        filler.insert(random.randint(0, len(filler)), fact1)
        filler.insert(random.randint(0, len(filler)), fact2)

        prompt = (
            " ".join(filler) +
            f"\n\nWhat is the code of the entity that {A} is connected to? Answer:"
        )
        return prompt, val

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _make_filler(self, approx_tokens: int) -> List[str]:
        """Generate a list of filler sentences totalling approx_tokens tokens."""
        fillers = [
            "The report concluded with a series of recommendations.",
            "Participants gathered to discuss ongoing developments.",
            "The committee approved the proposal after deliberation.",
            "New studies suggest that regular exercise improves cognitive function.",
            "The organization announced an expansion of its services.",
        ]
        sentences_needed = max(1, approx_tokens // 12)
        return [fillers[i % len(fillers)] for i in range(sentences_needed)]

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
