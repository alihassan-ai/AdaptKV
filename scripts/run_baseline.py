#!/usr/bin/env python3
"""Run a single baseline KV cache method on all configured benchmarks.

Usage:
    python scripts/run_baseline.py --config configs/cpu_debug.yaml --method h2o
    python scripts/run_baseline.py --config configs/runpod_4gpu.yaml --method snapkv
"""

import argparse
import logging
import os
import sys
import time

import torch

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, get_device, validate_config
from src.utils.logging_utils import setup_logger, ExperimentLogger
from src.models.model_loader import load_model
from src.cache import get_cache
from src.evaluation.longbench import LongBenchEvaluator
from src.evaluation.needle import NeedleEvaluator
from src.evaluation.ruler import RulerEvaluator

torch.manual_seed(42)

VALID_METHODS = ["full_cache", "h2o", "snapkv", "streaming"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a KV cache baseline evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--method", required=True, choices=VALID_METHODS,
                        help="Cache method to evaluate.")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["needle", "longbench", "ruler"],
                        help="Which benchmarks to run.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for result JSON files.")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per benchmark task (reduce for speed).")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)
    device = get_device(config)

    logger = setup_logger(f"baseline_{args.method}", args.output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model...")
    model_info = load_model(config)
    model_info["device"] = device

    # Build cache
    cache = get_cache(
        args.method,
        config,
        model_info["num_layers"],
        model_info["num_heads"],
        model_info["head_dim"],
        device=device,
    )

    exp_logger = ExperimentLogger(
        f"baseline_{args.method}",
        config,
        log_dir=args.output_dir,
    )
    exp_logger.start()

    all_results = {"method": args.method, "config": args.config}

    # Needle
    if "needle" in args.benchmarks:
        logger.info("=== Needle-in-a-Haystack ===")
        evaluator = NeedleEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run()
        all_results["needle"] = results
        exp_logger.log_metric("needle_avg", results.get("__average__", 0.0))

    # LongBench
    if "longbench" in args.benchmarks:
        logger.info("=== LongBench ===")
        evaluator = LongBenchEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run(max_samples_per_task=args.max_samples)
        all_results["longbench"] = results
        exp_logger.log_metric("longbench_avg", results.get("__average__", 0.0))

    # RULER
    if "ruler" in args.benchmarks:
        logger.info("=== RULER ===")
        evaluator = RulerEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run()
        all_results["ruler"] = results
        exp_logger.log_metric("ruler_avg", results.get("__average__", 0.0))

    record = exp_logger.finish()
    logger.info(f"Baseline {args.method} evaluation complete.")
    return all_results


if __name__ == "__main__":
    main()
