#!/usr/bin/env python3
"""Run AdaptKV (our method) inference evaluation on all configured benchmarks.

Usage:
    python scripts/run_adaptkv.py --config configs/cpu_debug.yaml
    python scripts/run_adaptkv.py --config configs/runpod_4gpu.yaml \
        --policy-path results/policy.pt
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, get_device, validate_config
from src.utils.logging_utils import setup_logger, ExperimentLogger
from src.models.model_loader import load_model
from src.cache.adaptkv_cache import AdaptKVCache
from src.policy.policy_network import AdaptKVPolicy
from src.policy.feature_extractor import FeatureExtractor
from src.policy.trainer import PolicyTrainer
from src.evaluation.longbench import LongBenchEvaluator
from src.evaluation.needle import NeedleEvaluator
from src.evaluation.ruler import RulerEvaluator

torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AdaptKV inference evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--policy-path", default=None,
                        help="Path to trained policy checkpoint (.pt file). "
                             "If not provided, uses attention-score fallback.")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["needle", "longbench", "ruler"],
                        help="Which benchmarks to run.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for result JSON files.")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per benchmark task.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)
    device = get_device(config)

    logger = setup_logger("adaptkv", args.output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model...")
    model_info = load_model(config)
    model_info["device"] = device

    # Build AdaptKV cache
    cache = AdaptKVCache(
        config,
        model_info["num_layers"],
        model_info["num_heads"],
        model_info["head_dim"],
        device=device,
    )

    # Load or initialize policy
    policy_cfg = config.get("policy", {})
    policy = AdaptKVPolicy(
        num_features=policy_cfg.get("num_features", 5),
        hidden_dim=policy_cfg.get("hidden_dim", 128),
        num_tiers=policy_cfg.get("num_tiers", 3),
    ).to(device)

    feature_extractor = FeatureExtractor(
        config,
        model_info["num_layers"],
        model_info["num_heads"],
        device=device,
    )

    if args.policy_path and os.path.exists(args.policy_path):
        logger.info(f"Loading policy from {args.policy_path}")
        trainer = PolicyTrainer(config, device=device)
        trainer.policy = policy
        trainer.feature_extractor = feature_extractor
        trainer.load_policy(args.policy_path)
        policy = trainer.policy
        feature_extractor = trainer.feature_extractor
    else:
        logger.warning("No policy checkpoint provided. Using attention-score fallback.")
        policy.eval()

    cache.set_policy(policy, feature_extractor)

    exp_logger = ExperimentLogger("adaptkv", config, log_dir=args.output_dir)
    exp_logger.start()

    all_results = {"method": "adaptkv", "config": args.config}

    # Needle
    if "needle" in args.benchmarks:
        logger.info("=== Needle-in-a-Haystack (AdaptKV) ===")
        evaluator = NeedleEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run()
        all_results["needle"] = results
        exp_logger.log_metric("needle_avg", results.get("__average__", 0.0))

    # LongBench
    if "longbench" in args.benchmarks:
        logger.info("=== LongBench (AdaptKV) ===")
        evaluator = LongBenchEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run(max_samples_per_task=args.max_samples)
        all_results["longbench"] = results
        exp_logger.log_metric("longbench_avg", results.get("__average__", 0.0))

    # RULER
    if "ruler" in args.benchmarks:
        logger.info("=== RULER (AdaptKV) ===")
        evaluator = RulerEvaluator(config, model_info, cache, args.output_dir)
        results = evaluator.run()
        all_results["ruler"] = results
        exp_logger.log_metric("ruler_avg", results.get("__average__", 0.0))

    exp_logger.log_metric("cache_stats", cache.get_stats())
    record = exp_logger.finish()
    logger.info("AdaptKV evaluation complete.")
    return all_results


if __name__ == "__main__":
    main()
