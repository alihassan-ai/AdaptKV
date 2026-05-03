#!/usr/bin/env python3
"""Multi-GPU distributed AdaptKV evaluation.

Launch with torchrun:
    torchrun --nproc_per_node=4 scripts/run_distributed.py \
        --config configs/runpod_4gpu.yaml

Or in simulation mode (CPU/single process):
    python scripts/run_distributed.py --config configs/cpu_debug.yaml
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, get_device, validate_config
from src.utils.logging_utils import setup_logger, save_results
from src.models.model_loader import load_model
from src.cache.adaptkv_cache import AdaptKVCache
from src.policy.policy_network import AdaptKVPolicy
from src.policy.feature_extractor import FeatureExtractor
from src.distributed.cache_manager import DistributedCacheManager
from src.distributed.comm_aware_placement import CommAwarePlacement
from src.distributed.prefetcher import AsyncPrefetcher
from src.distributed.utils import (
    init_distributed, get_rank, get_world_size, barrier,
)
from src.evaluation.needle import NeedleEvaluator
from src.evaluation.profiler import Profiler

torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed AdaptKV evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--policy-path", default=None,
                        help="Path to trained policy checkpoint.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--benchmark", default="needle",
                        choices=["needle", "longbench", "ruler"])
    return parser.parse_args()


class DistributedAdaptKVRunner:
    """Orchestrates AdaptKV inference with distributed cache management."""

    def __init__(self, config: dict, model_info: dict, rank: int,
                 world_size: int, output_dir: str):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = model_info["device"]
        self.output_dir = output_dir
        self.simulate = config.get("distributed", {}).get("simulate", True)

        # Build distributed cache components
        self.cache_manager = DistributedCacheManager(
            config,
            model_info["num_layers"],
            model_info["num_heads"],
            model_info["head_dim"],
            device=self.device,
        )

        self.placement = CommAwarePlacement(
            config,
            local_rank=rank,
            num_gpus=world_size if not self.simulate
            else config.get("distributed", {}).get("num_simulated_gpus", 4),
        )

        self.prefetcher = AsyncPrefetcher(
            config,
            self.cache_manager,
            model_info["num_layers"],
            model_info["num_heads"],
            model_info["head_dim"],
            device=self.device,
        )

        # Local AdaptKV cache (each rank manages its shard)
        self.local_cache = AdaptKVCache(
            config,
            model_info["num_layers"],
            model_info["num_heads"],
            model_info["head_dim"],
            device=self.device,
        )

        # Policy network
        policy_cfg = config.get("policy", {})
        self.policy = AdaptKVPolicy(
            num_features=policy_cfg.get("num_features", 5),
            hidden_dim=policy_cfg.get("hidden_dim", 128),
        ).to(self.device)
        self.policy.eval()

        self.feature_extractor = FeatureExtractor(
            config, model_info["num_layers"], model_info["num_heads"],
            device=self.device,
        )
        self.local_cache.set_policy(self.policy, self.feature_extractor)

    def load_policy(self, path: str) -> None:
        if not os.path.exists(path):
            logging.warning(f"Policy checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        if ckpt.get("head_types") is not None:
            self.feature_extractor._head_types = ckpt["head_types"].to(self.device)
        logging.info(f"Policy loaded from {path}")

    def run_benchmark(self, model_info: dict, benchmark: str) -> dict:
        profiler = Profiler(self.device)
        profiler.start_experiment()

        if benchmark == "needle":
            evaluator = NeedleEvaluator(
                self.config, model_info, self.local_cache, self.output_dir
            )
            results = evaluator.run()
        else:
            results = {"error": f"Benchmark '{benchmark}' not yet implemented for distributed."}

        exp_stats = profiler.end_experiment()
        results["__distributed_stats__"] = self.cache_manager.get_stats()
        results["__profiler__"] = exp_stats
        return results


def main():
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    simulate = config.get("distributed", {}).get("simulate", True)
    if not simulate and torch.cuda.is_available():
        init_distributed()

    rank = get_rank()
    world_size = get_world_size()

    logger = setup_logger(f"distributed_rank{rank}", args.output_dir)
    logger.info(f"Distributed eval: rank={rank}/{world_size}, simulate={simulate}")

    device = get_device(config)
    model_info = load_model(config)
    model_info["device"] = device

    runner = DistributedAdaptKVRunner(
        config, model_info, rank, world_size, args.output_dir
    )

    if args.policy_path:
        runner.load_policy(args.policy_path)

    logger.info(f"Running benchmark: {args.benchmark}")
    results = runner.run_benchmark(model_info, args.benchmark)

    # Only rank 0 saves aggregate results
    if rank == 0:
        out_path = os.path.join(
            args.output_dir,
            f"distributed_{args.benchmark}_{world_size}gpu.json"
        )
        save_results(results, out_path)
        logger.info(f"Results saved to {out_path}")

        avg = results.get("__average__", "N/A")
        comm_bytes = results.get("__distributed_stats__", {}).get("total_comm_bytes", 0)
        logger.info(f"Average score:      {avg}")
        logger.info(f"Total comm bytes:   {comm_bytes:,}")

    if not simulate:
        barrier()

    return results


if __name__ == "__main__":
    main()
