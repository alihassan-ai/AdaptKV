#!/usr/bin/env python3
"""Train the AdaptKV tier-assignment policy network via knowledge distillation.

Usage:
    python scripts/train_policy.py --config configs/cpu_debug.yaml
    python scripts/train_policy.py --config configs/runpod_4gpu.yaml
"""

import argparse
import logging
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, get_device, validate_config
from src.utils.logging_utils import setup_logger, ExperimentLogger
from src.models.model_loader import load_model
from src.policy.trainer import PolicyTrainer

torch.manual_seed(42)
random.seed(42)

# Default calibration prompts (used when datasets unavailable)
_DEFAULT_PROMPTS = [
    "The history of artificial intelligence dates back to the mid-20th century.",
    "Climate change is one of the most pressing challenges facing humanity today.",
    "The development of transformer-based language models has revolutionized NLP.",
    "Scientists are exploring new frontiers in quantum computing and quantum cryptography.",
    "The global economy is becoming increasingly interconnected through trade and finance.",
    "Advances in medical technology have improved patient outcomes and life expectancy.",
    "Renewable energy sources such as solar and wind power are growing rapidly worldwide.",
    "Space exploration continues to push the boundaries of human knowledge and technology.",
    "The art world is constantly evolving as new styles and movements emerge.",
    "Education systems around the world are adapting to meet the needs of the 21st century.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaptKV policy network.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--policy-save-path", default=None,
                        help="Override path for saving trained policy checkpoint.")
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset name/path for training prompts.")
    return parser.parse_args()


def load_training_prompts(config: dict, dataset_name: str = None) -> list:
    """Load training prompts from HuggingFace dataset or use defaults."""
    n_samples = config.get("policy", {}).get("training_samples", 100)

    if dataset_name:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            prompts = []
            for sample in ds:
                text = sample.get("text", sample.get("content", ""))
                if len(text) > 50:
                    prompts.append(text[:2000])
                if len(prompts) >= n_samples:
                    break
            if prompts:
                return prompts
        except Exception as e:
            logging.warning(f"Could not load dataset '{dataset_name}': {e}")

    # Fallback: extend default prompts to requested count
    prompts = []
    while len(prompts) < n_samples:
        prompts.extend(_DEFAULT_PROMPTS)
    return prompts[:n_samples]


def main():
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)
    device = get_device(config)

    logger = setup_logger("train_policy", args.output_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {config.get('policy', {}).get('training_samples', 100)}")

    # Load model
    logger.info("Loading model...")
    model_info = load_model(config)

    # Prepare training prompts
    logger.info("Loading training prompts...")
    prompts = load_training_prompts(config, args.dataset)
    logger.info(f"Loaded {len(prompts)} training prompts.")

    # Build trainer
    trainer = PolicyTrainer(config, device=device)

    exp_logger = ExperimentLogger("train_policy", config, log_dir=args.output_dir)
    exp_logger.start()

    # Collect training data (run teacher model)
    logger.info("Collecting training data from teacher model (full-cache forward passes)...")
    trainer.collect_training_data(
        model=model_info["model"],
        tokenizer=model_info["tokenizer"],
        prompts=prompts,
        num_samples=config.get("policy", {}).get("training_samples", 100),
    )

    # Train policy
    logger.info("Training policy MLP...")
    history = trainer.train(
        epochs=config.get("policy", {}).get("training_epochs", 3),
        save_path=args.policy_save_path or os.path.join(args.output_dir, "policy.pt"),
    )

    exp_logger.log_metrics({
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "final_accuracy": history["accuracy"][-1] if history["accuracy"] else None,
        "num_epochs": len(history["loss"]),
    })
    exp_logger.finish()

    logger.info("Policy training complete!")
    logger.info(f"  Final loss:     {history['loss'][-1]:.4f}" if history["loss"] else "")
    logger.info(f"  Final accuracy: {history['accuracy'][-1]:.3f}" if history["accuracy"] else "")


if __name__ == "__main__":
    main()
