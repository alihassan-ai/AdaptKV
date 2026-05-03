import yaml
import os
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as nested dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base config, override wins on conflicts."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def get_device(config: Dict = None) -> str:
    """Return 'cuda' if available, else 'cpu'. Respects force_cpu in config."""
    import torch
    if config and config.get("model", {}).get("force_cpu", False):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_name(config: Dict) -> str:
    """Return the actual model name to use based on config flags."""
    model_cfg = config.get("model", {})
    if model_cfg.get("use_tiny_model", False):
        return model_cfg.get("tiny_model_name", "facebook/opt-125m")
    return model_cfg.get("name", "meta-llama/Meta-Llama-3-8B")


def get_max_context_length(config: Dict) -> int:
    """Return the maximum context length from eval config."""
    lengths = config.get("evaluation", {}).get("context_lengths", [2048])
    return max(lengths)


def compute_cache_budget(config: Dict) -> int:
    """Compute absolute cache budget from ratio and max context length."""
    ratio = config.get("cache", {}).get("budget_ratio", 0.1)
    max_len = get_max_context_length(config)
    return max(1, int(ratio * max_len))


def validate_config(config: Dict) -> None:
    """Raise ValueError for obviously wrong config values."""
    required_sections = ["model", "cache", "policy", "distributed", "evaluation"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: '{section}'")

    ratio = config["cache"].get("budget_ratio", 0.1)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"cache.budget_ratio must be in (0, 1], got {ratio}")
