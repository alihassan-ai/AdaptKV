"""Load HuggingFace models with CPU/GPU compatibility and optional tiny-model fallback."""

import logging
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading with CUDA/CPU and size flexibility."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_info: Optional[Dict] = None

    def _resolve_model_name(self) -> str:
        model_cfg = self.config.get("model", {})
        if model_cfg.get("use_tiny_model", False):
            name = model_cfg.get("tiny_model_name", "facebook/opt-125m")
            logger.info(f"CPU debug mode: loading tiny model '{name}'")
            return name
        return model_cfg.get("name", "meta-llama/Meta-Llama-3-8B")

    def _resolve_dtype(self) -> torch.dtype:
        dtype_str = self.config.get("model", {}).get("dtype", "float32")
        if dtype_str == "float16" and self.device == "cpu":
            logger.warning("float16 not well-supported on CPU; falling back to float32.")
            return torch.float32
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}.get(dtype_str, torch.float32)

    def load(self) -> Dict[str, Any]:
        """Load model and tokenizer, return a model info dict."""
        if self._model_info is not None:
            return self._model_info

        model_name = self._resolve_model_name()
        dtype = self._resolve_dtype()

        logger.info(f"Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model '{model_name}' on {self.device} with dtype={dtype}...")
        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "output_attentions": False,
            }
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if self.device == "cpu":
                model = model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

        model.eval()
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        num_layers, num_heads, head_dim = self._extract_arch_params(model, model_config)

        self._model = model
        self._tokenizer = tokenizer
        self._model_info = {
            "model": model,
            "tokenizer": tokenizer,
            "model_config": model_config,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "device": self.device,
            "dtype": dtype,
            "model_name": model_name,
        }
        logger.info(f"Model loaded: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")
        return self._model_info

    def _extract_arch_params(self, model, model_config) -> tuple:
        """Extract num_layers, num_heads, head_dim from model config."""
        try:
            # LLaMA-style
            num_layers = model_config.num_hidden_layers
            num_heads = model_config.num_attention_heads
            hidden_size = model_config.hidden_size
            head_dim = hidden_size // num_heads
        except AttributeError:
            try:
                # OPT-style
                num_layers = model_config.num_hidden_layers
                num_heads = model_config.num_attention_heads
                hidden_size = model_config.hidden_size
                head_dim = hidden_size // num_heads
            except AttributeError:
                # Fallback: count layers from model
                num_layers = sum(1 for _ in model.children())
                num_heads = 8
                head_dim = 64
                logger.warning("Could not infer arch params; using defaults.")
        return num_layers, num_heads, head_dim

    def get_num_kv_heads(self) -> int:
        """Return num KV heads (may differ from num query heads for GQA)."""
        info = self._model_info
        if info is None:
            return 0
        cfg = info["model_config"]
        return getattr(cfg, "num_key_value_heads", info["num_heads"])


def load_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to load model and return info dict."""
    torch.manual_seed(42)
    loader = ModelLoader(config)
    return loader.load()
