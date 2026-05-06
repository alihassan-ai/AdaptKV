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


# ── Experiment-friendly functional API ────────────────────────────────────────

def load_model_for_experiments(
    device: str = "cuda:0",
    force_small: bool = False,
):
    """Load a model suitable for experiments — auto-selects size by VRAM.

    Decision:
      - HF_TOKEN set + GPU > 30 GB VRAM  →  meta-llama/Meta-Llama-3-8B
      - Otherwise                         →  facebook/opt-1.3b

    Returns: (model, tokenizer, model_name)
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available() and "cuda" in device

    if (not force_small
            and os.environ.get("HF_TOKEN")
            and use_cuda
            and torch.cuda.get_device_properties(device).total_memory > 30 * 1024**3):
        model_name = "meta-llama/Meta-Llama-3-8B"
    else:
        model_name = "facebook/opt-1.3b"

    logger.info(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device} if use_cuda else None,
        attn_implementation="eager",   # required for output_attentions
        token=os.environ.get("HF_TOKEN"),
    )
    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Loaded {model_name} ({n_params:.1f}B params)")
    return model, tokenizer, model_name


def get_attention_weights(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 256,
):
    """Run one forward pass and return per-layer attention tensors.

    Returns:
        attentions: list of [num_heads, seq_len, seq_len] float tensors (one per layer)
        seq_len:    actual sequence length after truncation
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, use_cache=False)

    attentions = [a.squeeze(0).float().cpu() for a in outputs.attentions]
    return attentions, input_ids.shape[1]
