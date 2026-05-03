from .base_cache import BaseKVCache
from .full_cache import FullCache
from .h2o_cache import H2OCache
from .snapkv_cache import SnapKVCache
from .streaming_cache import StreamingLLMCache
from .adaptkv_cache import AdaptKVCache
from .quantization import quantize_to_nf4, dequantize_from_nf4

CACHE_REGISTRY = {
    "full_cache": FullCache,
    "h2o": H2OCache,
    "snapkv": SnapKVCache,
    "streaming": StreamingLLMCache,
    "adaptkv": AdaptKVCache,
}

def get_cache(method: str, config: dict, num_layers: int, num_heads: int,
              head_dim: int, device: str = "cpu") -> BaseKVCache:
    if method not in CACHE_REGISTRY:
        raise ValueError(f"Unknown cache method '{method}'. Choose from: {list(CACHE_REGISTRY)}")
    return CACHE_REGISTRY[method](config, num_layers, num_heads, head_dim, device)
