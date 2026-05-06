"""CUDA-accurate timing, GPU memory helpers, system info, result serialization."""

import csv
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import torch


class CUDATimer:
    """Measure elapsed GPU time using CUDA Events (not wall-clock)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._use_cuda = torch.cuda.is_available() and "cuda" in device
        self._start_event = None
        self._end_event = None
        self._cpu_start: Optional[float] = None
        self._elapsed_ms: float = 0.0

    def __enter__(self):
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event   = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
        else:
            self._elapsed_ms = (time.perf_counter() - self._cpu_start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        if self._use_cuda and self._start_event and self._end_event:
            return self._start_event.elapsed_time(self._end_event)
        return self._elapsed_ms


def cuda_time_fn(fn, device: str = "cuda", warmup: int = 1, repeats: int = 5) -> float:
    """Run `fn()` and return median elapsed milliseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        with CUDATimer(device) as t:
            fn()
        times.append(t.elapsed_ms)

    times.sort()
    return times[len(times) // 2]


def gpu_memory_mb(device_idx: int = 0) -> Dict[str, float]:
    """Return current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    dev = f"cuda:{device_idx}"
    return {
        "allocated_mb": torch.cuda.memory_allocated(dev) / 1024**2,
        "reserved_mb":  torch.cuda.memory_reserved(dev)  / 1024**2,
        "peak_mb":      torch.cuda.max_memory_allocated(dev) / 1024**2,
    }


def reset_peak_memory(device_idx: int = 0) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(f"cuda:{device_idx}")


def get_system_info() -> Dict:
    """Return GPU and software environment info."""
    info: Dict = {
        "timestamp":       datetime.now().isoformat(),
        "torch_version":   torch.__version__,
        "cuda_available":  torch.cuda.is_available(),
        "gpu_count":       torch.cuda.device_count(),
        "gpus":            [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info["gpus"].append({
            "index":              i,
            "gpu_name":           p.name,
            "total_vram_gb":      round(p.total_memory / 1024**3, 1),
            "compute_capability": f"{p.major}.{p.minor}",
        })
    # Convenience flat fields used by dashboard
    if info["gpus"]:
        info["gpu_name"]      = info["gpus"][0]["gpu_name"]
        info["total_vram_gb"] = info["gpus"][0]["total_vram_gb"]
        info["device"]        = "cuda:0"
    else:
        info["gpu_name"]      = "N/A"
        info["total_vram_gb"] = "N/A"
        info["device"]        = "cpu"
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    return info


class GPULogger:
    """Polls GPU utilisation and memory every `interval_s` seconds to a CSV."""

    def __init__(self, log_path: str = "results/gpu_log.csv", interval_s: float = 2.0):
        self.log_path = log_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            use_pynvml = True
        except Exception:
            use_pynvml = False

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu", "util_pct", "mem_used_mb", "mem_total_mb"])
            while not self._stop.is_set():
                ts = datetime.now().isoformat()
                if use_pynvml:
                    for i, h in enumerate(handles):
                        try:
                            u = pynvml.nvmlDeviceGetUtilizationRates(h)
                            m = pynvml.nvmlDeviceGetMemoryInfo(h)
                            writer.writerow([ts, i, u.gpu,
                                             m.used // 1024**2,
                                             m.total // 1024**2])
                        except Exception:
                            pass
                    f.flush()
                self._stop.wait(self.interval_s)


def save_result(data: Dict, path: str) -> None:
    """Write a dict to a JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
    print(f"  Saved → {path}")


def _serialize(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
