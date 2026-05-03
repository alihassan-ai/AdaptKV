"""GPU memory, latency, and communication volume profiler."""

import logging
import subprocess
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class Profiler:
    """Tracks per-step latency, memory, and communication statistics."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._cuda_available = torch.cuda.is_available()

        self._step_latencies: List[float] = []
        self._compute_times: List[float] = []
        self._comm_times: List[float] = []
        self._step_start: Optional[float] = None

        self._total_tokens_generated: int = 0
        self._wall_start: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Step-level profiling                                                 #
    # ------------------------------------------------------------------ #

    def start_step(self) -> None:
        """Mark the start of a generation step."""
        if self._cuda_available:
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def end_step(self, num_new_tokens: int = 1) -> float:
        """Mark the end of a generation step. Returns step latency in ms."""
        if self._cuda_available:
            torch.cuda.synchronize()
        if self._step_start is None:
            return 0.0
        latency_ms = (time.perf_counter() - self._step_start) * 1000.0
        self._step_latencies.append(latency_ms)
        self._total_tokens_generated += num_new_tokens
        self._step_start = None
        return latency_ms

    def record_compute_time(self, ms: float) -> None:
        self._compute_times.append(ms)

    def record_comm_time(self, ms: float) -> None:
        self._comm_times.append(ms)

    # ------------------------------------------------------------------ #
    # Experiment-level                                                     #
    # ------------------------------------------------------------------ #

    def start_experiment(self) -> None:
        self._wall_start = time.perf_counter()
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats()

    def end_experiment(self) -> Dict:
        elapsed = (time.perf_counter() - self._wall_start) if self._wall_start else 0.0
        return self.get_summary(elapsed)

    # ------------------------------------------------------------------ #
    # GPU utilization (via nvidia-smi)                                    #
    # ------------------------------------------------------------------ #

    def sample_gpu_utilization(self) -> Optional[Dict]:
        """Query nvidia-smi for current GPU utilization and memory.

        Returns None if CUDA is unavailable or nvidia-smi is not found.
        """
        if not self._cuda_available:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            rows = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    rows.append({
                        "utilization_pct": int(parts[0]),
                        "memory_used_mb":  int(parts[1]),
                        "memory_total_mb": int(parts[2]),
                    })
            return {"gpus": rows}
        except Exception as e:
            logger.debug(f"nvidia-smi query failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def get_summary(self, elapsed_seconds: float = 0.0) -> Dict:
        n_steps = len(self._step_latencies)
        latencies = self._step_latencies

        summary: Dict = {
            "total_steps": n_steps,
            "total_tokens_generated": self._total_tokens_generated,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": (self._total_tokens_generated / elapsed_seconds
                                  if elapsed_seconds > 0 else 0.0),
            "latency_mean_ms": (sum(latencies) / n_steps if n_steps else 0.0),
            "latency_p50_ms":  (_percentile(latencies, 50) if latencies else 0.0),
            "latency_p95_ms":  (_percentile(latencies, 95) if latencies else 0.0),
            "latency_p99_ms":  (_percentile(latencies, 99) if latencies else 0.0),
        }

        if self._cuda_available:
            summary["peak_memory_mb"] = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
            summary["current_memory_mb"] = (
                torch.cuda.memory_allocated() / (1024 ** 2)
            )

        if self._compute_times:
            ct = self._compute_times
            summary["compute_mean_ms"] = sum(ct) / len(ct)

        if self._comm_times:
            cmt = self._comm_times
            summary["comm_mean_ms"] = sum(cmt) / len(cmt)

        gpu_stats = self.sample_gpu_utilization()
        if gpu_stats:
            summary["gpu_utilization"] = gpu_stats

        return summary

    def reset(self) -> None:
        self._step_latencies.clear()
        self._compute_times.clear()
        self._comm_times.clear()
        self._total_tokens_generated = 0
        self._step_start = None
        self._wall_start = None


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]
