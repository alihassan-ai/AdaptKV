#!/usr/bin/env python3
"""
Experiment 4: Async Prefetching Latency Hiding
================================================
CLAIM: Non-blocking CUDA stream prefetching hides >30% of communication
latency by overlapping data transfer with attention computation.
"""

import os, sys
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.distributed.prefetcher import measure_prefetch_overlap, benchmark_single_gpu_streams
from src.utils import get_system_info, save_result

SIZES_MB = [1.0, 5.0, 10.0, 25.0, 50.0]
N_REPS   = 15
COLORS = {"sync": "#EF4444", "async": "#3B82F6", "compute": "#94A3B8"}


def make_fns(n_elems: int, device: str):
    dtype = torch.float16 if "cuda" in device else torch.float32
    side  = max(64, int(n_elems ** 0.5))
    A     = torch.randn(side, side, dtype=dtype, device=device)
    B     = torch.randn(side, side, dtype=dtype, device=device)
    src   = torch.randn(n_elems, dtype=dtype, device=device)
    dst   = torch.empty(n_elems, dtype=dtype, device=device)

    def compute_fn():
        torch.mm(A, B)

    def copy_fn():
        dst.copy_(src, non_blocking=True)

    return compute_fn, copy_fn


def run(device: str = "cuda:0", save_dir: str = "results"):
    os.makedirs(f"{save_dir}/charts", exist_ok=True)
    print("\n[Exp 4] Async Prefetching Latency Hiding")

    rows = []
    for size_mb in SIZES_MB:
        n_elems = int(size_mb * 1024**2 / 2)   # fp16 = 2 bytes
        compute_fn, copy_fn = make_fns(n_elems, device)
        result_row = measure_prefetch_overlap(compute_fn, copy_fn, device, n_reps=N_REPS)
        result_row["size_mb"] = size_mb
        rows.append(result_row)
        print(f"  {size_mb:5.1f} MB │ sync={result_row['sync_ms']:.2f}ms  "
              f"async={result_row['async_ms']:.2f}ms  "
              f"overlap={result_row['overlap_pct']:.1f}%")

    overall_overlap = float(np.mean([r["overlap_pct"] for r in rows]))
    best_row        = max(rows, key=lambda r: r["overlap_pct"])
    print(f"\n  Average overlap: {overall_overlap:.1f}%")
    print(f"  Best overlap: {best_row['overlap_pct']:.1f}% at {best_row['size_mb']:.0f} MB")

    stream_bench = benchmark_single_gpu_streams(tensor_shape=(4096, 256), device=device, n_reps=N_REPS)

    sizes    = [r["size_mb"]    for r in rows]
    sync_ms  = [r["sync_ms"]    for r in rows]
    async_ms = [r["async_ms"]   for r in rows]
    comp_ms  = [r["compute_ms"] for r in rows]
    overlaps = [r["overlap_pct"]for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sizes, sync_ms,  "o-", color=COLORS["sync"],    label="Sequential (sync)",  linewidth=2)
    ax1.plot(sizes, async_ms, "s-", color=COLORS["async"],   label="Overlapped (async)", linewidth=2)
    ax1.plot(sizes, comp_ms,  "^--", color=COLORS["compute"], label="Compute only",       linewidth=1.5)
    ax1.fill_between(sizes, async_ms, sync_ms, alpha=0.12, color=COLORS["async"], label="Latency saved")
    ax1.set_xlabel("Transfer Size (MB)", fontsize=11)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Exp 4: Async Prefetch Latency Hiding", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(sizes, overlaps, "D-", color=COLORS["async"], linewidth=2, markersize=8)
    ax2.fill_between(sizes, 0, overlaps, alpha=0.15, color=COLORS["async"])
    ax2.axhline(y=overall_overlap, color="gray", linestyle="--", linewidth=1.5,
                label=f"Mean: {overall_overlap:.1f}%")
    ax2.set_xlabel("Transfer Size (MB)", fontsize=11)
    ax2.set_ylabel("Overlap Ratio (%)", fontsize=11)
    ax2.set_title("Communication Latency Hidden (%)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = f"{save_dir}/charts/exp4_async_prefetch.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    proven = overall_overlap > 30.0
    result = {
        "experiment":          "exp4_async_prefetch",
        "timestamp":           datetime.now().isoformat(),
        "device":              device,
        "system":              get_system_info(),
        "sizes_mb":            SIZES_MB,
        "measurements":        rows,
        "overall_overlap_pct": overall_overlap,
        "stream_benchmark":    stream_bench,
        "chart":               chart_path,
        "proven":              proven,
        "claim":               "Async prefetch hides >30% of communication latency",
    }
    save_result(result, f"{save_dir}/exp4_results.json")
    print(f"  RESULT: {'PROVEN ✓' if proven else 'NOT PROVEN ✗'}")
    return result


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run(device)
