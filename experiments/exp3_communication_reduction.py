#!/usr/bin/env python3
"""
Experiment 3: Communication-Aware Placement
=============================================
CLAIM: Adding a communication cost penalty to eviction scoring reduces
inter-GPU data transfer by >20%.
"""

import os, sys
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.model_loader import load_model_for_experiments, get_attention_weights
from src.distributed.cache_sharding import compute_comm_volume
from src.utils import get_system_info, save_result

NUM_VIRTUAL_GPUS = 4
BUDGET_RATIO     = 0.10
LAMBDA           = 0.15
NUM_PROMPTS      = 15
MAX_SEQ_LEN      = 256
SEQ_LENGTHS      = [128, 256, 512, 1024, 2048]
COLORS = {"naive": "#EF4444", "comm_aware": "#3B82F6"}

PROMPTS = [
    "Distributed machine learning systems face significant communication bottlenecks across GPU boundaries.",
    "The attention mechanism in transformers requires access to all previous key-value pairs in the cache.",
    "Network topology affects the communication patterns and latency in distributed deep learning training.",
    "Pipeline parallelism divides a neural network into stages each running on different hardware accelerators.",
    "NVLink provides high-bandwidth GPU-to-GPU communication for multi-GPU server configurations.",
    "Tensor parallelism splits individual weight matrices across multiple GPUs reducing per-device memory.",
    "The transformer block consists of multi-head self-attention followed by a feed-forward network.",
    "Memory-efficient attention algorithms reduce the quadratic memory requirement of standard attention.",
    "Gradient checkpointing trades compute for memory by recomputing activations during the backward pass.",
    "NCCL optimizes collective communication operations across NVIDIA GPUs in distributed training.",
    "Flash attention uses tiling to keep attention computations in fast SRAM rather than HBM.",
    "PagedAttention manages KV cache memory in pages inspired by virtual memory systems in operating systems.",
    "Speculative decoding uses a small draft model to generate candidate tokens verified by a larger model.",
    "Mixed-precision training uses FP16 for forward/backward passes and FP32 for gradient accumulation.",
    "Model parallelism distributes model parameters across GPUs enabling training of very large models.",
]


def simulate_generation_step(importance, budget_ratio, num_gpus, num_heads, head_dim, lambda_):
    S = importance.shape[0]
    budget_k = max(1, int(S * budget_ratio))
    naive      = compute_comm_volume(importance, budget_k, num_gpus, lambda_, num_heads, head_dim, "naive")
    comm_aware = compute_comm_volume(importance, budget_k, num_gpus, lambda_, num_heads, head_dim, "comm_aware")
    return {"naive": naive, "comm_aware": comm_aware}


def run(device: str = "cuda:0", save_dir: str = "results"):
    os.makedirs(f"{save_dir}/charts", exist_ok=True)
    print("\n[Exp 3] Communication-Aware Placement")

    model, tokenizer, model_name = load_model_for_experiments(device)
    n_heads  = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    naive_mb_all = []
    comm_aware_mb_all = []

    for prompt in tqdm(PROMPTS[:NUM_PROMPTS], desc="  Simulating generation steps"):
        try:
            attns, _ = get_attention_weights(model, tokenizer, prompt, device, MAX_SEQ_LEN)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        for layer_attn in attns:
            importance = layer_attn.mean(dim=0).mean(dim=0)
            out = simulate_generation_step(importance, BUDGET_RATIO, NUM_VIRTUAL_GPUS,
                                           n_heads, head_dim, LAMBDA)
            naive_mb_all.append(out["naive"]["comm_mb"])
            comm_aware_mb_all.append(out["comm_aware"]["comm_mb"])

    naive_mean      = float(np.mean(naive_mb_all))
    comm_aware_mean = float(np.mean(comm_aware_mb_all))
    reduction_pct   = (naive_mean - comm_aware_mean) / max(naive_mean, 1e-6) * 100

    print(f"  Naive:      {naive_mean:.4f} MB/step")
    print(f"  Comm-aware: {comm_aware_mean:.4f} MB/step")
    print(f"  Reduction:  {reduction_pct:.1f}%")

    # Scaling sweep
    seq_len_naive, seq_len_comm_aware = [], []
    for sl in SEQ_LENGTHS:
        importance = torch.rand(sl); importance /= importance.sum()
        bk = max(1, int(sl * BUDGET_RATIO))
        n  = compute_comm_volume(importance, bk, NUM_VIRTUAL_GPUS, LAMBDA, n_heads, head_dim, "naive")
        c  = compute_comm_volume(importance, bk, NUM_VIRTUAL_GPUS, LAMBDA, n_heads, head_dim, "comm_aware")
        seq_len_naive.append(n["comm_mb"])
        seq_len_comm_aware.append(c["comm_mb"])

    # Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cats = ["Naive\n(importance only)", "Comm-Aware\n(importance − λ·remote)"]
    vals = [naive_mean, comm_aware_mean]
    bars = ax1.bar(cats, vals, color=[COLORS["naive"], COLORS["comm_aware"]], alpha=0.85, width=0.5)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-5,
                 f"{bar.get_height():.4f} MB", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Avg. Communication Volume (MB/step)", fontsize=11)
    ax1.set_title(f"Exp 3: Comm Volume Reduction ({reduction_pct:.1f}%)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(SEQ_LENGTHS, seq_len_naive,      "o-", color=COLORS["naive"],      label="Naive",      linewidth=2)
    ax2.plot(SEQ_LENGTHS, seq_len_comm_aware,  "s-", color=COLORS["comm_aware"], label="Comm-Aware", linewidth=2)
    ax2.fill_between(SEQ_LENGTHS, seq_len_comm_aware, seq_len_naive, alpha=0.15,
                     color=COLORS["comm_aware"], label="Savings")
    ax2.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax2.set_ylabel("Communication Volume (MB/step)", fontsize=11)
    ax2.set_title("Comm Volume vs. Sequence Length", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = f"{save_dir}/charts/exp3_communication_reduction.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    proven = reduction_pct > 20.0
    result = {
        "experiment":        "exp3_communication_reduction",
        "timestamp":         datetime.now().isoformat(),
        "model_name":        model_name,
        "system":            get_system_info(),
        "num_virtual_gpus":  NUM_VIRTUAL_GPUS,
        "lambda":            LAMBDA,
        "budget_ratio":      BUDGET_RATIO,
        "naive_mean_mb":     naive_mean,
        "comm_aware_mean_mb": comm_aware_mean,
        "reduction_pct":     reduction_pct,
        "scaling_curve":     {str(sl): {"naive_mb": n, "comm_aware_mb": c}
                              for sl, n, c in zip(SEQ_LENGTHS, seq_len_naive, seq_len_comm_aware)},
        "chart":  chart_path,
        "proven": proven,
        "claim":  "Comm-aware placement reduces inter-GPU transfer by >20%",
    }
    save_result(result, f"{save_dir}/exp3_results.json")
    print(f"  RESULT: {'PROVEN ✓' if proven else 'NOT PROVEN ✗'}")
    return result


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run(device)
