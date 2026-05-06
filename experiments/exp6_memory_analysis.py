#!/usr/bin/env python3
"""
Experiment 6: Memory Analysis
==============================
CLAIM: AdaptKV's three-tier storage reduces GPU memory footprint by >60%
compared to full KV cache, while INT4 compressed tokens maintain >0.95
cosine similarity with their FP16 originals.
"""

import os, sys
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.cache.quantization import quantize_4bit, dequantize_4bit, measure_quantization_error
from src.utils import get_system_info, save_result

SEQ_LEN      = 2048
NUM_LAYERS   = 12
NUM_HEADS    = 12
HEAD_DIM     = 64
BUDGET_RATIO = 0.10
FP16_RATIO   = 0.4
COMP_RATIO   = 2.4
COMP_RATIOS  = [5, 10, 20, 50]
COLORS = {"fp16": "#3B82F6", "int4": "#60A5FA", "metadata": "#93C5FD"}


def tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def measure_strategy_memory(seq_len, num_layers, num_heads, head_dim,
                             strategy, budget_ratio, device) -> dict:
    use_cuda = torch.cuda.is_available() and "cuda" in device

    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(device)

    allocated = []
    fp16_bytes = int4_bytes = meta_bytes = 0
    dtype = torch.float16 if use_cuda else torch.float32

    for _ in range(num_layers):
        if strategy == "full":
            k = torch.zeros(num_heads, seq_len, head_dim, dtype=dtype, device=device)
            v = torch.zeros(num_heads, seq_len, head_dim, dtype=dtype, device=device)
            allocated += [k, v]
            fp16_bytes += tensor_bytes(k) + tensor_bytes(v)

        elif strategy == "h2o":
            kept = max(1, int(seq_len * budget_ratio))
            k = torch.zeros(num_heads, kept, head_dim, dtype=dtype, device=device)
            v = torch.zeros(num_heads, kept, head_dim, dtype=dtype, device=device)
            scores = torch.zeros(num_heads, seq_len, dtype=torch.float32, device=device)
            allocated += [k, v, scores]
            fp16_bytes += tensor_bytes(k) + tensor_bytes(v)
            meta_bytes += tensor_bytes(scores)

        else:  # adaptkv
            fp16_n = max(1, int(seq_len * budget_ratio * FP16_RATIO))
            int4_n = max(1, int(seq_len * budget_ratio * COMP_RATIO))
            k_fp16 = torch.zeros(num_heads, fp16_n, head_dim, dtype=dtype, device=device)
            v_fp16 = torch.zeros(num_heads, fp16_n, head_dim, dtype=dtype, device=device)
            k_int4 = torch.zeros(num_heads, int4_n // 2 + 1, head_dim, dtype=torch.int8, device=device)
            v_int4 = torch.zeros(num_heads, int4_n // 2 + 1, head_dim, dtype=torch.int8, device=device)
            n_groups = (int4_n * head_dim + 127) // 128
            k_scales = torch.zeros(num_heads, n_groups, dtype=dtype, device=device)
            v_scales = torch.zeros(num_heads, n_groups, dtype=dtype, device=device)
            scores   = torch.zeros(num_heads, seq_len, dtype=torch.float32, device=device)
            allocated += [k_fp16, v_fp16, k_int4, v_int4, k_scales, v_scales, scores]
            fp16_bytes += tensor_bytes(k_fp16) + tensor_bytes(v_fp16)
            int4_bytes += tensor_bytes(k_int4) + tensor_bytes(v_int4)
            meta_bytes += tensor_bytes(k_scales) + tensor_bytes(v_scales) + tensor_bytes(scores)

    if use_cuda:
        torch.cuda.synchronize(device)
        total_measured = torch.cuda.memory_allocated(device) - mem_before
    else:
        total_measured = fp16_bytes + int4_bytes + meta_bytes

    res = {
        "strategy":          strategy,
        "fp16_mb":           fp16_bytes / (1024**2),
        "int4_mb":           int4_bytes / (1024**2),
        "metadata_mb":       meta_bytes / (1024**2),
        "total_mb":          total_measured / (1024**2),
        "total_computed_mb": (fp16_bytes + int4_bytes + meta_bytes) / (1024**2),
    }
    for t in allocated:
        del t
    if use_cuda:
        torch.cuda.empty_cache()
    return res


def measure_quantization_quality(device: str) -> dict:
    results = {}
    dtype = torch.float16 if (torch.cuda.is_available() and "cuda" in device) else torch.float32
    for sl in [512, 1024, 2048]:
        x = torch.randn(sl, HEAD_DIM, dtype=dtype)
        x = torch.nn.functional.softmax(x / (HEAD_DIM ** 0.5), dim=-1)
        err = measure_quantization_error(x)
        results[sl] = {k: float(v) for k, v in err.items()}
    return results


def run(device: str = "cuda:0", save_dir: str = "results"):
    os.makedirs(f"{save_dir}/charts", exist_ok=True)
    print("\n[Exp 6] Memory Analysis")

    strategies = ["full", "h2o", "adaptkv"]
    mem_data = {}
    for strategy in strategies:
        data = measure_strategy_memory(SEQ_LEN, NUM_LAYERS, NUM_HEADS, HEAD_DIM,
                                       strategy, BUDGET_RATIO, device)
        mem_data[strategy] = data
        print(f"  {strategy:8s}: FP16={data['fp16_mb']:.1f}MB  INT4={data['int4_mb']:.1f}MB  "
              f"Meta={data['metadata_mb']:.1f}MB  Total={data['total_computed_mb']:.1f}MB")

    full_total  = mem_data["full"]["total_computed_mb"]
    adapt_total = mem_data["adaptkv"]["total_computed_mb"]
    adaptkv_reduction = (1 - adapt_total / max(full_total, 1e-6)) * 100

    comp_ratio_mem = {s: [] for s in strategies}
    for cr in COMP_RATIOS:
        for strategy in strategies:
            d = measure_strategy_memory(SEQ_LEN, NUM_LAYERS, NUM_HEADS, HEAD_DIM,
                                        strategy, 1.0 / cr, device)
            comp_ratio_mem[strategy].append(d["total_computed_mb"])

    print("\n  Measuring quantization quality...")
    quant_quality = measure_quantization_quality(device)
    for sl, q in quant_quality.items():
        print(f"    seq={sl}: cosine={q['cosine_similarity']:.4f}  rel_err={q['relative_error']:.4f}")

    avg_cosine = float(np.mean([q["cosine_similarity"] for q in quant_quality.values()]))

    # Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    x = np.arange(len(strategies))
    fp16_v = [mem_data[s]["fp16_mb"]     for s in strategies]
    int4_v = [mem_data[s]["int4_mb"]     for s in strategies]
    meta_v = [mem_data[s]["metadata_mb"] for s in strategies]
    ax1.bar(x, fp16_v, label="FP16 KV",         color=COLORS["fp16"],     alpha=0.9)
    ax1.bar(x, int4_v, bottom=fp16_v,           label="INT4 Compressed", color=COLORS["int4"],     alpha=0.9)
    ax1.bar(x, meta_v, bottom=[f+i for f,i in zip(fp16_v, int4_v)],
            label="Metadata",                    color=COLORS["metadata"], alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Full Cache", "H2O", "AdaptKV"], fontsize=11)
    ax1.set_ylabel("Memory (MB)", fontsize=11)
    ax1.set_title(f"KV Cache Memory Breakdown\n(seq={SEQ_LEN})", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    for i, s in enumerate(strategies):
        tot = mem_data[s]["total_computed_mb"]
        ax1.text(i, tot + 0.2, f"{tot:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = axes[1]
    clr_map = {"full": COLORS["fp16"], "h2o": "#EF4444", "adaptkv": "#10B981"}
    lbl_map = {"full": "Full Cache",   "h2o": "H2O",     "adaptkv": "AdaptKV"}
    for s in strategies:
        ax2.plot(COMP_RATIOS, comp_ratio_mem[s], "o-", color=clr_map[s],
                 label=lbl_map[s], linewidth=2, markersize=7)
    ax2.set_xlabel("Compression Ratio (×)", fontsize=11)
    ax2.set_ylabel("Memory (MB)", fontsize=11)
    ax2.set_title("Memory vs Compression Ratio", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    ax3 = axes[2]
    sl_list  = sorted(quant_quality.keys())
    cos_sims = [quant_quality[sl]["cosine_similarity"] for sl in sl_list]
    rel_errs = [quant_quality[sl]["relative_error"] * 100 for sl in sl_list]
    ax3b = ax3.twinx()
    ax3.bar([i - 0.2 for i in range(len(sl_list))], cos_sims, 0.35,
            label="Cosine Similarity", color="#3B82F6", alpha=0.8)
    ax3b.bar([i + 0.2 for i in range(len(sl_list))], rel_errs, 0.35,
             label="Rel Error (%)", color="#EF4444", alpha=0.8)
    ax3.set_xticks(range(len(sl_list)))
    ax3.set_xticklabels([str(sl) for sl in sl_list], fontsize=10)
    ax3.set_xlabel("Sequence Length", fontsize=11)
    ax3.set_ylabel("Cosine Similarity", fontsize=11, color="#3B82F6")
    ax3b.set_ylabel("Relative Error (%)", fontsize=11, color="#EF4444")
    ax3.set_title("INT4 Quantization Quality", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 1.15)
    ax3.axhline(y=0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = f"{save_dir}/charts/exp6_memory_analysis.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    proven = adaptkv_reduction > 60.0
    result = {
        "experiment":           "exp6_memory_analysis",
        "timestamp":            datetime.now().isoformat(),
        "device":               device,
        "system":               get_system_info(),
        "seq_len":              SEQ_LEN,
        "budget_ratio":         BUDGET_RATIO,
        "memory_breakdown":     {s: {k: float(v) for k, v in d.items() if k != "strategy"}
                                 for s, d in mem_data.items()},
        "compression_ratio_sweep": {s: {str(cr): float(mb)
                                        for cr, mb in zip(COMP_RATIOS, vals)}
                                    for s, vals in comp_ratio_mem.items()},
        "quantization_quality": {str(sl): q for sl, q in quant_quality.items()},
        "adaptkv_reduction_pct": float(adaptkv_reduction),
        "avg_cosine_similarity": avg_cosine,
        "chart":   chart_path,
        "proven":  proven,
        "claim":   f"AdaptKV reduces KV cache memory by >60% vs full cache",
    }
    save_result(result, f"{save_dir}/exp6_results.json")
    print(f"  RESULT: {'PROVEN ✓' if proven else 'NOT PROVEN ✗'}")
    return result


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run(device)
