#!/usr/bin/env python3
"""
Experiment 1: Three-Tier vs Binary Eviction
============================================
CLAIM: AdaptKV's three-tier policy (keep/compress/evict) retains more attention
mass than H2O's binary (keep/evict) at identical effective memory budgets.

Memory equivalence at budget_ratio=0.10:
  H2O:      0.10 × S × 2B  =  0.20 × S bytes
  AdaptKV:  0.04 × S × 2B  +  0.24 × S × 0.5B  =  0.20 × S bytes  ✓
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
from src.cache.adaptkv_cache import adaptkv_select_tokens, FP16_RATIO, COMPRESS_RATIO, COMPRESS_QUALITY
from src.cache.h2o_cache import h2o_select_tokens
from src.utils import get_system_info, save_result

COMPRESSION_RATIOS = [5, 10, 20, 50]
NUM_PROMPTS        = 20
MAX_SEQ_LEN        = 256
COLORS = {"adaptkv": "#3B82F6", "h2o": "#EF4444"}

PROMPTS = [
    "The history of artificial intelligence began in the 1950s when Alan Turing proposed the concept of a machine that could think.",
    "Climate change is accelerating, with global temperatures rising due to greenhouse gas emissions from industrial activities.",
    "The transformer architecture revolutionized natural language processing by enabling parallel training and attention mechanisms.",
    "Quantum computing leverages superposition and entanglement to solve problems intractable for classical computers.",
    "The Roman Empire at its height controlled territories spanning from Britain to Mesopotamia across three continents.",
    "DNA carries genetic information in sequences of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The International Space Station orbits Earth at an altitude of approximately 400 kilometers above the surface.",
    "Economic inequality has grown significantly over the past four decades as wages stagnate and capital returns increase.",
    "Machine learning algorithms improve their performance by finding patterns in large datasets without explicit programming.",
    "The French Revolution began in 1789 and fundamentally transformed political power structures across Europe.",
    "Neural networks are inspired by biological neurons and consist of layers of interconnected processing units.",
    "Ocean acidification threatens marine ecosystems as CO2 dissolves in seawater to form carbonic acid.",
    "The development of antibiotics in the 20th century drastically reduced mortality from bacterial infections.",
    "Blockchain technology provides a decentralized ledger that records transactions across a distributed network.",
    "The Silk Road connected civilizations from China to Rome facilitating trade in goods, ideas, and culture.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plant chloroplasts.",
    "The stock market crash of 1929 triggered the Great Depression, causing widespread unemployment and poverty.",
    "Gravitational waves were first directly detected in 2015 by the LIGO observatory confirming Einstein's predictions.",
    "Deep reinforcement learning agents have achieved superhuman performance in complex games like Go and StarCraft.",
    "The human brain contains approximately 86 billion neurons forming trillions of synaptic connections.",
]


def compute_importance(attn: torch.Tensor) -> torch.Tensor:
    return attn.mean(dim=0).mean(dim=0)   # [S]


def h2o_retention(importance: torch.Tensor, budget_ratio: float) -> float:
    S = importance.shape[0]
    budget_k = max(1, int(S * budget_ratio))
    kept  = h2o_select_tokens(importance, budget_k, recent_k=max(1, int(S * 0.05)))
    total = importance.sum().item()
    return importance[list(kept)].sum().item() / max(total, 1e-8)


def adaptkv_retention(importance: torch.Tensor, budget_ratio: float) -> float:
    S = importance.shape[0]
    budget_k  = max(1, int(S * budget_ratio))
    fp16_idx, comp_idx = adaptkv_select_tokens(importance, budget_k, FP16_RATIO, COMPRESS_RATIO)
    fp16_mass = importance[fp16_idx].sum().item() if fp16_idx else 0.0
    comp_mass = importance[comp_idx].sum().item() * COMPRESS_QUALITY if comp_idx else 0.0
    total     = importance.sum().item()
    return (fp16_mass + comp_mass) / max(total, 1e-8)


def run(device: str = "cuda:0", save_dir: str = "results"):
    os.makedirs(f"{save_dir}/charts", exist_ok=True)
    print("\n[Exp 1] Three-Tier vs Binary Eviction")

    model, tokenizer, model_name = load_model_for_experiments(device)
    results_by_ratio = {}

    for cr in COMPRESSION_RATIOS:
        budget_ratio = 1.0 / cr
        h2o_retentions, adapt_retentions = [], []

        for prompt in tqdm(PROMPTS, desc=f"  {cr}× compression", leave=False):
            try:
                attentions, _ = get_attention_weights(model, tokenizer, prompt, device, MAX_SEQ_LEN)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            for layer_attn in attentions:
                if layer_attn.shape[-1] < 8:
                    continue
                imp = compute_importance(layer_attn)
                h2o_retentions.append(h2o_retention(imp, budget_ratio))
                adapt_retentions.append(adaptkv_retention(imp, budget_ratio))

        h2o_arr   = np.array(h2o_retentions)
        adapt_arr = np.array(adapt_retentions)
        results_by_ratio[cr] = {
            "compression_ratio":   cr,
            "budget_ratio":        budget_ratio,
            "h2o_mean":            float(h2o_arr.mean()),
            "h2o_std":             float(h2o_arr.std()),
            "adaptkv_mean":        float(adapt_arr.mean()),
            "adaptkv_std":         float(adapt_arr.std()),
            "improvement_abs":     float(adapt_arr.mean() - h2o_arr.mean()),
            "improvement_rel_pct": float((adapt_arr.mean() - h2o_arr.mean()) / max(h2o_arr.mean(), 1e-8) * 100),
            "n_measurements":      len(h2o_arr),
        }
        r = results_by_ratio[cr]
        print(f"  {cr:>3}× │ H2O {r['h2o_mean']*100:.1f}%  AdaptKV {r['adaptkv_mean']*100:.1f}%"
              f"  Δ={r['improvement_abs']*100:+.1f}pp")

    # Chart
    ratios     = COMPRESSION_RATIOS
    h2o_vals   = [results_by_ratio[r]["h2o_mean"]   * 100 for r in ratios]
    adapt_vals = [results_by_ratio[r]["adaptkv_mean"]* 100 for r in ratios]
    h2o_stds   = [results_by_ratio[r]["h2o_std"]    * 100 for r in ratios]
    adapt_stds = [results_by_ratio[r]["adaptkv_std"] * 100 for r in ratios]

    x, w = np.arange(len(ratios)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, h2o_vals,   w, yerr=h2o_stds,   label="H2O (binary)",
                color=COLORS["h2o"], alpha=0.85, capsize=4)
    b2 = ax.bar(x + w/2, adapt_vals, w, yerr=adapt_stds, label="AdaptKV (three-tier)",
                color=COLORS["adaptkv"], alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}×" for r in ratios], fontsize=11)
    ax.set_xlabel("Compression Ratio", fontsize=11)
    ax.set_ylabel("Attention Mass Retained (%)", fontsize=11)
    ax.set_title("Exp 1: Three-Tier vs Binary — Attention Retention at Same Memory Budget",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    chart_path = f"{save_dir}/charts/exp1_three_tier_vs_binary.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    proven = all(results_by_ratio[cr]["adaptkv_mean"] > results_by_ratio[cr]["h2o_mean"]
                 for cr in COMPRESSION_RATIOS)

    result = {
        "experiment":  "exp1_three_tier_vs_binary",
        "timestamp":   datetime.now().isoformat(),
        "model_name":  model_name,
        "system":      get_system_info(),
        "results":     results_by_ratio,
        "chart":       chart_path,
        "proven":      proven,
        "claim":       "AdaptKV three-tier retains more attention mass than H2O binary at same memory",
    }
    save_result(result, f"{save_dir}/exp1_results.json")
    print(f"  RESULT: {'PROVEN ✓' if proven else 'NOT PROVEN ✗'}")
    return result


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run(device)
