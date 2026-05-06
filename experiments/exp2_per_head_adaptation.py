#!/usr/bin/env python3
"""
Experiment 2: Per-Head vs Uniform Policy
==========================================
CLAIM: Adapting compression strategy per attention head type gives better
attention retention than applying one uniform strategy to all heads.
"""

import os, sys
from collections import defaultdict
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
from src.policy.head_classifier import HeadClassifier, HEAD_LOCAL, HEAD_GLOBAL, HEAD_SINK
from src.utils import get_system_info, save_result

BUDGET_RATIO = 0.10
QUALITY      = 0.95
NUM_PROMPTS  = 15
MAX_SEQ_LEN  = 256
COLORS = {"adaptkv": "#3B82F6", "h2o": "#EF4444"}

CALIBRATION_PROMPTS = [
    "The neural network was trained on millions of examples to learn complex patterns.",
    "Scientists discovered a new exoplanet orbiting a distant star in the Milky Way galaxy.",
    "The government announced new policies to address economic inequality and social welfare.",
    "Researchers published a groundbreaking study on the effects of meditation on the brain.",
    "The ancient city was excavated revealing artifacts from civilizations thousands of years old.",
]

EVAL_PROMPTS = [
    "Quantum mechanics describes the behavior of particles at subatomic scales using wave functions.",
    "The Amazon rainforest produces 20% of the world's oxygen and houses an extraordinary diversity of species.",
    "Machine learning models require large amounts of labeled training data to achieve high accuracy.",
    "The Berlin Wall fell in 1989 marking the end of the Cold War division between East and West Germany.",
    "Proteins are large biomolecules consisting of amino acid chains folded into functional three-dimensional shapes.",
    "The global financial crisis of 2008 was triggered by the collapse of mortgage-backed securities markets.",
    "Photonic quantum computers use photons to perform computations at speeds beyond classical silicon processors.",
    "The Great Wall of China stretches over 21,000 kilometers and was built across many dynasties.",
    "CRISPR-Cas9 allows scientists to edit DNA sequences with unprecedented precision and efficiency.",
    "Social media algorithms optimize for engagement which can amplify extreme content and misinformation.",
    "The Hubble Space Telescope has captured images of galaxies more than 13 billion light-years away.",
    "Renewable energy sources now account for over 30% of global electricity generation capacity.",
    "The human immune system uses antibodies to recognize and neutralize foreign pathogens.",
    "Large language models are trained by predicting the next token in sequences of text.",
    "The Industrial Revolution began in Britain in the 18th century and transformed manufacturing worldwide.",
]


def uniform_retention(attn_layer: torch.Tensor, budget_ratio: float) -> float:
    H, S, _ = attn_layer.shape
    budget_k = max(1, int(S * budget_ratio))
    retained = []
    for h in range(H):
        imp = attn_layer[h].mean(dim=0)
        fp16_idx, comp_idx = adaptkv_select_tokens(imp, budget_k, FP16_RATIO, COMPRESS_RATIO)
        total = imp.sum().item()
        ret   = (imp[fp16_idx].sum() + imp[comp_idx].sum() * QUALITY).item()
        retained.append(ret / max(total, 1e-8))
    return float(np.mean(retained))


def perhead_retention(attn_layer: torch.Tensor, head_types: list, budget_ratio: float) -> float:
    H, S, _ = attn_layer.shape
    budget_k = max(1, int(S * budget_ratio))
    retained = []
    for h in range(H):
        imp    = attn_layer[h].mean(dim=0)
        total  = imp.sum().item()
        ht     = head_types[h] if h < len(head_types) else HEAD_LOCAL
        params = HeadClassifier.TYPE_PARAMS[ht]

        if ht == HEAD_SINK:
            sink_n = min(4, S)
            remain_k = budget_k - sink_n
            sorted_idx = imp.argsort(descending=True)
            non_sink = [i for i in sorted_idx.tolist() if i >= sink_n]
            fp16_idx = list(range(sink_n))
            comp_k   = min(len(non_sink), max(0, remain_k + int(budget_k * params["compress_ratio"])))
            comp_idx = non_sink[:comp_k]
        else:
            fp16_idx, comp_idx = adaptkv_select_tokens(
                imp, budget_k, params["fp16_ratio"], params["compress_ratio"]
            )

        fp16_mass = imp[fp16_idx].sum().item() if fp16_idx else 0.0
        comp_mass = imp[comp_idx].sum().item() * QUALITY if comp_idx else 0.0
        retained.append((fp16_mass + comp_mass) / max(total, 1e-8))
    return float(np.mean(retained))


def run(device: str = "cuda:0", save_dir: str = "results"):
    os.makedirs(f"{save_dir}/charts", exist_ok=True)
    print("\n[Exp 2] Per-Head vs Uniform Policy")

    model, tokenizer, model_name = load_model_for_experiments(device)

    print("  Classifying attention heads on calibration set...")
    calib_per_layer = defaultdict(list)
    for prompt in CALIBRATION_PROMPTS:
        attns, _ = get_attention_weights(model, tokenizer, prompt, device, MAX_SEQ_LEN)
        for l, a in enumerate(attns):
            calib_per_layer[l].append(a)

    avg_per_layer = [
        torch.stack(calib_per_layer[l]).mean(dim=0)
        for l in range(len(calib_per_layer))
    ]
    classifier = HeadClassifier().fit(avg_per_layer)
    head_types_per_layer = classifier.head_types

    counts = classifier.stats["counts"]
    print(f"  Head types: local={counts['local']}  global={counts['global']}  sink={counts['sink']}")

    uniform_results = defaultdict(list)
    perhead_results = defaultdict(list)
    pertype_uniform = defaultdict(list)
    pertype_perhead = defaultdict(list)

    for prompt in tqdm(EVAL_PROMPTS[:NUM_PROMPTS], desc="  Eval prompts"):
        try:
            attns, _ = get_attention_weights(model, tokenizer, prompt, device, MAX_SEQ_LEN)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue

        for l, layer_attn in enumerate(attns):
            if layer_attn.shape[-1] < 8:
                continue
            H = layer_attn.shape[0]
            htypes = head_types_per_layer[l].tolist() if l < len(head_types_per_layer) else []

            uniform_results[l].append(uniform_retention(layer_attn, BUDGET_RATIO))
            perhead_results[l].append(perhead_retention(layer_attn, htypes, BUDGET_RATIO))

            bk = max(1, int(layer_attn.shape[-1] * BUDGET_RATIO))
            for h in range(min(H, len(htypes))):
                ht  = htypes[h]
                imp = layer_attn[h].mean(dim=0)
                total = imp.sum().item()
                fp_i, cp_i = adaptkv_select_tokens(imp, bk, FP16_RATIO, COMPRESS_RATIO)
                u_r = (imp[fp_i].sum() + imp[cp_i].sum() * QUALITY).item() / max(total, 1e-8)
                pertype_uniform[ht].append(u_r)

                ph_p = HeadClassifier.TYPE_PARAMS[ht]
                if ht == HEAD_SINK:
                    sn = min(4, imp.shape[0])
                    non_s = [i for i in imp.argsort(descending=True).tolist() if i >= sn]
                    ck = min(len(non_s), bk + int(bk * ph_p["compress_ratio"]))
                    ci = non_s[:ck]
                    p_r = (imp[list(range(sn))].sum() + imp[ci].sum() * QUALITY).item() / max(total, 1e-8)
                else:
                    fp2, cp2 = adaptkv_select_tokens(imp, bk, ph_p["fp16_ratio"], ph_p["compress_ratio"])
                    p_r = (imp[fp2].sum() + imp[cp2].sum() * QUALITY).item() / max(total, 1e-8)
                pertype_perhead[ht].append(p_r)

    all_u = [v for vals in uniform_results.values() for v in vals]
    all_p = [v for vals in perhead_results.values() for v in vals]
    U_mean, U_std = float(np.mean(all_u)), float(np.std(all_u))
    P_mean, P_std = float(np.mean(all_p)), float(np.std(all_p))

    print(f"  Uniform:  {U_mean*100:.2f}% ± {U_std*100:.2f}%")
    print(f"  Per-head: {P_mean*100:.2f}% ± {P_std*100:.2f}%")
    print(f"  Δ = {(P_mean-U_mean)*100:+.2f}pp")

    # Chart
    type_names = {HEAD_LOCAL: "Local", HEAD_GLOBAL: "Global", HEAD_SINK: "Sink"}
    type_keys  = [HEAD_LOCAL, HEAD_GLOBAL, HEAD_SINK]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels = ["Uniform Policy", "Per-Head Policy"]
    vals   = [U_mean * 100, P_mean * 100]
    stds   = [U_std  * 100, P_std  * 100]
    bars = ax1.bar(labels, vals, yerr=stds, color=[COLORS["h2o"], COLORS["adaptkv"]],
                   alpha=0.85, capsize=6, width=0.5)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Attention Mass Retained (%)", fontsize=11)
    ax1.set_title("Overall: Per-Head vs Uniform Policy", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    x = np.arange(len(type_keys)); w = 0.35
    u_type = [np.mean(pertype_uniform.get(t, [0])) * 100 for t in type_keys]
    p_type = [np.mean(pertype_perhead.get(t,  [0])) * 100 for t in type_keys]
    ax2.bar(x - w/2, u_type, w, label="Uniform",  color=COLORS["h2o"],     alpha=0.85)
    ax2.bar(x + w/2, p_type, w, label="Per-Head", color=COLORS["adaptkv"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([type_names[t] for t in type_keys], fontsize=11)
    ax2.set_ylabel("Attention Mass Retained (%)", fontsize=11)
    ax2.set_title("Per Head-Type Breakdown", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = f"{save_dir}/charts/exp2_per_head_adaptation.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    proven = P_mean > U_mean
    result = {
        "experiment":      "exp2_per_head_adaptation",
        "timestamp":       datetime.now().isoformat(),
        "model_name":      model_name,
        "system":          get_system_info(),
        "head_type_counts": counts,
        "uniform_mean":    U_mean,
        "uniform_std":     U_std,
        "perhead_mean":    P_mean,
        "perhead_std":     P_std,
        "improvement_pp":  float((P_mean - U_mean) * 100),
        "per_type": {
            type_names[t]: {
                "uniform":  float(np.mean(pertype_uniform.get(t, [0]))),
                "per_head": float(np.mean(pertype_perhead.get(t, [0]))),
            } for t in type_keys
        },
        "chart":  chart_path,
        "proven": proven,
        "claim":  "Per-head policy retains more attention mass than uniform policy",
    }
    save_result(result, f"{save_dir}/exp2_results.json")
    print(f"  RESULT: {'PROVEN ✓' if proven else 'NOT PROVEN ✗'}")
    return result


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run(device)
