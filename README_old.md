# AdaptKV: Adaptive KV Cache Compression for Distributed LLM Inference

**Student:** Ali Hassan (22i-0541) | BS Artificial Intelligence | FAST-NUCES Islamabad  
**Course:** Parallel & Distributed Computing (PDC)  
**Hardware:** RunPod Cloud — 4× NVIDIA A40 (48 GB VRAM each)

---

## Abstract

Large Language Models maintain a Key-Value (KV) cache during autoregressive generation that grows linearly with context length and becomes the dominant memory bottleneck at scale. Existing eviction methods (H2O, SnapKV, StreamingLLM) apply a binary keep-or-evict decision, discarding potentially useful information.

**AdaptKV** introduces a *three-tier* adaptive policy: a small MLP trained via knowledge distillation assigns each cached entry to one of three actions:

| Tier | Action | Storage |
|------|--------|---------|
| 0 | **Keep** | Full FP16 precision |
| 1 | **Compress** | 4-bit NormalFloat (NF4) — 4× smaller |
| 2 | **Evict** | Removed entirely |

The policy is conditioned on per-head features (rolling attention momentum, variance, key redundancy, relative position, head type) and operates with a hard token budget constraint. A distributed cache manager shards the KV cache across multiple GPUs using communication-aware placement and async prefetching.

---

## Installation

```bash
# 1. Clone and enter project
cd adaptKV

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install as a package
pip install -e .
```

**CPU-only Mac development** requires no NVIDIA GPU. The project automatically detects device availability and switches to simulation mode.

---

## Quick Start

### CPU Mode (MacBook — development & testing)

```bash
# Runs with OPT-125m, simulated multi-GPU, short contexts
bash scripts/run_all_experiments.sh cpu
```

This will:
1. Train the policy MLP on synthetic prompts using OPT-125m
2. Evaluate all 5 cache methods (full_cache, H2O, SnapKV, StreamingLLM, AdaptKV)
3. Run Needle-in-a-Haystack and RULER benchmarks
4. Simulate 4-GPU distributed cache management
5. Print and save a comparison table

Expected runtime: ~10–30 minutes on a MacBook.

### GPU Mode (RunPod — full experiments)

```bash
# Upload project to RunPod, then:
bash scripts/run_all_experiments.sh gpu
```

This runs the full pipeline with LLaMA-3-8B on 4× A40, including LongBench.  
Expected runtime: ~4–8 hours for complete evaluation.

---

## Running Individual Components

### Train the policy only

```bash
python scripts/train_policy.py --config configs/cpu_debug.yaml
```

### Run a single baseline

```bash
python scripts/run_baseline.py \
    --config configs/cpu_debug.yaml \
    --method h2o \
    --benchmarks needle ruler
```

Available methods: `full_cache`, `h2o`, `snapkv`, `streaming`

### Run AdaptKV with a trained policy

```bash
python scripts/run_adaptkv.py \
    --config configs/cpu_debug.yaml \
    --policy-path results/policy.pt
```

### Distributed multi-GPU evaluation

```bash
# 4-GPU (RunPod):
torchrun --nproc_per_node=4 scripts/run_distributed.py \
    --config configs/runpod_4gpu.yaml \
    --policy-path results/policy.pt

# CPU simulation:
python scripts/run_distributed.py --config configs/cpu_debug.yaml
```

### Collect and display all results

```bash
python scripts/collect_results.py
```

---

## Results Summary

*(Populated after running experiments)*

| Method | Needle Avg | RULER Avg | LongBench Avg | Mem (GB) |
|--------|-----------|-----------|---------------|----------|
| Full Cache (oracle) | — | — | — | — |
| StreamingLLM | — | — | — | — |
| H2O | — | — | — | — |
| SnapKV | — | — | — | — |
| **AdaptKV (ours)** | — | — | — | — |

---

## Project Structure

```
adaptKV/
├── configs/
│   ├── base_config.yaml          # Default hyperparameters
│   ├── cpu_debug.yaml            # CPU/Mac development mode
│   ├── runpod_4gpu.yaml          # 4× A40 RunPod production
│   └── runpod_8gpu.yaml          # 8× A40 extended experiments
├── src/
│   ├── models/
│   │   └── model_loader.py       # HuggingFace model loading (auto CPU/GPU)
│   ├── cache/
│   │   ├── base_cache.py         # Abstract interface for all strategies
│   │   ├── full_cache.py         # Oracle baseline (no compression)
│   │   ├── h2o_cache.py          # Heavy-Hitter Oracle (Zhang et al. 2023)
│   │   ├── snapkv_cache.py       # SnapKV observation-window (Li et al. 2024)
│   │   ├── streaming_cache.py    # StreamingLLM sink+window (Xiao et al. 2024)
│   │   ├── adaptkv_cache.py      # OUR METHOD: three-tier adaptive cache
│   │   └── quantization.py       # 4-bit NF4 quantization (manual + bnb)
│   ├── policy/
│   │   ├── policy_network.py     # AdaptKV MLP architecture
│   │   ├── feature_extractor.py  # EWMA momentum, variance, redundancy, position
│   │   └── trainer.py            # Distillation training loop + oracle labeling
│   ├── distributed/
│   │   ├── cache_manager.py      # Ring-buffer sharding, CPU simulation
│   │   ├── comm_aware_placement.py # effective_score = importance − λ·comm_cost
│   │   ├── prefetcher.py         # Async double-buffering prefetcher
│   │   └── utils.py              # NCCL helpers, rank/world_size
│   ├── evaluation/
│   │   ├── longbench.py          # 16-task LongBench runner
│   │   ├── needle.py             # Needle-in-a-Haystack (5 depths × N lengths)
│   │   ├── ruler.py              # RULER synthetic task suite
│   │   ├── metrics.py            # Perplexity, F1, ROUGE-L, edit similarity
│   │   └── profiler.py           # GPU memory, latency, comm volume
│   └── utils/
│       ├── config.py             # YAML loading + validation
│       └── logging_utils.py      # Structured experiment logging
├── scripts/
│   ├── run_baseline.py           # Evaluate any single baseline
│   ├── train_policy.py           # Train AdaptKV policy via distillation
│   ├── run_adaptkv.py            # Evaluate AdaptKV
│   ├── run_distributed.py        # Multi-GPU distributed evaluation
│   ├── run_all_experiments.sh    # Master pipeline script
│   └── collect_results.py        # Aggregate JSON → comparison table
└── results/                      # Output directory (JSON + logs)
```

---

## Key Algorithms

### AdaptKV Policy Features

| # | Feature | Computation |
|---|---------|-------------|
| 0 | Attention momentum | EWMA(α=0.1) of attention weights over last 32 steps |
| 1 | Attention variance | Var of attention over the rolling window |
| 2 | Key redundancy | max cosine similarity with all other keys in the head |
| 3 | Relative position | position_index / total_cached_length |
| 4 | Head type | 0=local, 1=global, 2=sink (clustered from calibration entropy) |

### Communication-Aware Placement

```
effective_score(i, j) = importance(i) − λ × comm_cost(j, local)

where comm_cost = 0.0 if same GPU, else 1.0
```

Entries are sorted by effective score; eviction targets lowest scores first — biasing removal toward remote, less-attended tokens.

---

## Baselines

| Method | Paper | Strategy |
|--------|-------|----------|
| Full Cache | Oracle | No eviction |
| H2O | Zhang et al., NeurIPS 2023 | Cumulative attention score + heavy hitters |
| SnapKV | Li et al., arXiv 2024 | Observation-window pooling during prefill |
| StreamingLLM | Xiao et al., ICLR 2024 | Attention sinks + sliding window |

---

## Configuration

Key config parameters in `configs/cpu_debug.yaml`:

```yaml
cache:
  budget_ratio: 0.2        # Keep 20% of max context length
  recent_window: 32         # Always keep last 32 tokens (H2O/AdaptKV)
  sink_tokens: 4            # StreamingLLM: number of sink tokens

policy:
  hidden_dim: 64            # MLP hidden layer size
  attention_window: 16      # EWMA window for features
  training_samples: 100     # Samples for policy distillation

distributed:
  simulate: true            # CPU simulation of multi-GPU
  num_simulated_gpus: 4     # How many virtual GPUs to simulate
```

---

## Citation

```bibtex
@misc{hassan2024adaptkv,
  title   = {AdaptKV: Adaptive KV Cache Compression for Distributed LLM Inference},
  author  = {Ali Hassan},
  year    = {2024},
  note    = {FAST-NUCES Islamabad, BS AI, PDC Course Project}
}
```

**Baseline papers:**

- Zhang et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference. *NeurIPS*.
- Li et al. (2024). SnapKV: LLM Knows What You are Looking for Before Generation. *arXiv*.
- Xiao et al. (2024). Efficient Streaming Language Models with Attention Sinks. *ICLR*.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.
