# AdaptKV: Communication-Aware Adaptive KV Cache Compression

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)

**AdaptKV** is a communication-aware adaptive Key-Value (KV) cache compression framework for distributed long-context Large Language Model (LLM) inference across multi-GPU systems. 

By introducing a **three-tier decision policy** and a **communication-aware placement strategy**, AdaptKV dramatically reduces the memory bottleneck of autoregressive generation, achieving **8.3% higher accuracy** on LongBench compared to state-of-the-art baselines like H₂O at 10× compression, and reducing inter-GPU data transfer by **47%**.

> **Note:** This is the official implementation for the paper *"AdaptKV: Communication-Aware Adaptive KV Cache Compression for Distributed Long-Context LLM Inference Across Multi-GPU Systems"* by Ali Hassan.

---

## 🌟 Key Innovations

1. **Three-Tier Adaptive Policy (Keep / Compress / Evict)**
   Unlike preceding binary (keep-or-evict) heuristics, AdaptKV leverages selective quantization. Moderately important tokens are quantized to 4-bit NormalFloat (NF4), preserving degraded but useful information rather than permanently losing it.
   
2. **Learned Per-Head Policy Network**
   A lightweight 2-layer MLP (∼0.5M parameters) trained via distillation evaluates KV entries on 5 features—attention momentum, variance, key redundancy, relative position, and clustered head-type—learning optimal, head-specific compression strategies natively.
   
3. **Communication-Aware Placement**
   In multi-GPU environments, cache blocks are distributed. AdaptKV co-optimizes eviction with data locality, heavily penalizing the retention of non-local data. This naturally reduces cross-GPU traffic by migrating or preferentially evicting remote entries.
   
4. **Asynchronous Prefetching Pipeline**
   Predictive prefetching of remote KV entries via non-blocking NCCL `irecv`/`isend` completely overlaps remote data transfers with current attention computations, hiding up to **73%** of communication latency.

---

## 📊 Performance & Evaluation

Evaluated on LLaMA-3-8B and Mistral-7B across up to 16× NVIDIA A40 GPUs.

### 1. Generation Quality (LongBench & Needle-in-a-Haystack)

At **10× compression ratio**, AdaptKV strongly outperforms binary-eviction baselines:

| Method | LongBench Avg (LLaMA-3-8B) | Needle @ 128k Ctx |
| ------ | :------------------------: | :---------------: |
| Full Cache (Upper Bound)| 74.2% | ~100% |
| StreamingLLM | 53.4% | - |
| H₂O | 65.9% | 41.2% |
| SnapKV | 67.4% | 47.8% |
| **AdaptKV (Ours)** | **71.4%** | **68.4%** |

*AdaptKV is capable of matching H₂O's 10× quality while compressing the cache by 21.3× (a 2.1× improvement in overall efficiency).*

### 2. Multi-GPU Distributed Systems Efficiency

*Results evaluating inter-GPU traffic on a 64K context step (4× A40).*

| Placement Strategy | Comm. Volume (MB/step) | Effective Latency | GPU Util. |
| ------------------ | :--------------------: | :---------------: | :-------: |
| Naive Uniform | 847.3 MB | 12.4 ms | 62.1% |
| **Comm-Aware + Prefetch**| **449.2 MB** | **1.9 ms** | **91.3%** |

AdaptKV dynamically pushes efficient scaling, achieving 79.6% throughput scaling efficiency even at 16× A40 GPUs simulating 262,144 maximum context tokens.

---

## 🚀 Quick Start

### Installation

Requires Python 3.10+ and PyTorch (2.3+).

```bash
git clone https://github.com/alihassan-ai/AdaptKV.git
cd adaptkv
pip install -r requirements.txt
pip install -e .
```

### Running the Evaluation Pipelines

AdaptKV includes full support for CPU simulation prototyping (for macOS) as well as heavy distributed multi-GPU runs using RunPod configuration.

**CPU Mode (MacBook / Prototyping)**
Includes OPT-125m distillation training, RULER, and simulation testing.
```bash
# Simulates a 4-GPU distributed cache manager environment on CPU
bash scripts/run_all_experiments.sh cpu
```

**Distributed Multi-GPU Deployment**
Execute the pipeline on 4× NVIDIA A40 Nodes:
```bash
# Example starting the 4-tier communication evaluation using torchrun
torchrun --nproc_per_node=4 scripts/run_distributed.py \
    --config configs/runpod_4gpu.yaml \
    --policy-path results/policy.pt
```

---

## 📂 Project Architecture

```text
adaptKV/
 ├── src/
 │   ├── cache/          # Implementations: BaseCache, AdaptKVCache, H2O, SnapKV
 │   ├── policy/         # 0.5M param Multi-Tier Decision MLP & Feature Extraction
 │   ├── distributed/    # Comm-aware placement, NCCL Async Prefetching manager
 │   └── evaluation/     # Needle-in-a-Haystack, LongBench, RULER benchmarking
 ├── configs/            # Configuration definitions (cpu_debug, runpod_8gpu, etc)
 ├── scripts/            # Training, ablation, and parallel execution pipelines
 └── results/            # Automatically stores trained MLP policies and raw metrics
```

---

## 📖 Citation

If you find AdaptKV helpful in your research, please consider citing:

```bibtex
@inproceedings{hassan2024adaptkv,
  title={AdaptKV: Communication-Aware Adaptive KV Cache Compression for Distributed Long-Context LLM Inference Across Multi-GPU Systems},
  author={Hassan, Ali},
  booktitle={Parallel and Distributed Computing (PDC)},
  year={2024},
  organization={FAST-NUCES Islamabad}
}
```
