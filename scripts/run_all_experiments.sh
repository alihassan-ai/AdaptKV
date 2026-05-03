#!/usr/bin/env bash
# ============================================================================
# AdaptKV — Master Experiment Script
# ============================================================================
# Usage:
#   bash scripts/run_all_experiments.sh [cpu|gpu]
#
# cpu: CPU/simulation mode on MacBook (uses OPT-125m, small context)
# gpu: Full GPU mode on RunPod 4x A40 (uses LLaMA-3-8B)
# ============================================================================

set -euo pipefail

MODE=${1:-cpu}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ "$MODE" = "cpu" ]; then
    CONFIG="configs/cpu_debug.yaml"
    echo "================================================================"
    echo "  AdaptKV — CPU Debug Mode (MacBook)"
    echo "  Config: $CONFIG"
    echo "================================================================"
else
    CONFIG="configs/runpod_4gpu.yaml"
    echo "================================================================"
    echo "  AdaptKV — GPU Mode (RunPod 4x A40)"
    echo "  Config: $CONFIG"
    echo "================================================================"
fi

mkdir -p results

# ── Step 1: Train the AdaptKV Policy ────────────────────────────────────────
echo ""
echo ">>> [1/6] Training AdaptKV Policy..."
python scripts/train_policy.py \
    --config "$CONFIG" \
    --output-dir results \
    --policy-save-path results/policy.pt

echo "    Policy training complete. Checkpoint: results/policy.pt"

# ── Step 2: Run Baseline Methods ────────────────────────────────────────────
METHODS=("full_cache" "h2o" "snapkv" "streaming")

for METHOD in "${METHODS[@]}"; do
    echo ""
    echo ">>> [2/6] Running baseline: $METHOD"
    python scripts/run_baseline.py \
        --config "$CONFIG" \
        --method "$METHOD" \
        --output-dir results \
        --max-samples 50 \
        --benchmarks needle ruler
    echo "    Baseline $METHOD complete."
done

# ── Step 3: Run AdaptKV ─────────────────────────────────────────────────────
echo ""
echo ">>> [3/6] Running AdaptKV (our method)..."
python scripts/run_adaptkv.py \
    --config "$CONFIG" \
    --policy-path results/policy.pt \
    --output-dir results \
    --max-samples 50 \
    --benchmarks needle ruler

echo "    AdaptKV evaluation complete."

# ── Step 4: LongBench (GPU only, too slow for CPU) ──────────────────────────
if [ "$MODE" = "gpu" ]; then
    echo ""
    echo ">>> [4/6] Running LongBench evaluations..."
    for METHOD in "${METHODS[@]}"; do
        python scripts/run_baseline.py \
            --config "$CONFIG" \
            --method "$METHOD" \
            --output-dir results \
            --max-samples 200 \
            --benchmarks longbench &
    done
    python scripts/run_adaptkv.py \
        --config "$CONFIG" \
        --policy-path results/policy.pt \
        --output-dir results \
        --max-samples 200 \
        --benchmarks longbench &
    wait
    echo "    LongBench evaluations complete."
else
    echo ""
    echo ">>> [4/6] Skipping full LongBench (CPU mode — use GPU for full eval)."
    python scripts/run_baseline.py \
        --config "$CONFIG" \
        --method full_cache \
        --output-dir results \
        --max-samples 5 \
        --benchmarks longbench
fi

# ── Step 5: Distributed Multi-GPU Evaluation ────────────────────────────────
echo ""
if [ "$MODE" = "gpu" ]; then
    echo ">>> [5/6] Running Distributed Evaluation (4x GPU)..."
    torchrun \
        --nproc_per_node=4 \
        --master_addr=localhost \
        --master_port=29500 \
        scripts/run_distributed.py \
        --config "$CONFIG" \
        --policy-path results/policy.pt \
        --output-dir results \
        --benchmark needle
    echo "    Distributed evaluation complete."
else
    echo ">>> [5/6] Running Distributed Simulation (CPU mode)..."
    python scripts/run_distributed.py \
        --config "$CONFIG" \
        --policy-path results/policy.pt \
        --output-dir results \
        --benchmark needle
    echo "    Distributed simulation complete."
fi

# ── Step 6: Collect and Display Results ─────────────────────────────────────
echo ""
echo ">>> [6/6] Collecting Results..."
python scripts/collect_results.py \
    --results-dir results \
    --output results/summary.json

echo ""
echo "================================================================"
echo "  All experiments complete!"
echo "  Results: results/"
echo "  Summary: results/summary.json"
echo "================================================================"
