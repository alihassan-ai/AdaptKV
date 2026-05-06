#!/usr/bin/env bash
set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AdaptKV: Adaptive KV Cache Compression                     ║"
echo "║  Ali Hassan | 22i-0541 | FAST-NUCES Islamabad               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1: Install dependencies ─────────────────────────────────────────────
echo "[1/5] Installing dependencies..."
pip install -q --upgrade pip 2>/dev/null || true
pip install -q torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 2>/dev/null \
    || pip install -q torch torchvision torchaudio
pip install -q -r requirements.txt
echo "  ✓ Dependencies installed"

# ── Step 2: Detect GPUs ───────────────────────────────────────────────────────
echo ""
echo "[2/5] Detecting GPUs..."
GPU_INFO=$(python3 -c "
import torch
n = torch.cuda.device_count()
print(n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gb = p.total_memory // (1024**3)
    print(f'    GPU {i}: {p.name} ({gb} GB)')
" 2>/dev/null || echo "0")

GPU_COUNT=$(echo "$GPU_INFO" | head -1)
echo "  ✓ Found $GPU_COUNT GPU(s)"
echo "$GPU_INFO" | tail -n +2

if [ "${GPU_COUNT:-0}" -gt 0 ] 2>/dev/null; then
    DEVICE="cuda:0"
else
    echo "  ⚠ No CUDA GPUs — running in CPU simulation mode"
    DEVICE="cpu"
fi

mkdir -p results/charts

# ── Step 3: Run all experiments ───────────────────────────────────────────────
echo ""
echo "[3/5] Running proof experiments..."
python3 experiments/run_all_experiments.py --device "$DEVICE" --save_dir results
echo "  ✓ Experiments complete"

# ── Step 4: Generate report ───────────────────────────────────────────────────
echo ""
echo "[4/5] Generating report..."
python3 experiments/generate_report.py --results_dir results
echo "  ✓ Report generated → results/report.html"

# ── Step 5: Launch Gradio dashboard ──────────────────────────────────────────
echo ""
echo "[5/5] Launching Gradio dashboard..."
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Local:  http://localhost:7860                              ║"
echo "║  Public: see Gradio output below                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
python3 dashboard/app.py
