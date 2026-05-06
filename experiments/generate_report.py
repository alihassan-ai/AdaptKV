#!/usr/bin/env python3
"""
Generate text + HTML report from experiment JSON results.

Usage (from project root):
    python experiments/generate_report.py [--results_dir results]
"""

import argparse, json, os
from datetime import datetime


def load(name, results_dir):
    path = os.path.join(results_dir, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def badge(proven):
    if proven is True:  return "✓ PROVEN"
    if proven is False: return "✗ NOT PROVEN"
    return "? UNKNOWN"


def html_badge(proven):
    if proven is True:
        return '<span style="color:#22c55e;font-weight:bold">✓ PROVEN</span>'
    if proven is False:
        return '<span style="color:#ef4444;font-weight:bold">✗ NOT PROVEN</span>'
    return '<span style="color:#94a3b8">? UNKNOWN</span>'


def generate_text(rd):
    r1,r2,r3,r4,r5,r6 = [load(f"exp{i}_results", rd) for i in range(1,7)]
    lines = [
        "="*70,
        "  AdaptKV — Experiment Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
    ]

    # Exp 1
    lines += ["\n── Exp 1: Three-Tier vs Binary ─────────────────────────────────────",
              f"  Claim:  {r1.get('claim','N/A')}",
              f"  Result: {badge(r1.get('proven'))}"]
    results = r1.get("results", {})
    if results:
        lines.append("  Ratio │ H2O      │ AdaptKV  │ Δ")
        lines.append("  " + "-"*38)
        for cr in sorted([int(k) for k in results.keys()]):
            d = results[str(cr)]
            lines.append(f"  {cr:>4}× │ {d['h2o_mean']*100:>5.1f}%   │ "
                         f"{d['adaptkv_mean']*100:>5.1f}%   │ "
                         f"{d['improvement_abs']*100:>+.1f}pp")

    # Exp 2
    lines += ["\n── Exp 2: Per-Head Adaptation ──────────────────────────────────────",
              f"  Claim:  {r2.get('claim','N/A')}",
              f"  Result: {badge(r2.get('proven'))}",
              f"  Uniform:  {r2.get('uniform_mean',0)*100:.2f}%",
              f"  Per-head: {r2.get('perhead_mean',0)*100:.2f}%",
              f"  Δ = {r2.get('improvement_pp',0):+.2f} pp"]

    # Exp 3
    lines += ["\n── Exp 3: Communication Reduction ──────────────────────────────────",
              f"  Claim:  {r3.get('claim','N/A')}",
              f"  Result: {badge(r3.get('proven'))}",
              f"  Naive:      {r3.get('naive_mean_mb',0):.4f} MB/step",
              f"  Comm-aware: {r3.get('comm_aware_mean_mb',0):.4f} MB/step",
              f"  Reduction:  {r3.get('reduction_pct',0):.1f}%"]

    # Exp 4
    lines += ["\n── Exp 4: Async Prefetch ───────────────────────────────────────────",
              f"  Claim:  {r4.get('claim','N/A')}",
              f"  Result: {badge(r4.get('proven'))}",
              f"  Mean overlap: {r4.get('overall_overlap_pct',0):.1f}%"]
    meas = r4.get("measurements", [])
    if meas:
        lines.append("  Size (MB) │ Sync (ms) │ Async (ms) │ Overlap %")
        lines.append("  " + "-"*46)
        for m in meas:
            lines.append(f"  {m['size_mb']:>7.1f}   │ {m['sync_ms']:>8.2f}  │ "
                         f"{m['async_ms']:>9.2f}  │ {m['overlap_pct']:>6.1f}%")

    # Exp 5
    lines += ["\n── Exp 5: Scaling Efficiency ───────────────────────────────────────",
              f"  Claim:  {r5.get('claim','N/A')}",
              f"  Result: {badge(r5.get('proven'))}",
              f"  H2O reduction:     {r5.get('memory_reduction_pct',{}).get('h2o',0):.1f}%",
              f"  AdaptKV reduction: {r5.get('memory_reduction_pct',{}).get('adaptkv',0):.1f}%"]

    # Exp 6
    lines += ["\n── Exp 6: Memory Analysis ──────────────────────────────────────────",
              f"  Claim:  {r6.get('claim','N/A')}",
              f"  Result: {badge(r6.get('proven'))}",
              f"  AdaptKV reduction: {r6.get('adaptkv_reduction_pct',0):.1f}%",
              f"  Avg cosine sim:    {r6.get('avg_cosine_similarity',0):.4f}"]

    # Overall
    all_r = [r1, r2, r3, r4, r5, r6]
    n_proven = sum(1 for r in all_r if r.get("proven") is True)
    lines += ["\n" + "="*70,
              f"  OVERALL: {n_proven}/{len(all_r)} claims proven",
              "="*70]
    return "\n".join(lines)


def generate_html(rd):
    r1,r2,r3,r4,r5,r6 = [load(f"exp{i}_results", rd) for i in range(1,7)]
    experiments = [
        (1, r1, "Three-Tier vs Binary"),
        (2, r2, "Per-Head Adaptation"),
        (3, r3, "Communication Reduction"),
        (4, r4, "Async Prefetch"),
        (5, r5, "Scaling Efficiency"),
        (6, r6, "Memory Analysis"),
    ]
    rows = "".join(f"""
        <tr>
          <td style="padding:8px;border:1px solid #e2e8f0">{num}</td>
          <td style="padding:8px;border:1px solid #e2e8f0">{name}</td>
          <td style="padding:8px;border:1px solid #e2e8f0;font-size:13px">{r.get('claim','N/A')}</td>
          <td style="padding:8px;border:1px solid #e2e8f0;text-align:center">{html_badge(r.get('proven'))}</td>
        </tr>"""
        for num, r, name in experiments)

    n_proven = sum(1 for _, r, _ in experiments if r.get("proven") is True)
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>AdaptKV Report</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;max-width:960px;margin:40px auto;padding:0 20px}}
  h1{{color:#1e3a5f}} h2{{color:#2563eb}}
  table{{width:100%;border-collapse:collapse;margin:16px 0}}
  th{{background:#1e3a5f;color:white;padding:10px;text-align:left}}
  .stat{{background:#f8fafc;padding:12px;border-radius:8px;margin:8px 0}}
</style></head><body>
<h1>AdaptKV — Experiment Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="stat"><strong>Overall: {n_proven}/{len(experiments)} claims proven</strong></div>
<h2>Claims Summary</h2>
<table><tr><th>#</th><th>Experiment</th><th>Claim</th><th>Result</th></tr>{rows}</table>
<h2>Key Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">Attention retention improvement (10× compression)</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{
          f"{r1.get('results',{}).get('10',r1.get('results',{}).get(10,{})).get('improvement_abs',0)*100:+.1f} pp"
          if r1 else "N/A"}</td></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">Per-head improvement</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{f"{r2.get('improvement_pp',0):+.2f} pp" if r2 else "N/A"}</td></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">Communication reduction</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{f"{r3.get('reduction_pct',0):.1f}%" if r3 else "N/A"}</td></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">Async prefetch overlap</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{f"{r4.get('overall_overlap_pct',0):.1f}%" if r4 else "N/A"}</td></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">AdaptKV memory reduction</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{f"{r6.get('adaptkv_reduction_pct',0):.1f}%" if r6 else "N/A"}</td></tr>
  <tr><td style="padding:8px;border:1px solid #e2e8f0">INT4 avg cosine similarity</td>
      <td style="padding:8px;border:1px solid #e2e8f0">{f"{r6.get('avg_cosine_similarity',0):.4f}" if r6 else "N/A"}</td></tr>
</table></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    text = generate_text(args.results_dir)
    print(text)

    txt_path  = f"{args.results_dir}/report.txt"
    html_path = f"{args.results_dir}/report.html"
    with open(txt_path,  "w") as f: f.write(text)
    with open(html_path, "w") as f: f.write(generate_html(args.results_dir))
    print(f"\n  Text report → {txt_path}")
    print(f"  HTML report → {html_path}")


if __name__ == "__main__":
    main()
