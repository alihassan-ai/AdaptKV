#!/usr/bin/env python3
"""Collect all JSON result files and produce comparison tables.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir results --output results/summary.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing JSON result files.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to results/summary.json.")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_scores(result: dict) -> dict:
    """Pull the top-level benchmark averages from a result dict."""
    scores = {}
    for bench in ["needle", "longbench", "ruler"]:
        val = result.get(bench, {})
        if isinstance(val, dict):
            scores[bench] = val.get("__average__", None)
        else:
            scores[bench] = None
    return scores


def collect_method_results(results_dir: str) -> dict:
    """Walk results_dir for known method result files."""
    methods = ["full_cache", "h2o", "snapkv", "streaming", "adaptkv"]
    collected = {}

    for method in methods:
        method_results = {}

        # Look for files named: *_{method}.json or baseline_{method}_*.json
        patterns = [
            os.path.join(results_dir, f"*{method}*.json"),
            os.path.join(results_dir, f"baseline_{method}*.json"),
        ]
        matched = []
        for pat in patterns:
            matched.extend(glob.glob(pat))
        matched = list(set(matched))

        for path in matched:
            try:
                data = load_json(path)
                filename = os.path.basename(path)
                method_results[filename] = data
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

        if method_results:
            collected[method] = method_results

    return collected


def build_comparison_table(collected: dict) -> list:
    """Build a flat list of rows for easy display."""
    rows = []
    for method, files in collected.items():
        for filename, data in files.items():
            scores = extract_scores(data)
            row = {
                "method": method,
                "file": filename,
                "needle_avg": scores.get("needle"),
                "longbench_avg": scores.get("longbench"),
                "ruler_avg": scores.get("ruler"),
            }
            # Add cache stats if available
            cache_stats = data.get("cache_stats", {})
            row["cache_size_bytes"] = cache_stats.get("cache_size_bytes")
            row["budget"] = cache_stats.get("budget")
            rows.append(row)
    return rows


def print_table(rows: list) -> None:
    """Print a formatted comparison table to stdout."""
    if not rows:
        print("No results found.")
        return

    header = f"{'Method':<15} {'Needle':>8} {'LongBench':>10} {'RULER':>8} {'CacheSize':>12}"
    print("\n" + "=" * 60)
    print("AdaptKV Results Summary")
    print("=" * 60)
    print(header)
    print("-" * 60)

    for row in sorted(rows, key=lambda r: r["method"]):
        needle   = f"{row['needle_avg']:.3f}"   if row["needle_avg"]    is not None else "  N/A"
        longbench= f"{row['longbench_avg']:.3f}" if row["longbench_avg"] is not None else "  N/A"
        ruler    = f"{row['ruler_avg']:.3f}"    if row["ruler_avg"]     is not None else "  N/A"
        cs = f"{row['cache_size_bytes']//1024}K" if row["cache_size_bytes"] else "  N/A"

        print(f"{row['method']:<15} {needle:>8} {longbench:>10} {ruler:>8} {cs:>12}")

    print("=" * 60 + "\n")


def main():
    args = parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Results directory '{args.results_dir}' not found.")
        print("Run the experiments first with: bash scripts/run_all_experiments.sh")
        return

    print(f"Scanning results in: {args.results_dir}")
    collected = collect_method_results(args.results_dir)

    if not collected:
        print("No result files found. Have you run the experiments yet?")
        return

    rows = build_comparison_table(collected)
    print_table(rows)

    # Build summary JSON
    summary = {
        "collected_methods": list(collected.keys()),
        "comparison_table": rows,
        "raw_results": {
            method: {fname: data for fname, data in files.items()}
            for method, files in collected.items()
        },
    }

    out_path = args.output or os.path.join(args.results_dir, "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
