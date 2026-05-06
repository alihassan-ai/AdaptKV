#!/usr/bin/env python3
"""
Run all 6 AdaptKV proof experiments in sequence.

Usage (from project root):
    python experiments/run_all_experiments.py [--device cuda:0] [--save_dir results]
"""

import argparse, json, os, sys, time, traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run_experiment(name, fn, **kwargs):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    t0 = time.time()
    try:
        result  = fn(**kwargs)
        elapsed = time.time() - t0
        proven  = result.get("proven", False)
        print(f"  [{'PROVEN ✓' if proven else 'NOT PROVEN ✗'}]  ({elapsed:.1f}s)")
        return result, None
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--skip",     nargs="*", default=[], help="Experiment numbers to skip (1-6)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/charts", exist_ok=True)

    print("\n" + "="*60)
    print("  AdaptKV Proof Experiments")
    print(f"  Device:   {args.device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    from experiments.exp1_three_tier_vs_binary    import run as run1
    from experiments.exp2_per_head_adaptation     import run as run2
    from experiments.exp3_communication_reduction import run as run3
    from experiments.exp4_async_prefetch          import run as run4
    from experiments.exp5_scaling_efficiency      import run as run5
    from experiments.exp6_memory_analysis         import run as run6

    experiments = [
        ("1", "Three-Tier vs Binary",        run1),
        ("2", "Per-Head Adaptation",          run2),
        ("3", "Communication Reduction",      run3),
        ("4", "Async Prefetch",               run4),
        ("5", "Scaling Efficiency",           run5),
        ("6", "Memory Analysis",              run6),
    ]

    all_results = {}
    t_start = time.time()

    for num, name, fn in experiments:
        if num in args.skip:
            print(f"\n  [SKIPPED] {name}")
            continue
        result, error = run_experiment(name, fn,
                                       device=args.device,
                                       save_dir=args.save_dir)
        all_results[f"exp{num}"] = {"result": result, "error": error}

    total_time  = time.time() - t_start
    proven_count = sum(
        1 for v in all_results.values()
        if v.get("result") and v["result"].get("proven")
    )
    run_count = len(all_results)

    print("\n" + "="*60 + "\n  SUMMARY\n" + "="*60)
    for num, name, _ in experiments:
        if num in args.skip:
            continue
        key = f"exp{num}"
        res = all_results.get(key, {})
        if res.get("error"):
            status = f"ERROR: {res['error'][:55]}"
        elif res.get("result") and res["result"].get("proven"):
            status = "PROVEN ✓"
        else:
            status = "NOT PROVEN ✗"
        print(f"  {name:<40} {status}")

    print(f"\n  {proven_count}/{run_count} claims proven  |  {total_time:.1f}s total")
    print("="*60)

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "device":       args.device,
        "total_time_s": total_time,
        "proven_count": proven_count,
        "run_count":    run_count,
        "experiments":  {
            k: {
                "proven": v.get("result", {}).get("proven") if v.get("result") else False,
                "error":  v.get("error"),
                "claim":  v.get("result", {}).get("claim", "") if v.get("result") else "",
            }
            for k, v in all_results.items()
        },
    }
    summary_path = f"{args.save_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {summary_path}")
    return summary


if __name__ == "__main__":
    main()
