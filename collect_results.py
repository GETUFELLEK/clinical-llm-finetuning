"""Aggregate every training run and evaluation into markdown tables for the README.

  python collect_results.py            # print to stdout
  python collect_results.py --md       # markdown table syntax, ready to paste
"""
import argparse
import glob
import json
import os


def load_runs():
    out = []
    for f in sorted(glob.glob("runs/*/run_stats.json")):
        s = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        if name.startswith("smoke"):
            continue
        out.append((name, s))
    return out


def load_evals():
    out = []
    for f in sorted(glob.glob("results/*_metrics.json")):
        m = json.load(open(f))
        out.append((m["tag"], m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="emit markdown tables")
    args = ap.parse_args()
    sep = " | " if args.md else "  "
    edge = "| " if args.md else ""
    end = " |" if args.md else ""

    evals = load_evals()
    print("\n## Quality\n")
    hdr = ["run", "ROUGE-L", "ROUGE-1", "ROUGE-2", "avg words"]
    print(edge + sep.join(hdr) + end)
    if args.md:
        print("|" + "|".join(["---"] * len(hdr)) + "|")
    for tag, m in evals:
        o = m["overall"]
        row = [tag, f"{o['rougeL']:.4f}", f"{o['rouge1']:.4f}", f"{o['rouge2']:.4f}", f"{m['avg_pred_words']:.1f}"]
        print(edge + sep.join(row) + end)

    runs = load_runs()
    print("\n## Training cost\n")
    hdr = ["run", "model", "method", "peak GB", "hours", "samples/s", "eval loss", "token acc"]
    print(edge + sep.join(hdr) + end)
    if args.md:
        print("|" + "|".join(["---"] * len(hdr)) + "|")
    for name, s in runs:
        row = [
            name,
            s["model"].split("/")[-1],
            s["mode"],
            f"{s['peak_gpu_mem_gb']:.2f}",
            f"{s['train_runtime_sec'] / 3600:.2f}",
            f"{s['train_samples_per_sec']:.2f}",
            f"{s['final_eval']['eval_loss']:.4f}",
            f"{s['final_eval']['eval_mean_token_accuracy']:.4f}",
        ]
        print(edge + sep.join(row) + end)

    if evals:
        print("\n## Per-task ROUGE-L\n")
        tags = [t for t, _ in evals]
        tasks = sorted(evals[0][1]["per_task"])
        print(edge + sep.join(["task"] + tags) + end)
        if args.md:
            print("|" + "|".join(["---"] * (len(tags) + 1)) + "|")
        for task in tasks:
            row = [task.strip()] + [f"{m['per_task'][task]['rougeL']:.4f}" for _, m in evals]
            print(edge + sep.join(row) + end)


if __name__ == "__main__":
    main()
