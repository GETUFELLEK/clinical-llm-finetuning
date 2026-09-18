"""Prepare Asclepius synthetic clinical notes for SFT.

Output: data/train, data/val, data/test (HF datasets saved to disk), each row:
  prompt:     [system, user]  chat messages (note + instruction)
  completion: [assistant]     chat message (reference answer)
  task:       task category (for per-task evaluation)

Split is grouped by note/patient so the same clinical note never appears
in both train and test (otherwise test scores are inflated by leakage).
"""
import argparse
import random

from datasets import load_dataset

SYSTEM = (
    "You are a clinical documentation assistant. Answer the task using only "
    "information contained in the provided clinical note. If the note does not "
    "contain the information, say so."
)


def to_chat(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Clinical note:\n{ex['note']}\n\nTask: {ex['question']}"},
        ],
        "completion": [{"role": "assistant", "content": ex["answer"]}],
        "task": ex.get("task", "unknown"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="starmpcc/Asclepius-Synthetic-Clinical-Notes")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max_chars", type=int, default=10000, help="drop very long note+answer pairs (~2.5k tokens)")
    ap.add_argument("--test_groups", type=int, default=1000, help="number of held-out notes for test")
    ap.add_argument("--val_groups", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")
    print("Columns:", ds.column_names, "| rows:", len(ds))
    for col in ("note", "question", "answer"):
        assert col in ds.column_names, f"Expected column '{col}' not found; inspect ds.column_names"

    ds = ds.filter(lambda ex: len(ex["note"]) + len(ex["answer"]) <= args.max_chars, num_proc=8)
    print("After length filter:", len(ds))

    group_col = "patient_id" if "patient_id" in ds.column_names else "note"
    groups = sorted(set(ds[group_col]), key=str)
    random.Random(args.seed).shuffle(groups)
    test_g = set(groups[: args.test_groups])
    val_g = set(groups[args.test_groups : args.test_groups + args.val_groups])

    split = {
        "test": ds.filter(lambda ex: ex[group_col] in test_g, num_proc=8),
        "val": ds.filter(lambda ex: ex[group_col] in val_g, num_proc=8),
        "train": ds.filter(lambda ex: ex[group_col] not in test_g and ex[group_col] not in val_g, num_proc=8),
    }

    for name, d in split.items():
        d = d.map(to_chat, remove_columns=d.column_names, num_proc=8).shuffle(seed=args.seed)
        d.save_to_disk(f"{args.out}/{name}")
        print(f"{name}: {len(d)} rows")

    ex = split["test"][0]
    print("\n--- Example ---\nTASK:", ex.get("task"), "\nQ:", ex["question"], "\nA:", ex["answer"][:500])


if __name__ == "__main__":
    main()
