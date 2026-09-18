"""Evaluate base model or fine-tuned adapter on the held-out test notes.

  python evaluate_model.py --tag baseline
  python evaluate_model.py --tag llama31_lora_r16 --adapter runs/llama31_lora_r16/final

Writes results/<tag>_outputs.jsonl (for reading side by side) and results/<tag>_metrics.json.
"""
import argparse
import json
import os
from collections import defaultdict

import evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    test = load_from_disk(f"{args.data}/test").select(range(args.n))  # same fixed subset every run

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation with decoder-only models

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    preds = []
    for i in tqdm(range(0, len(test), args.bsz)):
        batch = test[i : i + args.bsz]
        enc = tok.apply_chat_template(
            batch["prompt"], add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        preds.extend(tok.batch_decode(new, skip_special_tokens=True))

    refs = [c[0]["content"] for c in test["completion"]]
    rouge = evaluate.load("rouge")
    overall = rouge.compute(predictions=preds, references=refs)

    by_task = defaultdict(lambda: ([], []))
    for p, r, t in zip(preds, refs, test["task"]):
        by_task[t][0].append(p)
        by_task[t][1].append(r)
    per_task = {
        t: {"n": len(p), "rougeL": round(rouge.compute(predictions=p, references=r)["rougeL"], 4)}
        for t, (p, r) in sorted(by_task.items())
    }

    metrics = {
        "tag": args.tag,
        "adapter": args.adapter,
        "n": len(preds),
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "avg_pred_words": round(sum(len(p.split()) for p in preds) / len(preds), 1),
        "per_task": per_task,
    }
    with open(f"results/{args.tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"results/{args.tag}_outputs.jsonl", "w") as f:
        for ex, p, r in zip(test, preds, refs):
            f.write(json.dumps({"task": ex["task"], "question": ex["prompt"][1]["content"][-300:], "reference": r, "prediction": p}) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
