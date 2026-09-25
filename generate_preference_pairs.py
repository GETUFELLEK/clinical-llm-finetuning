import argparse
import json
import re

import torch
from datasets import Dataset, load_from_disk
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP = set("""the a an and or of to in for with on at by is are was were be been being this that these
those it its as from has have had not no if then than which who whom whose what when where how patient
patients summary discharge note clinical""".split())

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in STOP}


def grounding(response, note):
    words = content_words(response)
    if not words:
        return 0.0
    return len(words & content_words(note)) / len(words)


def score(response, note, reference, w_ground, w_ref):
    if not response.strip():
        return -1.0
    g = grounding(response, note)
    r = SCORER.score(reference, response)["rougeL"].fmeasure
    return w_ground * g + w_ref * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged/llama31_lora_r16")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--w_ground", type=float, default=0.5)
    ap.add_argument("--w_ref", type=float, default=0.5)
    ap.add_argument("--min_margin", type=float, default=0.10)
    ap.add_argument("--out", default="data/dpo_pairs")
    args = ap.parse_args()

    ds = load_from_disk(f"{args.data}/{args.split}")
    ds = ds.select(range(len(ds) - args.n, len(ds)))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    pairs = []
    stats = {"kept": 0, "dropped_margin": 0, "dropped_identical": 0}

    for i in tqdm(range(0, len(ds), args.bsz)):
        batch = ds[i : i + args.bsz]
        enc = tok.apply_chat_template(
            batch["prompt"], add_generation_prompt=True, return_tensors="pt",
            padding=True, return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95,
                num_return_sequences=args.k, pad_token_id=tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)

        for j in range(len(batch["prompt"])):
            cands = texts[j * args.k : (j + 1) * args.k]
            note = batch["prompt"][j][1]["content"]
            ref = batch["completion"][j][0]["content"]
            scored = sorted(
                ((score(c, note, ref, args.w_ground, args.w_ref), c) for c in cands),
                key=lambda x: x[0], reverse=True,
            )
            best_s, best = scored[0]
            worst_s, worst = scored[-1]
            if best.strip() == worst.strip():
                stats["dropped_identical"] += 1
                continue
            if best_s - worst_s < args.min_margin:
                stats["dropped_margin"] += 1
                continue
            pairs.append({
                "prompt": batch["prompt"][j],
                "chosen": [{"role": "assistant", "content": best.strip()}],
                "rejected": [{"role": "assistant", "content": worst.strip()}],
                "margin": round(best_s - worst_s, 4),
                "task": batch["task"][j],
            })
            stats["kept"] += 1

    Dataset.from_list(pairs).save_to_disk(args.out)
    print(json.dumps(stats, indent=2))
    print(f"Saved {len(pairs)} pairs to {args.out}")
    if pairs:
        p = pairs[0]
        print("\n--- example pair ---\nTASK:", p["task"], "| margin:", p["margin"])
        print("\nCHOSEN:\n", p["chosen"][0]["content"][:400])
        print("\nREJECTED:\n", p["rejected"][0]["content"][:400])


if __name__ == "__main__":
    main()
