"""DPO on top of the SFT model.

The merged SFT model is both the starting point and the reference policy: a fresh LoRA
adapter is trained, and TRL uses the same model with the adapter disabled as the frozen
reference, so no second copy of the weights is needed.

  python train_dpo.py --model merged/llama31_lora_r16 --output runs/llama31_dpo
"""
import argparse
import json
import math
import os
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged/llama31_lora_r16")
    ap.add_argument("--data", default="data/dpo_pairs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--beta", type=float, default=0.1, help="KL strength; lower drifts further")
    ap.add_argument("--lr", type=float, default=5e-6, help="much lower than SFT")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_prompt_len", type=int, default=1600)
    ap.add_argument("--report_to", default="tensorboard")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ds = load_from_disk(args.data).train_test_split(test_size=0.05, seed=42)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.pad_token_id = tok.pad_token_id

    peft_cfg = LoraConfig(
        r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )

    total = math.ceil(len(ds["train"]) / (args.bsz * args.grad_accum) * args.epochs)
    cfg = DPOConfig(
        output_dir=args.output,
        run_name=os.path.basename(args.output.rstrip("/")),
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.1 * total)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        report_to=args.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tok,
        peft_config=peft_cfg,   # reference = same model with adapter disabled
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(f"{args.output}/final")
    tok.save_pretrained(f"{args.output}/final")

    stats = {
        "model": args.model,
        "method": "dpo",
        "beta": args.beta,
        "n_pairs": len(ds["train"]),
        "train_runtime_sec": round(elapsed, 1),
        "final_train_loss": result.metrics.get("train_loss"),
        "final_eval": trainer.evaluate(),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    with open(f"{args.output}/run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
