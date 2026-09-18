"""Supervised fine-tuning of Llama 3.1 8B Instruct (or Mistral) with LoRA / QLoRA.

Examples:
  # 5-minute smoke test
  python train_sft.py --n_train 400 --max_steps 20 --output runs/smoke --report_to none

  # Tonight's real run (bf16 LoRA)
  python train_sft.py --mode lora --n_train 30000 --output runs/llama31_lora_r16

  # Phase 2 comparison
  python train_sft.py --mode qlora --n_train 30000 --output runs/llama31_qlora_r16
  python train_sft.py --model mistralai/Mistral-7B-Instruct-v0.3 --output runs/mistral_lora_r16
"""
import argparse
import json
import math
import os
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--mode", choices=["lora", "qlora"], default="lora")
    ap.add_argument("--data", default="data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--report_to", default="wandb")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", "clinical-llm-sft")

    train = load_from_disk(f"{args.data}/train")
    train = train.select(range(min(args.n_train, len(train))))
    val = load_from_disk(f"{args.data}/val")
    val = val.select(range(min(args.n_val, len(val))))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        # Llama 3.1 ships a dedicated pad token; fall back to EOS for other models
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token

    quant = None
    if args.mode == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        quantization_config=quant,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tok.pad_token_id

    peft_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train) / (args.bsz * args.grad_accum) * args.epochs)
    warmup_steps = max(1, int(0.03 * total_steps))

    cfg = SFTConfig(
        output_dir=args.output,
        run_name=os.path.basename(args.output.rstrip("/")),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        completion_only_loss=True,  # loss only on the assistant answer, not the note
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=2,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    final_dir = f"{args.output}/final"
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    stats = {
        "model": args.model,
        "mode": args.mode,
        "n_train": len(train),
        "lora_r": args.r,
        "train_runtime_sec": round(elapsed, 1),
        "train_samples_per_sec": result.metrics.get("train_samples_per_second"),
        "final_train_loss": result.metrics.get("train_loss"),
        "final_eval": trainer.evaluate(),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    with open(f"{args.output}/run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
