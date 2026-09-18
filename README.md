# Fine-tuning open-weight LLMs for clinical note understanding

Supervised fine-tuning of **Llama 3.1 8B**, **Mistral 7B**, and **Qwen2.5 7B** on synthetic
clinical notes, comparing **LoRA vs QLoRA** across eight clinical NLP tasks — with an honest
look at where the headline metric is misleading.

**Headline:** ROUGE-L improved **39% relative** (0.410 → 0.571) on Llama 3.1 8B, while average
answer length fell **48%** (111 → 57 words), roughly halving output tokens per request.
QLoRA matched LoRA quality (−0.5% relative) at **47% less GPU memory** (25.4 → 13.4 GB).

---

## Task

Given a clinical note and an instruction, produce the requested clinical output. Eight task
types: abbreviation expansion, coreference resolution, named entity recognition, paraphrasing,
question answering, relation extraction, summarization, and temporal information extraction.

**Dataset:** [Asclepius Synthetic Clinical Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes)
(~157k instruction–answer pairs over synthetic discharge summaries). Synthetic data avoids the
credentialing delay of MIMIC while keeping the shape of real clinical documentation work.

**Split:** grouped by patient, so no note appears in both train and test. A random split would
leak notes across the boundary — several questions share a single note — and inflate test scores.

---

## Results

### Quality (300 held-out examples, greedy decoding)

| Model | Method | ROUGE-L | ROUGE-1 | ROUGE-2 | Avg words |
|---|---|---|---|---|---|
| Llama 3.1 8B | baseline | 0.4101 | 0.5071 | 0.3498 | 111.4 |
| Llama 3.1 8B | **LoRA r16** | **0.5711** | **0.6646** | **0.5122** | 57.4 |
| Llama 3.1 8B | QLoRA r16 | 0.5682 | 0.6629 | 0.5097 | 59.8 |
| Mistral 7B v0.3 | baseline | 0.4262 | 0.5308 | 0.3617 | 93.9 |
| Mistral 7B v0.3 | LoRA r16 | 0.5514 | 0.6564 | 0.4926 | 55.0 |
| Qwen2.5 7B | baseline | 0.4306 | 0.5241 | 0.3573 | 61.3 |
| Qwen2.5 7B | LoRA r16 | 0.5544 | 0.6489 | 0.4883 | 57.5 |

### Per-task ROUGE-L

| Task | Llama base | Llama LoRA | Llama QLoRA | Mistral base | Mistral LoRA | Qwen base | Qwen LoRA |
|---|---|---|---|---|---|---|---|
| Abbreviation Expansion | 0.5042 | 0.6314 | 0.6188 | 0.4807 | 0.6214 | 0.4129 | 0.6205 |
| Coreference Resolution | 0.3181 | 0.5153 | 0.5150 | 0.3248 | 0.5214 | 0.3651 | 0.4835 |
| Named Entity Recognition | 0.3373 | 0.5321 | 0.5376 | 0.3639 | 0.5086 | 0.3747 | 0.5193 |
| Paraphrasing | 0.2663 | 0.4211 | 0.4606 | 0.3222 | 0.4458 | 0.3111 | 0.4352 |
| Question Answering | 0.4905 | 0.6492 | 0.6319 | 0.5162 | 0.6162 | 0.4833 | 0.6305 |
| Relation Extraction | 0.3791 | 0.5257 | 0.5117 | 0.3855 | 0.5280 | 0.4663 | 0.4961 |
| Summarization | 0.4376 | 0.6007 | 0.5922 | 0.4877 | 0.5667 | 0.4922 | 0.5882 |
| Temporal Info Extraction | 0.5466 | 0.6805 | 0.6682 | 0.5287 | 0.6019 | 0.5201 | 0.6564 |

Every task improved. The weakest baselines gained most — coreference +62%, NER +58%,
paraphrasing +58% relative — because those gaps were mostly output-format mismatches, not
missing capability.

### Training cost (60k examples, 1 epoch, single H100 80GB)

| Run | Method | Peak GPU mem | Wall clock | Eval loss | Token acc |
|---|---|---|---|---|---|
| Llama 3.1 8B | LoRA r16 | 25.44 GB | 2.59 h | 0.4237 | 0.8622 |
| Llama 3.1 8B | QLoRA r16 | **13.38 GB** | 2.47 h | 0.4259 | 0.8628 |
| Mistral 7B v0.3 | LoRA r16 | 25.14 GB | 2.89 h | 0.4280 | 0.8649 |
| Qwen2.5 7B | LoRA r16 | 24.84 GB | 2.12 h | 0.4224 | 0.8632 |

LoRA r16 on all linear layers trains ~0.5% of parameters. Wall-clock times were measured on a
**shared, contended server** and are indicative only — QLoRA appearing faster than LoRA is an
artifact of GPU contention, not a real speedup. Memory figures are unaffected by contention.

---

## What the numbers hide

Reading model outputs side by side surfaced four things the metrics table does not.

**1. Baseline ROUGE-L tracks verbosity, not capability.** The three untuned models scored in a
narrow band (0.410 / 0.426 / 0.431) while averaging 111, 94, and 61 words. Qwen led on ROUGE-L
largely because it was already close to the reference length. After tuning, all three converge to 55–57 words — so Llama's larger raw gain (+39% vs Qwen's +29%) partly reflects having more
verbosity to shed. A naive "which model improved most" reading gets this backwards.

**2. Fine-tuning taught the model to be less complete.** On an NER example, the baseline listed
five conditions and three procedures; the reference listed two and one. The tuned model learned
the reference's narrower scope and scored better for it. For a real clinical product, the
baseline's higher recall may be preferable. ROUGE rewarded the model for extracting less.

**3. Reference labels are imperfect.** Asclepius answers are machine-generated. One reference
states a treatment "was not specified" and then specifies it; both models gave the answer the
note actually supports and were penalized. Evaluating against references alone conflates model
error with label error.

**4. Task drift.** On a coreference example, the tuned model abandoned the requested task and
produced a summary instead — having learned the house style strongly enough to apply the wrong
task behavior. A known single-epoch multi-task SFT failure, and a natural source of DPO
preference pairs.

**The model never abstains.** The system prompt instructs it to say when a note lacks the
requested information, but no training example rewards that. It always produces a confident
answer — a real risk in clinical deployment, and worth measuring with a set of deliberately
unanswerable questions.

Training loss was a poor model-selection signal: eval loss spanned only 0.4224–0.4280 across
four runs while ROUGE-L differed meaningfully, and the loss ordering did not match the ROUGE
ordering.

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
# 1. Build grouped train/val/test splits
python prepare_data.py

# 2. Fine-tune (pick a free GPU explicitly on a shared box)
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --mode lora --n_train 60000 --bsz 8 --grad_accum 2 \
  --output runs/llama31_lora_r16

# 3. Evaluate before and after
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model meta-llama/Llama-3.1-8B-Instruct --tag llama31_baseline
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter runs/llama31_lora_r16/final --tag llama31_lora_r16

# 4. Collect every run into the tables above
python collect_results.py
```

`--mode qlora` loads the base model in 4-bit NF4 with double quantization. Swap `--model` for
`mistralai/Mistral-7B-Instruct-v0.3` or `Qwen/Qwen2.5-7B-Instruct`.

### Implementation notes

- **Completion-only loss** (`completion_only_loss=True`): gradient flows only through the
  assistant answer. Without it most of the signal goes into reproducing the note, which is the
  longer part of every example.
- **Chat templates** are applied per model via `apply_chat_template`, so the same code runs
  across Llama, Mistral, and Qwen despite different special-token conventions.
- **Left padding** at generation time, as required for batched decoder-only inference.
- **Fixed evaluation slice** (first 300 test rows) so every run is scored on identical inputs.

---

## Repository layout

```
prepare_data.py      # download, filter, patient-grouped split, chat formatting
train_sft.py         # LoRA/QLoRA SFT via TRL; writes run_stats.json (time, throughput, memory)
evaluate_model.py    # batched greedy generation, ROUGE overall + per task, per-example outputs
collect_results.py   # aggregate every run into the tables above
results/             # metrics JSON + per-example predictions for each run
```

Adapter weights and checkpoints are not committed (see `.gitignore`); `results/` holds the
metrics and generated outputs needed to reproduce every number above.

---

## Limitations

- Synthetic notes, not real EHR text. Real clinical documentation is messier: abbreviations,
  inconsistent structure, dictation errors.
- ROUGE measures lexical overlap with imperfect references, not clinical correctness. The
  faithfulness evaluation below is the fix.
- Single epoch, single seed, one hyperparameter setting per model. No confidence intervals;
  differences under ~2% relative should be treated as noise at n=300.
- Throughput measured under GPU contention.

## Next

- **Faithfulness evaluation** — entity grounding measured against the *note* rather than the
  reference, sidestepping the label-quality problem, plus an abstention set of unanswerable
  questions.
- **DPO** — preference pairs from task-drift failures like (4) above, to reduce off-task
  responses and unsupported detail.
- **Serving** — vLLM with p95 latency, throughput, and cost per 1k requests, before and after
  AWQ quantization.

## License and attribution

Code under MIT. Llama 3.1 is used under the Llama Community License; Mistral and Qwen under
Apache 2.0. Asclepius is released by its authors for research use.
