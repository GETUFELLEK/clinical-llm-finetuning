# Why I did not run DPO

Short version: I built the preference-pair pipeline, inspected the pairs against my own
judgement, found the automated reward ranked responses **wrongly more often than not**, and
stopped before training. Training on pairs I did not believe in would have produced a model
whose behaviour I could not defend.

## The setup

With no human preference labels and no judge model available, preference pairs came from
**rejection sampling with a scored reward**: sample K=4 responses per prompt from the SFT
model at temperature 0.9, score each, pair best against worst, drop pairs whose score gap
falls below a threshold.

The score combined two cheap signals:

- **Grounding** — the fraction of the response's content words that appear in the source
  clinical note. Intended to penalise invented clinical detail. Measured against the *note*,
  not the reference answer, deliberately: the Asclepius reference answers are themselves
  machine-generated and imperfect.
- **Reference agreement** — ROUGE-L against the dataset answer, intended to capture task
  adherence.

Two configurations were generated and inspected:

| Version | Grounding weight | Reference weight | Min margin | Pairs kept |
|---|---|---|---|---|
| v1 | 0.5 | 0.5 | 0.10 | — |
| v2 | 0.2 | 0.8 | 0.12 | 1293 |

## v1: the reward preferred a factually wrong answer

The clearest failure. Task: expand abbreviations in a discharge summary.

> **Chosen (higher reward):** "The abbreviation ASL in the discharge summary stands for
> **Ammonia Synthetase** …"
>
> **Rejected (lower reward):** "… ASL (**Argininosuccinate Lyase**), MRI (Magnetic Resonance
> Imaging), and ICU (Intensive Care Unit) …"

ASL is argininosuccinate lyase. The rejected response is correct; the chosen one is a
hallucination. It won because the grounding term rewards vocabulary that appears in the note,
and the *correct* expansion does not — abbreviation expansion is precisely the task whose
correct answer must come from outside the source text.

Two further v1 inversions:

- **Coreference resolution.** Rejected correctly identified `"the patient"` and `"she"` as
  referring to the 28-year-old patient. Chosen restated treatment facts and never performed
  the task — but echoed more note vocabulary.
- **Coreference resolution.** Chosen was a near-verbatim block copied from the note (maximal
  lexical overlap). Rejected explained the coreference in its own words, which is what was
  asked.

Roughly **3 of 5 inspected pairs were ranked backwards.** A lexical-overlap reward does not
measure answer quality; it measures how much of the note the response copies.

## v2: hallucination bias fixed, density bias remains

Dropping grounding to 0.2 removed the pathological case — no v2 pair was observed preferring
a factually wrong answer. But inspection of five pairs gave roughly **2 correct, 2 backwards,
1 wash**:

- **Correct:** preferred a focused entity list over one that enumerated every ventilator
  setting and then complained about missing family history.
- **Backwards (NER).** Chosen listed treatments and lab studies as entities "related to the
  diagnosis of FMF"; rejected correctly identified *FMF itself* as the diagnosis entity.
  Chosen won on entity count.
- **Backwards (coreference).** Rejected named both coreference instances with their
  containing sentences. Chosen conflated a coreference with a symptom description.

ROUGE-L against a reference still rewards breadth, so longer, denser responses win regardless
of whether they answer the question asked.

## Conclusion

**Cheap lexical rewards cannot rank clinical answers reliably.** The two failure modes are
structural rather than tuning artefacts:

1. **Grounding rewards copying.** It inverts on any task whose correct answer legitimately
   introduces vocabulary absent from the source — abbreviation expansion above all.
2. **Reference agreement rewards breadth.** Coverage and length correlate with ROUGE
   independently of correctness.

Doing this properly requires either human preference labels or a stronger model as judge.
Neither was available, so the honest outcome is a negative result rather than a trained model.

The pipeline itself (`generate_preference_pairs.py`, `train_dpo.py`) is complete and runnable;
what is missing is a reward worth optimising against.

## If continued

- **Task-aware reward:** disable grounding entirely for abbreviation expansion and
  paraphrasing, where new vocabulary is correct; retain it for NER, relation extraction, and
  summarisation, where the answer should stay in the note.
- **Length normalisation** to remove the density bias.
- **Validate the reward before using it:** hand-label ~50 pairs, measure the reward's
  agreement with those labels, and only proceed above some agreement threshold. That check —
  not the DPO run — is the step that actually determines whether the result means anything.
