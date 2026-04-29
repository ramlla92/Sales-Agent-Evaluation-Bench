# Methodology

## Objective

Build `Tenacious-Bench v0.1`, a custom benchmark for a Tenacious-style B2B sales agent, then train a small improvement component targeted at a specific failure mode.

## Path declaration

Recommended default path: `Path B - judge/critic`.

Initial rationale:

- Week 11 emphasizes benchmark quality more than heavy training.
- A critic is deployable as a guardrail even if generation stays unchanged.
- A critic can be trained from preference pairs derived from benchmark failures.
- This path remains useful even if Week 10 traces are limited.

Replace this section with citations to at least `3` Week 10 trace IDs and `2+` papers.

## Dataset construction plan

- `30%` trace-derived tasks
- `30%` programmatic tasks
- `25%` multi-LLM synthesis tasks
- `15%` hand-authored adversarial tasks

## Partitioning

- `50%` train
- `30%` dev
- `20%` held_out

Held-out tasks should remain sealed from training-time scripts and future public release should be delayed until after final measurement.

## Rubric dimensions

- signal grounding
- Tenacious tone/style
- banned phrases
- calendar CTA
- hallucination / unsupported claims
- segment fit
- overall score

## Inter-rater agreement protocol

- label `30` tasks
- relabel the same `30` tasks after `24` hours
- revise any rubric dimension below `0.80` agreement

Record the final matrix in `inter_rater_agreement.md`.

## Contamination protocol

- n-gram overlap check
- embedding similarity check placeholder
- time-shift verification for public-signal tasks

## Training plan

Phase 1 uses only rule-based scoring.  
Phase 2 adds a small learned critic using chosen/rejected pairs.

## Cost discipline

- avoid paid APIs in starter version
- log every future API or GPU charge in `cost_log.csv`

## Publication notes

- dataset target: Hugging Face
- likely license: `CC-BY-4.0`
- model artifact target: LoRA adapter or small critic
