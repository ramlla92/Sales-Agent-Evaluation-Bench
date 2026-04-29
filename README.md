# Tenacious-Bench v0.1

Tenacious-Bench v0.1 is a custom benchmark for evaluating a Tenacious-style B2B sales agent. It is built around the actual Week 10 workflow failures rather than generic email quality, with emphasis on:

- signal grounding
- unsupported claims
- generic/template language
- CTA quality
- action correctness across `send`, `exploratory_send`, `abstain`, and `review`
- segment and context fit

## Current Status

Target benchmark size:

- `200` total tasks

Built so far:

- `60` trace-derived tasks
- `60` programmatic tasks
- `120` total tasks currently authored

Still pending:

- multi-LLM synthesis slice
- hand-authored adversarial slice

## Repository Structure

Core benchmark logic:

- `schema.json`: compact task schema
- `scoring_evaluator.py`: rule-based evaluator for benchmark tasks

Generation pipeline:

- `generation_scripts/convert_trace_workflows.py`: converts Week 10 workflow outputs into compact trace-derived tasks
- `generation_scripts/assign_family_splits.py`: assigns family-based contamination-safe trace splits
- `generation_scripts/generate_programmatic_tasks.py`: generates the balanced programmatic task pool

Documentation:

- `methodology.md`
- `audit_memo.md`
- `datasheet.md`
- `docs/implementation_plan.md`
- `docs/progress.md`
- `cost_log.csv`

Dataset artifacts:

- `tenacious_bench_v0.1/trace_pool_unsplit.jsonl`
- `tenacious_bench_v0.1/trace_pool_unsplit_summary.json`
- `tenacious_bench_v0.1/programmatic_pool_unsplit.jsonl`
- `tenacious_bench_v0.1/programmatic_pool_unsplit_summary.json`
- `tenacious_bench_v0.1/splits_trace/train.jsonl`
- `tenacious_bench_v0.1/splits_trace/dev.jsonl`
- `tenacious_bench_v0.1/splits_trace/held_out.jsonl`
- `tenacious_bench_v0.1/splits_trace_summary.json`

## Benchmark Design

Chosen path:

- `Path B - judge/critic`

Why this path:

- the dominant Week 10 issue is not fluency
- the main failure is weak or unsafe send decisions under low-confidence evidence
- a critic is the most direct way to block low-confidence or poorly grounded outreach before send

This decision is documented in more detail in `methodology.md` and `audit_memo.md`.

## Dataset Construction

Planned final composition:

- `30%` trace-derived
- `30%` programmatic
- `25%` multi-LLM synthesis
- `15%` hand-authored adversarial

Implemented source modes:

### Trace-derived

- built from `60` Week 10 workflow outputs
- preserves observed behavior and corrected expected behavior
- current action mix: `33 abstain`, `27 exploratory_send`

### Programmatic

- built as `60` balanced tasks
- action mix:
  - `15 send`
  - `15 exploratory_send`
  - `15 abstain`
  - `15 review`

## Split and Contamination Strategy

The current trace-derived split uses family-based assignment rather than row-level random splitting.

Process:

1. generate source tasks into `unsplit` pools
2. assign `metadata.family_id`
3. split by family, not by row
4. run lexical and embedding-level contamination checks
5. seal held-out only after review

Current trace split:

- `train`: `30`
- `dev`: `18`
- `held_out`: `12`

Contamination controls:

- family-first split assignment
- n-gram overlap review
- sentence-embedding similarity review with cosine similarity `> 0.85`
- time-shift verification for public-signal cases

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Target Python version:

- `3.11`

## Common Commands

Generate the programmatic pool:

```bash
python generation_scripts/generate_programmatic_tasks.py
```

Convert Week 10 traces into the compact trace pool:

```bash
python generation_scripts/convert_trace_workflows.py
```

Assign family-based trace splits:

```bash
python generation_scripts/assign_family_splits.py
```

## What Comes Next

- build the adversarial/manual slice
- build the multi-LLM synthesis slice
- run contamination checks across all source modes together
- finalize global `train/dev/held_out` composition for the full `200`-task benchmark
- prepare preference pairs for critic training
