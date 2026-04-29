# Tenacious-Bench v0.1

Tenacious-Bench v0.1 is a starter evaluation benchmark for a Tenacious-style B2B sales agent. It focuses on grounded outreach quality, segment fit, tone adherence, and simple machine-verifiable checks before any paid-model judging is added.

This repo is designed for the Week 11 assignment:

- Build a custom benchmark with `200-300` tasks.
- Keep `train/dev/held_out` partitions separate.
- Start with a rule-based evaluator.
- Add a small improvement component later, ideally a lightweight critic/judge.

## What is in this starter

- `schema.json`: JSON schema for benchmark tasks
- `scoring_evaluator.py`: runnable rule-based scorer
- `generation_scripts/`: starter generation and contamination scripts
- `tenacious_bench_v0.1/`: dataset partitions with example tasks
- `methodology.md`, `datasheet.md`, `audit_memo.md`: report starters
- `cost_log.csv`, `evidence_graph.json`: evidence and cost tracking starters

## Recommended default path

Default recommendation: `Path B` (judge/critic).

Why:

- easiest to defend even if Week 10 artifacts are thin
- production-relevant as a rejection or rollback layer
- works well with mostly rule-based evaluation plus preference data later
- cheaper and lower-risk than training a full generation adapter

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Python `3.11` is the target. The starter code uses only the standard library.

## Score one task

```bash
python scoring_evaluator.py tenacious_bench_v0.1/dev/task_programmatic_001.json examples/candidate_output_good.txt
```

You can also score the task's own reference answer:

```bash
python scoring_evaluator.py tenacious_bench_v0.1/dev/task_programmatic_001.json --use-ground-truth
```

## Generate more template tasks

```bash
python generation_scripts/generate_programmatic_tasks.py --count 12 --split train --output-dir tenacious_bench_v0.1/train
```

## Run contamination checks

```bash
python generation_scripts/contamination_check.py --train-dir tenacious_bench_v0.1/train --heldout-dir tenacious_bench_v0.1/held_out
```

## Suggested workflow

1. Inventory Week 10 artifacts and map them to failure modes.
2. Refine `schema.json` and the rubric dimensions.
3. Expand programmatic and trace-derived tasks.
4. Use the evaluator to filter low-quality synthetic tasks.
5. Convert the train split into preference data for a small judge.

## Current status

- schema: starter complete
- evaluator: runnable
- example tasks: included
- contamination script: included
- training artifacts: placeholder only

## Next

- add real Week 10 traces
- author `200-300` tasks
- complete inter-rater agreement
- prepare `Path B` preference pairs
- train a lightweight critic with LoRA or a small classifier
