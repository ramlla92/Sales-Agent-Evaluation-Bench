# Datasheet for Tenacious-Bench v0.1

## Overview

Tenacious-Bench v0.1 is a compact benchmark for evaluating a Tenacious-style B2B sales agent. It focuses on decisions and outputs that generic agent benchmarks tend to miss:

- signal grounding
- unsupported claims
- generic/template language
- CTA quality
- send vs abstain vs review correctness
- segment/context fit

The benchmark is designed around the Week 10 workflow failures, not as a general email-writing dataset.

## Current Build Status

The full benchmark target is `200` tasks.

Implemented so far:

- `60` trace-derived tasks in `trace_pool_unsplit.jsonl`
- `60` programmatic tasks in `programmatic_pool_unsplit.jsonl`

Planned but not yet built:

- multi-LLM synthesis slice
- hand-authored adversarial slice

So the current built pool is `120` tasks, with the remaining `80` still to be authored.

## Task Format

Tasks follow the compact schema in `schema.json`.

Each task contains:

- task identity and source metadata
- `input`
- `expected_behavior`
- `rubric`
- `metadata`

Important task fields:

- `expected_behavior.action`
- `expected_behavior.expected_output`
- optional `expected_behavior.observed_output` for trace-derived cases
- `failure_mode_tags` using `T-01` to `T-09`
- `metadata.family_id` for contamination-safe partitioning

## Data Sources

### 1. Trace-derived tasks

Trace-derived tasks are converted from the `60` Week 10 workflow outputs using `generation_scripts/convert_trace_workflows.py`.

Purpose:

- preserve real Tenacious-specific failure patterns
- keep observed behavior alongside corrected expected behavior
- turn actual workflow traces into compact benchmark tasks

Current trace-derived distribution:

- total: `60`
- `abstain`: `33`
- `exploratory_send`: `27`

Current trace failure-tag coverage:

- `T-01`: `36`
- `T-02`: `50`
- `T-03`: `20`
- `T-04`: `47`
- `T-05`: `26`
- `T-06`: `1`

### 2. Programmatic tasks

Programmatic tasks are generated with `generation_scripts/generate_programmatic_tasks.py`.

Purpose:

- create balanced coverage by design
- vary only a small number of factors per task
- create realistic but controlled cases

Generation rules used:

- realistic company and hiring-signal inputs
- only `1-2` primary factors varied per scenario
- each task tests `1-2` failure modes
- clear `expected_behavior`
- include both safe/good cases and failure-triggering cases
- avoid vague or duplicate tasks

Current programmatic distribution:

- total: `60`
- `send`: `15`
- `exploratory_send`: `15`
- `abstain`: `15`
- `review`: `15`

## Split Strategy

The trace-derived slice has already been partitioned into:

- `train`: `30`
- `dev`: `18`
- `held_out`: `12`

The split is not random by row. It is assigned by `family_id` using `generation_scripts/assign_family_splits.py`.

That means:

- related examples stay in the same partition
- near-duplicate behavioral patterns do not leak across splits
- the model is less able to exploit repeated task-family templates

Trace family construction currently uses a combination of:

- task type
- observed-to-expected action transition
- failure tags
- AI maturity bucket
- confidence bucket
- evidence-source label
- industry
- country
- honesty-flag bucket

## Contamination Controls

The current contamination protocol is:

1. generate source pools as `unsplit`
2. assign `family_id`
3. split by family, not by row
4. run lexical overlap checks
5. run embedding similarity checks
6. rewrite or remove flagged held-out items before sealing

Implemented thresholds and rules:

- n-gram overlap review between held-out and non-held-out tasks
- sentence-embedding similarity review with cosine similarity `> 0.85`
- time-shift verification for public-signal cases

The family-first rule is the main safeguard against pattern-learning contamination.

## Recommended Uses

- benchmark a Tenacious-style sales agent
- compare generation-only vs critic-gated behavior
- build preference pairs for a small critic or judge model
- evaluate whether the system sends, abstains, or routes to review appropriately

## Not Intended For

- general-purpose email quality scoring
- legal or compliance approval
- evaluation outside the Tenacious outreach setting without adaptation

## Current Limitations

- the benchmark is not complete yet; only `120/200` tasks are built
- trace-derived coverage is currently concentrated in `abstain` and `exploratory_send`
- the trace split is contamination-aware but still somewhat distributionally narrow
- inter-rater agreement artifacts are not yet written
- adversarial and multi-LLM slices are still pending

## Cost Log Status

The project tracks spend in `cost_log.csv`.

Current logged cost:

- `0.00 USD` recorded so far

Interpretation:

- the current trace-derived and programmatic dataset work is now logged as explicit process entries with zero direct cost
- no paid API or GPU usage has been recorded yet
- future synthesis, judging, or training costs should be appended there as they occur

## Repository Artifacts

Current main dataset artifacts:

- `tenacious_bench_v0.1/trace_pool_unsplit.jsonl`
- `tenacious_bench_v0.1/trace_pool_unsplit_summary.json`
- `tenacious_bench_v0.1/programmatic_pool_unsplit.jsonl`
- `tenacious_bench_v0.1/programmatic_pool_unsplit_summary.json`
- `tenacious_bench_v0.1/splits_trace/train.jsonl`
- `tenacious_bench_v0.1/splits_trace/dev.jsonl`
- `tenacious_bench_v0.1/splits_trace/held_out.jsonl`
- `tenacious_bench_v0.1/splits_trace_summary.json`

Supporting benchmark logic:

- `schema.json`
- `scoring_evaluator.py`
- `generation_scripts/convert_trace_workflows.py`
- `generation_scripts/assign_family_splits.py`
- `generation_scripts/generate_programmatic_tasks.py`
