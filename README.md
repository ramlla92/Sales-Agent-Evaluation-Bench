# Tenacious-Bench v0.1

Tenacious-Bench v0.1 is a custom benchmark for evaluating a Tenacious-style B2B sales agent. It is built around the actual Week 10 workflow failures rather than generic email quality, with emphasis on:

- signal grounding
- unsupported claims
- generic/template language
- CTA quality
- action correctness across `send`, `exploratory_send`, `abstain`, and `review`
- segment and context fit

## Current Status

Final benchmark size:

- `200` total tasks

Final composition:

- `60` trace-derived tasks
- `60` programmatic tasks
- `50` multi-LLM synthesis tasks
- `30` manual tasks

Final split:

- `train`: `100`
- `dev`: `60`
- `held_out`: `40`

## Repository Structure

Core benchmark logic:

- `schema.json`: compact task schema
- `scoring_evaluator.py`: rule-based evaluator for benchmark tasks

Generation pipeline:

- `generation_scripts/convert_trace_workflows.py`: converts Week 10 workflow outputs into compact trace-derived tasks
- `generation_scripts/assign_family_splits.py`: assigns family-based contamination-safe trace splits
- `generation_scripts/generate_programmatic_tasks.py`: generates the balanced programmatic task pool

Documentation:

- `methodology.md`: Path B rationale and contamination protocol
- `audit_memo.md`: Audit of Week 10 failure modes
- `datasheet.md`: Dataset documentation (Gebru/Pushkarna standard)
- `dataset_card.md`: Hugging Face dataset card
- `model_card.md`: Hugging Face model card (negative results)
- `blog_post.md`: Technical retrospective and "honest failure" analysis
- `memo.md`: Two-page executive decision memo for stakeholders
- `evidence_graph.json`: Traceability for all numeric claims
- `cost_log.csv`: Detailed spend tracking

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

## Final Artifacts

Dataset:

- `tenacious_bench_v0.1/final_dataset/train/tenacious_bench_train_all_sources_100.jsonl`
- `tenacious_bench_v0.1/final_dataset/dev/tenacious_bench_dev_all_sources_60.jsonl`
- `tenacious_bench_v0.1/final_dataset/held_out/tenacious_bench_held_out_all_sources_40.jsonl`
- `tenacious_bench_v0.1/final_dataset/tenacious_bench_final_summary.json`

Preference tuning:

- `training_data/path_b_train_preferences.jsonl`
- `training_data/path_b_dev_preferences.jsonl`
- `training/run_path_b_orpo.py`
- `training/run_path_b_benchmark_eval.py`

Ablations:

- `ablations/week10_baseline_eval_held_out.json`
- `ablations/week10_prompt_adjusted_eval_held_out.json`
- `ablations/path_b_orpo_full_v3_eval_held_out.json`
- `ablations/ablation_results.json`

## Outcome

The final Path B ORPO run is a negative result on the corrected evaluator:

- baseline held-out: `2.923`
- prompt-adjusted held-out: `2.878`
- trained Path B held-out: `2.614`

Recommendation:

- **Do not deploy** the trained component in its current form.
- Stay on the prompt-engineered baseline while investigating dataset quality regressions.

## What I Did Wrong & Lessons Learned

- **Dataset Diversity Flaw:** The preference pairs were too heavily weighted toward "hiring signal" failures, causing the model to learn a rigid (and incorrect) "job post" template rather than nuanced sales grounding.
- **Judge Filter Lenience:** The automated judge-filter used during dataset authoring was too lenient on speaker-perspective shifts, allowing "good-sounding" but context-incorrect messages into the training set.
- **Backbone Constraints:** A 1.5B backbone may be too small to reliably unlearn strong pre-training priors like "hiring = job post" without a significantly larger and more diverse dataset.

## Future Work

- **Contrastive Grounding:** Revise v0.2 to include explicit "contrastive" examples where the model is rewarded for ignoring a signal that would lead to a template regression.
- **Process Reward Modeling (Path C):** Investigate if a stepwise process scorer can catch the "job post" regression earlier in the generation chain than a final-turn critic.
- **Ensemble Rejection Sampling:** Deploy a committee of small specialized judges to increase robustness against single-model preference bias.

## Next Steps

1. Finalize Hugging Face publication (Dataset and Model LoRA).
2. Distribute the Technical Blog Post to the community.
3. Submit the GitHub repository and Executive Memo for the final Act V Audit.
