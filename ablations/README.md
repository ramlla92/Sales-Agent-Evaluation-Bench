# Ablations

This folder contains the main comparison artifacts used for the final memo and demo.

## Key Files

- `week10_baseline_eval_held_out.json`: held-out baseline evaluation
- `week10_prompt_adjusted_eval_held_out.json`: held-out prompt-adjusted baseline evaluation
- `path_b_orpo_full_v3_eval_dev.json`: corrected dev evaluation for the trained Path B adapter
- `path_b_orpo_full_v3_eval_held_out.json`: corrected held-out evaluation for the trained Path B adapter
- `ablation_results.json`: compact held-out comparison summary for reporting
- `held_out_traces.jsonl`: flattened held-out traces for quick inspection in demos

## Headline Comparison

Held-out average overall score on the 40-task split:

- `week10_baseline`: `2.923`
- `week10_prompt_adjusted`: `2.878`
- `path_b_orpo_full_v3`: `2.614`

Headline deltas:

- trained minus baseline: `-0.309`
- trained minus prompt-adjusted: `-0.264`

Interpretation:

- the trained component is a negative result on the corrected evaluator
- it underperforms both prompt-only baselines on held-out
- the dominant failures are weak signal grounding, poor Tenacious tone adherence, and speaker-perspective regression

## Demo Use

For the demo video:

1. open `ablation_results.json` to show the numeric held-out comparison
2. open `path_b_orpo_full_v3_eval_held_out.json` and jump to one low-scoring trace
3. open `held_out_traces.jsonl` if you want a more compact view of the same evidence
