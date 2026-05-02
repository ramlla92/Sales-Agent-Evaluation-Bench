---
language:
- en
license: mit
size_categories:
- n<1K
task_categories:
- text-generation
- question-answering
tags:
- sales
- evaluation
- agentic
- benchmark
pretty_name: Tenacious-Bench v0.1
dataset_info:
  features:
  - name: task_id
    dtype: string
  - name: input
    struct:
    - name: lead_context
      dtype: string
    - name: hiring_signal
      dtype: string
    - name: company_profile
      dtype: string
  - name: expected_behavior
    struct:
    - name: action
      dtype: string
    - name: expected_output
      dtype: string
    - name: reasoning_anchor
      dtype: string
  - name: rubric
    struct:
    - name: grounding_rules
      sequence: string
    - name: tone_constraints
      sequence: string
  splits:
  - name: train
    num_bytes: 450000
    num_examples: 100
  - name: dev
    num_bytes: 270000
    num_examples: 60
  - name: held_out
    num_bytes: 180000
    num_examples: 40
  download_size: 900000
  dataset_size: 900000
---

# Tenacious-Bench v0.1

Tenacious-Bench v0.1 is a custom benchmark for evaluating a Tenacious-style B2B sales agent. It is built around actual workflow failures observed during development, focusing on high-stakes outreach decisions where grounding and tone are critical.

## Dataset Description

- **Repository:** [link-to-your-repo]
- **Leaderboard:** N/A
- **Point of Contact:** [your-name/org]

### Dataset Summary

A 200-task benchmark for evaluating grounded, low-hype Tenacious-style B2B outreach decisions and messages. Unlike generic email datasets, Tenacious-Bench prioritizes:

- **Signal Grounding:** Does the message actually use the provided evidence?
- **Action Correctness:** Choosing correctly between `send`, `exploratory_send`, `abstain`, and `review`.
- **Tone Adherence:** Avoiding "hypy" or generic template language.
- **Segment Fit:** Ensuring the CTA and value proposition match the lead's specific context.

### Supported Tasks and Leaderboards

The dataset is intended for evaluating LLMs acting as sales agents or critics. Evaluation is performed using the rule-based `scoring_evaluator.py` included in the repository.

## Dataset Structure

### Data Instances

Each instance consists of a lead context, hiring signals, and a company profile, paired with the expected action and a detailed rubric for scoring.

### Data Fields

- `task_id`: Unique identifier for the task.
- `input`: The context provided to the model.
- `expected_behavior`: The ground-truth action and message.
- `rubric`: Specific rules used by the evaluator.
- `metadata`: Additional context including `family_id` for contamination-safe splitting.

### Data Splits

The dataset is partitioned into three splits using a family-based strategy to prevent pattern leakage:

- **Train:** 100 tasks
- **Dev:** 60 tasks
- **Held-out:** 40 tasks

## Dataset Creation

### Curation Rationale

Generic LLM evaluations often miss the nuance of B2B sales (e.g., when *not* to send an email). This benchmark was created to bridge that gap.

### Source Data

- **Trace-derived:** 60 tasks converted from real Week 10 workflow traces.
- **Programmatic:** 60 tasks generated to provide balanced coverage across actions.
- **Multi-LLM Synthesis:** 50 tasks generated using an ensemble of models.
- **Manual:** 30 high-adversarial tasks hand-authored by domain experts.

## Considerations for Using the Data

### Discussion of Biases

The trace-derived portion of the dataset reflects the specific distribution of failures seen in the Tenacious project's early phases.

### Other Known Limitations

The benchmark is relatively small (200 tasks) and focused on specific B2B outreach patterns.

## Additional Information

### Licensing Information

MIT License

### Citation Information

Please cite the Tenacious-Bench repository if you use this dataset in your research.
