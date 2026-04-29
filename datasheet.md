# Datasheet for Tenacious-Bench v0.1

## 1. Motivation

Tenacious-Bench v0.1 evaluates a B2B sales agent on domain-specific outreach quality that generic agent benchmarks do not score well, especially signal grounding, tone adherence, and safe claims.

## 2. Composition

Planned size: `200-300` tasks.

Each task includes:

- prospect context
- hiring signal brief
- bench summary
- optional prior thread
- instructions to agent
- ground truth
- rubric
- metadata

## 3. Collection process

The dataset is assembled from four sources:

- trace-derived tasks from Week 10 outputs
- template-driven programmatic tasks
- multi-LLM synthesis tasks
- hand-authored adversarial tasks

## 4. Preprocessing / cleaning

Starter assumptions:

- all tasks stored as individual JSON files
- IDs are unique
- contamination checks run before held-out sealing
- any private details must be redacted before publication

## 5. Recommended uses

- benchmark a Tenacious-style sales agent
- compare prompt-only versus trained interventions
- create preference data for a critic or judge

## 6. Out-of-scope / misuses

- not a general email quality benchmark
- not a replacement for legal or compliance review
- not suitable for scoring unsupported verticals without adaptation

## 7. Distribution

Planned distribution: Hugging Face dataset repository with documentation and quickstart examples.

## 8. Maintenance

Version `0.1` is expected to have blind spots. Future versions should expand:

- segment coverage
- longer multi-turn threads
- richer contamination controls
- stronger inter-rater analysis

## Layered detail

### Telescopic

Purpose: evaluate grounded, on-brand sales outreach.

### Periscopic

Task units are single sales-agent tasks with structured context and a scoreable response.

### Microscopic

Document exact generation routes, judge settings, and contamination thresholds once the full dataset is built.
