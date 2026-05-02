---
library_name: adapter-transformers
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- orpo
- sales-evaluation
- negative-result
- tenacious-bench
license: mit
datasets:
- tenacious-bench-v0.1
---

# path-b-orpo-full-v3

This is a negative-result research artifact. On corrected held-out evaluation using Tenacious-Bench v0.1, this trained adapter underperformed both the baseline and the prompt-adjusted baseline.

## Model Details

- **Model Type:** LoRA Adapter for Qwen2.5-1.5B-Instruct
- **Language(s):** English
- **License:** MIT
- **Finetuning Technique:** ORPO (Odds Ratio Preference Optimization)
- **Primary Task:** B2B Sales Outreach Decision & Critique (Path B)

## Performance Comparison (Tenacious-Bench v0.1 Held-out)

| Evaluation Mode | Avg. Overall Score (1-5) | Delta vs Baseline |
|-----------------|--------------------------|-------------------|
| Baseline (Prompt-only) | 2.923 | - |
| Prompt-Adjusted | 2.878 | -0.045 |
| **path-b-orpo-full-v3** | **2.614** | **-0.309** |

### Production Recommendation: `do_not_deploy`

The trained component shows regressions in:
- **Signal Grounding:** Frequent hallucination of hiring evidence not present in the input.
- **Tone Adherence:** Reverting to generic marketing templates despite preference tuning.
- **Speaker Perspective:** Occasional confusion about whether it is representing Tenacious or the target company.

## Training Configuration

- **Epochs:** 3.0
- **Learning Rate:** 5e-06
- **Batch Size:** 1 (Effective batch size 4 with Gradient Accumulation)
- **Optimizer:** AdamW (as per default ORPO settings)
- **LR Scheduler:** Cosine
- **Beta (ORPO):** 0.1
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32

## Training Procedure

The model was trained on 100 preference pairs derived from the Tenacious-Bench v0.1 training set. Preference pairs were constructed using `(input, chosen_output, rejected_output)` triplets, where chosen outputs were expert-corrected or high-scoring synthetic examples.

### Hardware

- Training was performed on a single GPU (estimated T4 or similar based on logged runtimes).
- Total training runtime: ~11 minutes.

## Environmental Impact

- **GPU Type:** [Specify e.g., NVIDIA T4]
- **Hours:** 0.18
- **Provider:** [Specify e.g., Google Colab]

## Citation

Please refer to the Tenacious-Bench v0.1 repository for more information on the evaluation methodology.
