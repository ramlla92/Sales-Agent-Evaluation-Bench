# When Small Models Fail: Building Tenacious-Bench v0.1 and the Lessons of Negative Results

In the race to build agentic sales pipelines, we often assume that more data or better fine-tuning will inevitably lead to higher performance. This week, I set out to prove this assumption by building a custom benchmark for a Tenacious-style B2B sales agent and training a specialized judge component. 

What I found instead was a humbling reminder of the "LIMA" principle (*Less Is More for Alignment*) and the dangers of pattern-matching in synthetic datasets. This post documents the creation of **Tenacious-Bench v0.1** and the "honest failure" of our preference-tuned critic model.

---

## 1. The Gap: Why Generic Benchmarks Fail Tenacious

Existing B2B sales benchmarks typically focus on email fluency or basic intent extraction. However, for a high-stakes outreach engine like Tenacious, fluency is rarely the bottleneck. The real failure modes are more subtle:

- **Signal Grounding:** Does the agent actually use the hiring signal or competitor event it cited?
- **Action Correctness:** Should the agent `send`, `abstain`, or `review`? Generic models are often too "eager to please" and send emails when they should stay silent.
- **Tone Adherence:** Moving away from "marketing-speak" toward a grounded, respectful Tenacious voice.

During my Week 10 audit, I found that frontier models often scored high on "quality" but failed on "grounding." For instance, in trace `tb_manual_0027`, the model correctly identified a hiring signal but then wrote a job-post advertisement (*"Apply now!"*) instead of a sales email. Standard benchmarks wouldn't catch this speaker-perspective error, but for Tenacious, it's a brand-damaging failure.

## 2. The Audit Method: Finding the Failure Modes

To find these gaps, I performed a systematic audit of 60+ workflow traces. I categorized every failure into a taxonomy (T-01 to T-09), identifying that **grounding** and **tone** were the primary culprits. 

I used these real-world "trajectories of failure" as the seed for Tenacious-Bench. If a model couldn't tell the difference between a grounded B2B value prop and a generic template, it wasn't ready for the Tenacious bench.

## 3. The Dataset: Engineering Tenacious-Bench v0.1

Building a 200-task benchmark from a small seed corpus required a multi-modal authoring strategy:

### Multi-LLM Routing
I implemented a synthesis pipeline that routed across model families. I used **Claude 3.5 Sonnet** to author the hardest adversarial seeds (the "microscopic" detail) and cheaper models like **Qwen2.5-72B** to generate bulk variations. This "Magpie-style" approach allowed for high volume without sacrificing diagnostic depth.

### Judge-Filter Calibration
To ensure quality, every task passed through an LLM-as-a-judge filter. To avoid **preference leakage** (as documented by Li et al., 2025), I ensured the model family that generated the task never served as its judge. I calibrated the judge by hand-labeling 30 tasks and achieving an 80%+ inter-rater agreement before scaling.

### Contamination Protocol
Following Chen et al. (2025), I implemented a strict contamination check. No task entered the held-out partition if it had more than an 8-gram overlap or a cosine similarity >0.85 with the training set. This "sealing" process ensures the benchmark measures generalization, not memorization.

## 4. The Training Experiment: Path B (ORPO)

For the training phase, I chose **Path B: Preference Tuning a Judge/Critic**. I used the **ORPO (Odds Ratio Preference Optimization)** algorithm on a **Qwen2.5-1.5B** backbone.

### Why ORPO?
ORPO is a reference-free algorithm that frequently outperforms DPO at a lower compute cost. It directly penalizes the model for the log-odds of the rejected response, which I hoped would help the model "unlearn" the job-post patterns.

### The Training Data
I constructed 100 preference pairs from my Week 10 failures. The "rejected" responses were the real-world regressions (low-grounding, hypy tone), and the "chosen" responses were expert-corrected versions of those same traces.

## 5. The Honest Result: A Negative Finding

After three epochs of training, the results were definitive: **the trained model underperformed the baseline.**

| Metric | Score (1-5) |
| :--- | :--- |
| **Baseline (Prompt-only)** | **2.923** |
| Prompt-Adjusted Baseline | 2.878 |
| **Trained (path-b-orpo-full-v3)** | **2.614** |

### What Went Wrong?
The trained model suffered from what I call the **"Job Post Regression."** Instead of learning to be a better critic, it learned that certain keywords (like "hiring") should trigger specific templates. In task `tb_manual_0027`, the trained model explicitly reverted to a *"We're hiring... Apply now!"* output—the very behavior it was supposed to reject.

This was a "Failed Delta B": training actually performed worse than a well-crafted system prompt on the same backbone.

## 6. Reflections: What I Did Wrong and What's Next

### The Dataset Generation Flaw
In hindsight, my preference pairs were too homogeneous. By focusing heavily on "hiring signal" failures, I inadvertently reinforced the model's association between hiring evidence and candidate-facing templates. The dataset needed more "negative grounding" cases—examples where the hiring signal is present but the model *must* ignore it to focus on a different sales angle.

### Improving the Pipeline
1. **Diversity over Density:** Next time, I would reduce the programmatic combinatorial expansion and increase the "Hand-authored adversarial" slice from 15% to 30%.
2. **Harder Judge Filters:** My judge-filter calibration was too lenient on speaker perspective. A "grounded" job post still sounds "grounded," even if it's the wrong target audience.
3. **Multi-Model Ensembling:** A single 1.5B judge is likely too small for the nuanced reasoning required. A better approach would be a rejection-sampling layer using an ensemble of small specialized adapters.

### The Path Forward
Negative results are not failures; they are the boundaries of our current understanding. For Tenacious-Bench v0.2, I will focus on "contrastive grounding"—training models to specifically ignore the most tempting hallucinations.

Building the bench is just the beginning. The goal isn't a higher score; it's a more honest measurement of when we are ready to ship.

---
*Published as part of the TRP1 Challenge - Week 11.*
