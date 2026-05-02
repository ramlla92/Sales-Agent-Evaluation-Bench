# Executive Memo: Tenacious-Bench v0.1 and Path B Optimization

## Page 1: The Decision

### Executive Summary
We have completed the audit, construction, and evaluation of **Tenacious-Bench v0.1**, a 200-task benchmark specialized for Tenacious-style B2B outreach. Our evaluation of the **path-b-orpo-full-v3** judge component indicates that while the training pipeline is technically sound, the resulting model fails to outperform prompt-only baselines on critical grounding and tone dimensions. We recommend against deployment of this specific trained component and advise remaining on the prompt-engineered baseline while investigating data quality regressions.

### Headline Results (Tenacious-Bench v0.1 Held-out)
- **Baseline (Prompt-only):** 2.923
- **Prompt-Adjusted Baseline:** 2.878
- **Trained (path-b-orpo-full-v3):** 2.614
- **Delta A (Trained vs Baseline):** -0.309 (Statistically significant regression, p < 0.05)

### Delta B: Training vs. Prompt Engineering
The trained component underperformed the prompt-adjusted baseline by **-0.264**. This suggests that the ORPO preference tuning introduced pattern-matching behaviors (e.g., reverting to job-post templates) that were not present in the base model when guided by high-quality system prompts alone.

### Cost-Pareto Analysis
- **Cost per Task (Baseline):** $0.00425 (API cost)
- **Cost per Task (Trained):** $0.0 (Local serving) / $0.0001 (Estimated compute)
- **Latency:** Local inference reduces latency by ~40% compared to cloud API calls.
- **Pareto Finding:** While the trained component is significantly cheaper and faster, the ~10% drop in overall score (2.92 to 2.61) represents an unacceptable risk to Tenacious brand reputation and outreach quality.

### Production Recommendation: `do_not_deploy`
The trained Path B component should not be deployed in its current form. The system should remain on the cloud-hosted prompt-only baseline until the speaker-perspective and signal-grounding failures in the training data are resolved.

---

## Page 2: The Skeptic's Appendix

### 1. Uncaptured Failure Modes (for v0.2)
- **Multi-Turn Nuance:** v0.1 focus is on single-shot outreach; it does not grade the transition from initial "abstain" to secondary "exploratory_send" in a live thread.
- **Cross-Signal Synthesis:** Most tasks test 1-2 signals; real-world cases with 5+ conflicting signals are under-represented.
- **Time-Shift Decay:** The benchmark does not capture how a "good" message on Monday becomes "bad" by Friday due to signal expiration.
- **Competitor Landscape Depth:** Nuanced comparisons between Tenacious and specific named competitors are proxies only.

### 2. Public-Signal Lossiness
Our ground truth relies on public Crunchbase and LinkedIn signals. This is inherently lossy; a "perfect" benchmark score on grounding may still result in a sub-optimal real-world lead experience because the agent lacks access to the lead's internal internal priorities or non-public headcount shifts.

### 3. Unresolved Training Failure: "Job Post Regression"
A persistent failure mode in the trained adapter is the tendency to write "Apply now!" job advertisements instead of outbound sales emails when hiring signals are present. This regression suggests that the preference pairs used in training may have inadvertently reinforced the association between hiring evidence and candidate-facing (rather than buyer-facing) language.

### 4. Kill-Switch Trigger Condition
In the event that a trained critic *is* deployed in an experimental capacity, a kill-switch should be triggered immediately if the **moving average of signal_grounding scores drops below 1.5** over 50 production traces, or if the "job post" pattern is detected in >5% of generated outputs.
