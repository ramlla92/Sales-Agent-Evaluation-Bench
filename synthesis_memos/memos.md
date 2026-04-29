# Synthesis Memo – Data Contamination & Benchmarking in LLMs

## Core Position

Across these readings, a consistent theme emerges: **LLM evaluation is fundamentally unreliable due to data contamination**, and most proposed solutions (synthetic data scaling, datasheets, dynamic benchmarks) attempt to mitigate symptoms rather than address root causes.  

I agree with the problem framing, but I disagree with several key design choices—especially the reliance on **scaling synthetic data** and **evaluation-only fixes**.

---

## My Experimental Setup (Grounding My Argument)

In my own pipeline, I constructed a dataset using multiple generation strategies:

- 60 samples from **trace-based generation**
- 60 samples from **programmatic generation**
- 50 samples from **LLM-generated data**
- 30 samples from **manual curation**

Additionally:
- I plan to incorporate **datasheets** into my dataset design
- I use **family-based splitting** to reduce contamination risk

This setup gives me direct evidence about how different data sources behave.

---

## Key Disagreement #1: Scaling Synthetic Data

Many papers (especially synthetic data work) implicitly assume:

> “More data → better performance”

I disagree.

### My Evidence:
From my own dataset:
- LLM-generated samples often:
  - repeat patterns
  - introduce subtle errors
  - lack true diversity
- Programmatic and trace-based data are:
  - more structured
  - more reliable

### My Argument:
> Scaling already low-quality or biased synthetic data does not improve models—it **amplifies errors and bias**.

This aligns with the contamination paper’s concern that:
- synthetic or regenerated data can **overlap semantically**
- contamination is not just exact duplication, but also **structural similarity**

### Conclusion:
> Synthetic data should be **constrained and verified**, not scaled blindly.

---

## Key Disagreement #2: Dynamic Benchmarking as a “Solution”

The contamination survey suggests:
> Moving from static → dynamic benchmarks improves evaluation reliability

I partially disagree.

### My Reasoning:
- Dynamic benchmarks:
  - delay contamination
  - but do not prevent it
- Any published benchmark will eventually:
  - be scraped
  - enter training data

### My Insight:
> Dynamic benchmarking creates a **moving target**, not a true fix.

In my own work:
- Even programmatically generated data can:
  - become predictable
  - introduce distributional leakage

---

## Key Disagreement #3: Evaluation-Focused Fixes vs Data-Centric Fixes

Most papers focus on:
- better benchmarks
- better evaluation metrics

But the real issue is:
> **Uncontrolled training data**

### My Argument:
> Improving evaluation without controlling training data is fundamentally insufficient.

This is supported by:
- inability to trace training data
- increasing overlap between web data and benchmarks

---

## Where I Agree: Datasheets for Datasets

The datasheets paper is one of the few approaches I strongly agree with.

### Why:
- It forces:
  - explicit documentation
  - reflection on data sources
  - awareness of bias and limitations

### My Plan:
I will incorporate datasheets into my design to:
- document generation sources (trace, programmatic, LLM, manual)
- clarify intended use
- highlight risks (e.g., synthetic bias, contamination)

### Limitation (acknowledged):
- Datasheets do not *solve* contamination
- But they improve **transparency and accountability**

---

## My Design Choice: Family-Based Splitting

To mitigate contamination, I use:
> **family-based splitting**

### Rationale:
- Prevents:
  - near-duplicate leakage
  - syntactic contamination (as defined in the survey)
- Enforces:
  - stronger separation between train/test

### Why this matters:
The paper highlights:
- contamination includes **paraphrases and transformations**

Family-based splitting directly addresses this by:
- grouping related samples
- preventing cross-split leakage

---

## Final Position

Across all readings, my stance is:

> The field is overestimating the effectiveness of **scaling and evaluation fixes**, while underestimating the importance of **data quality, structure, and governance**.

### My Key Principles:
1. **Quality > Quantity**
2. **Structure > Scale**
3. **Transparency (datasheets) > blind usage**
4. **Separation (family splitting) > post-hoc fixes**

---

## Bottom Line

- Synthetic data scaling → **risky and often harmful**
- Dynamic benchmarks → **partial, not fundamental solution**
- Datasheets → **useful but insufficient alone**
- My approach → **hybrid generation + structured splitting + documentation**

> True progress requires controlling how data is created and used—not just how models are evaluated.