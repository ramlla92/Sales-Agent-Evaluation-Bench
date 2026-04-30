# Synthesis Memo – Benchmarking LLMs Under Data Contamination

## Core Position

This paper argues that static benchmarks are unreliable due to data contamination and proposes **dynamic benchmarking** as a solution. While this is a reasonable direction, I do not adopt it in my design due to its practical limitations.

---

## Key Idea from the Paper

- Data contamination:
  - inflates model performance
  - breaks evaluation reliability

- Proposed solution:
  - shift from **static → dynamic benchmarks**
  - continuously update or regenerate test data

---

## Where I Agree

The paper correctly identifies:
- contamination as a serious issue
- limitations of static benchmarks

This aligns with my own observations.

---

## Key Disagreement: Practical Feasibility

### My Concern:
> Dynamic benchmarking is too expensive and complex to maintain.

### Costs include:
- **Computational cost**
  - continuous regeneration of datasets
- **Human cost**
  - validation and verification
  - ongoing updates

---

## My Argument

> The solution introduces more complexity than it removes.

Dynamic systems:
- require constant maintenance
- are harder to standardize
- may still suffer from contamination over time

---

## Why I Did Not Use It

In my design:
- I prioritize:
  - **controlled dataset construction**
  - **family-based splitting**
  - **contamination checks**

Instead of:
- continuously regenerating benchmarks

### Rationale:
> Prevention is more efficient than continuous correction.

---

## Alternative Approach

My approach focuses on:
- structured data generation
- contamination-aware splitting
- maintaining dataset integrity upfront

---

## Final Position

Dynamic benchmarking is:
- conceptually strong
- but **impractical at scale**

---

## Bottom Line

While the paper presents a valid direction, I do not adopt dynamic benchmarking due to its **high computational and human cost**, and instead focus on more efficient contamination prevention strategies.