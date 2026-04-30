# Synthesis Memo – Synthetic Data for Language Models

## Core Position

This paper argues that synthetic data is essential for scaling language models, especially as real-world data becomes limited, expensive, or restricted. While I agree that synthetic data is useful, I strongly disagree with the implicit design choice of **scaling synthetic data as a primary strategy**.

---

## Key Disagreement: Scaling Synthetic Data

The paper assumes:
> More synthetic data → better performance

I disagree with this assumption.

### My Argument:
Scaling synthetic data can introduce **compounding error**:
- Errors in generated data get repeated and amplified
- Biases become more entrenched
- Hallucinations propagate across training cycles

> Scaling bad data does not fix it—it **magnifies the problem**.

Over time, this can:
- drown out signal from high-quality data
- reduce model generalization
- create misleading improvements

---

## Evidence from My Design

In my dataset construction:
- I observed that LLM-generated data:
  - tends to repeat patterns
  - lacks true diversity
  - can include subtle inaccuracies

To address this, I explicitly avoid over-reliance on synthetic data.

---

## My Design Choice

To maintain realism and reduce error propagation:

- I use **30% real-world data**
- I include **contamination checks** to:
  - detect overlap
  - reduce leakage
  - prevent synthetic drift

### Rationale:
> Real data acts as an anchor to prevent synthetic data from drifting too far from reality.

---

## Alternative Approach

Instead of scaling synthetic data:
- prioritize **quality control**
- combine:
  - real data
  - structured generation (programmatic/trace-based)
- apply **verification mechanisms**

---

## Final Position

> Synthetic data is useful, but scaling it blindly is dangerous.

### My principle:
- **Quality > Quantity**
- **Controlled generation > uncontrolled scaling**

---

## Bottom Line

The paper overestimates the benefits of synthetic data scaling and underestimates the risks of **compounding error**. My design mitigates this by anchoring the dataset in real data and actively checking for contamination.