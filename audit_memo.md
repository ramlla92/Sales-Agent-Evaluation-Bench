# Audit Memo

## Question

Why is a generic benchmark such as `tau2-bench` not enough for this project, and what does Week 10 evidence show that Tenacious needs instead?

## Answer

`tau2-bench` is useful for broad agent reliability, but it is not enough for Tenacious because Tenacious does not primarily need a fluent general-purpose agent. It needs a sales agent that decides when *not* to send, grounds every claim in weak public evidence, preserves a constrained brand voice, and closes with the right discovery-call behavior. Those are not secondary preferences in this workflow. They are the product.

Week 10 evidence shows the gap clearly. In `trace_id=3703cd92-c3f8-404d-abf0-97398c0133b6` (`Airtouch`), the system sent outreach with `segment_confidence=0.41`, weak-signal honesty flags, and `signal_grounding_score=0.5`. A generic benchmark may still reward that output for coherence. Tenacious should penalize it because the system should probably have abstained. In `trace_id=932fe046-f65c-41e3-92a9-f85c297618e5` (`WISE`), the email framed the company as a growth/funding case despite contradictory evidence and a `factual_accuracy_score=0.25`. That is not just a weak answer; it is a brand and trust failure. In `trace_id=13664253-82cb-4831-8360-cf11a9e752f0` (`Pascal Biosciences`), the message sounds plausible but turns sparse evidence into a speculative "scaling research challenges" pitch. In `trace_id=9bef7093-6237-448f-b2d8-3f1917a4627c` (`nokyooz`), the language is mostly template filler and the CTA is soft. In `trace_id=5f6d37a8-3a7c-45d1-8b72-02a319dac8c9` (`Productboard`), even a better-grounded case still falls back to generic language like "explore potential synergies," which is exactly the kind of polished-but-low-signal output Tenacious wants to avoid.

The Week 10 probes anticipated these same gaps. `signal_001`, `signal_002`, `signal_003`, and `signal_004` all focus on evidence quality and over-claiming. `abstain_001`, `abstain_002`, and `abstain_003` focus on low-confidence gating and whether the system should decline to automate. `tone_001` and `tone_002` focus on tone drift and factuality. `schedule_001` and `schedule_002` show that CTA behavior is not generic conversation quality; it is workflow-specific business logic. `competitor_001` and `competitor_002` show that unsupported competitive framing is a critical failure, not a minor stylistic issue.

These observations map directly to the Tenacious taxonomy. `T-01` covers signal grounding, `T-02` covers hallucination and unsupported claims, `T-03` covers generic template language, `T-04` covers CTA weakness, `T-05` covers under-blocking on thin evidence, and `T-06` covers segment/context mismatch. A general benchmark may catch some bad outputs indirectly, but it does not bind scoring to these sales-domain failures in a mechanically checkable way.

That is why Tenacious-Bench must grade more than fluency or task completion. It must score whether the message cites a real supplied signal, whether the signal is interpreted conservatively, whether the segment framing is correct, whether banned generic phrases appear, whether the CTA is specific enough for discovery-call motion, and whether the system should have sent anything at all. In this workflow, a fluent wrong email is worse than no email.

## Decision

`tau2-bench` is not rejected; it is simply incomplete for this use case. Tenacious-Bench v0.1 should therefore prioritize machine-verifiable sales quality, abstention correctness, and evidence-grounded outreach behavior over broad conversational competence, because those are the failure modes Week 10 actually exposed.
