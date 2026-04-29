# Audit Memo

## Question

What does a generic benchmark miss about Tenacious-style B2B sales work, and what does Week 10 evidence show about that gap?

## Starter position

Generic sales or agent benchmarks usually miss the combination of:

- public-signal grounding
- segment-specific fit
- Tenacious tone constraints
- appropriate discovery-call CTA behavior
- avoidance of unsupported business claims

That means a generic benchmark may reward fluent outreach that is still unusable in production because it sounds generic, overclaims, or ignores the specific hiring signal.

## Evidence to add from Week 10

Fill this with at least:

- `8` probe IDs from `probe_library.md`
- `5` trace IDs from `trace_log.jsonl`
- one example each of:
  - weak grounding
  - tone drift
  - wrong segment targeting
  - over-commitment / hallucinated value claims
  - missing calendar CTA

## Benchmark implication

Tenacious-Bench v0.1 should score:

- whether the message cites a real supplied signal
- whether the message fits the target segment
- whether the voice matches Tenacious constraints
- whether the message avoids banned phrasing
- whether it closes with an appropriate calendar-oriented CTA
- whether claims are supported by the provided context

## Expected failure modes

- generic personalization without evidence
- outreach that mentions the wrong trigger
- polished tone with no actual signal grounding
- inflated claims about ROI, speed, or customer fit
- enterprise-style messaging sent to SMB prospects

## Decision

The benchmark should prioritize mechanically checkable sales quality over broad conversational quality, because that is the gap most likely to matter in a Tenacious deployment.
