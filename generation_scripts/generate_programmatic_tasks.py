import argparse
import json
from itertools import product
from pathlib import Path


COMPANIES = [
    {
        "company_name": "Aster Cloud",
        "segment": "SMB SaaS",
        "company_context": "SMB SaaS company expanding GTM capacity while trying to keep rep ramp predictable.",
    },
    {
        "company_name": "Granite Freight",
        "segment": "mid-market logistics",
        "company_context": "Mid-market logistics team balancing pipeline coverage with process cleanup.",
    },
    {
        "company_name": "Northwind Health",
        "segment": "mid-market healthcare",
        "company_context": "Healthcare operator growing cautiously and sensitive to overclaiming.",
    },
    {
        "company_name": "Lattice Works",
        "segment": "enterprise manufacturing",
        "company_context": "Manufacturing team with slower sales cycles and operational handoff complexity.",
    },
    {
        "company_name": "Crestline Capital",
        "segment": "financial services",
        "company_context": "Financial services firm where credibility and evidence quality matter more than aggressive outreach.",
    },
]


SCENARIOS = [
    {
        "name": "send_hiring_spike_grounded",
        "signal": "The company opened 4 SDR roles and 2 AE roles in the last month.",
        "evidence": [
            "4 SDR roles opened in the last month",
            "2 AE roles opened in the last month",
            "GTM hiring expansion is directly observable",
        ],
        "segment_confidence": 0.78,
        "ai_maturity": 2,
        "action": "send",
        "failure_mode_tags": ["T-01", "T-04"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["4 SDR roles opened in the last month", "calendar"],
        "must_avoid": ["synergies", "best-in-class", "10x your pipeline", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "send_market_entry_specific",
        "signal": "The company appears to be entering a new geographic market while expanding commercial hiring.",
        "evidence": [
            "Commercial hiring increased alongside market expansion",
            "The company is entering a new geographic market",
            "Execution capacity likely matters in the near term",
        ],
        "segment_confidence": 0.74,
        "ai_maturity": 2,
        "action": "send",
        "failure_mode_tags": ["T-01", "T-03"],
        "banned_phrases": ["synergies", "revolutionary platform", "scaling challenges"],
        "tone_markers": ["concise", "grounded", "specific", "low-hype"],
        "must_include": ["new geographic market", "calendar"],
        "must_avoid": ["synergies", "revolutionary platform", "scaling challenges", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "send_bench_fit_clear",
        "signal": "The company is hiring across SDR and AE roles while keeping stack and GTM workflow conventional.",
        "evidence": [
            "Parallel SDR and AE hiring is visible",
            "The GTM workflow looks conventional and execution-heavy",
            "The opportunity is specific enough for direct outreach",
        ],
        "segment_confidence": 0.72,
        "ai_maturity": 2,
        "action": "send",
        "failure_mode_tags": ["T-01", "T-04"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "specific"],
        "must_include": ["SDR and AE hiring", "calendar"],
        "must_avoid": ["synergies", "best-in-class", "10x your pipeline", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "exploratory_ops_strain",
        "signal": "The company is hiring RevOps support alongside SDR hiring.",
        "evidence": [
            "RevOps hiring and SDR hiring are happening in parallel",
            "This can indicate ramp consistency pressure",
            "The signal is useful but not conclusive",
        ],
        "segment_confidence": 0.58,
        "ai_maturity": 2,
        "action": "exploratory_send",
        "failure_mode_tags": ["T-01", "T-02"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["RevOps hiring", "calendar"],
        "must_avoid": ["synergies", "best-in-class", "10x your pipeline", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "exploratory_sparse_signal",
        "signal": "The company shows moderate GTM motion, but the public evidence is not strong enough for a hard claim.",
        "evidence": [
            "A GTM motion is visible but sparse",
            "The evidence suggests pressure, not certainty",
            "A cautious outreach path is more appropriate than a direct pitch",
        ],
        "segment_confidence": 0.52,
        "ai_maturity": 1,
        "action": "exploratory_send",
        "failure_mode_tags": ["T-02", "T-05"],
        "banned_phrases": ["synergies", "revolutionary platform", "scaling challenges"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["cautious", "calendar"],
        "must_avoid": ["synergies", "revolutionary platform", "scaling challenges", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "exploratory_ai_hint",
        "signal": "There are AI-adjacent hiring hints, but no verified operational AI buildout.",
        "evidence": [
            "AI-adjacent hiring hints are present",
            "Operational AI investment is not yet verified",
            "The right move is exploratory, not assertive",
        ],
        "segment_confidence": 0.49,
        "ai_maturity": 1,
        "action": "exploratory_send",
        "failure_mode_tags": ["T-02", "T-06"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "specific"],
        "must_include": ["AI-adjacent hiring", "calendar"],
        "must_avoid": ["synergies", "best-in-class", "10x your pipeline", "unsupported competitor gap"],
        "cta_required": True,
    },
    {
        "name": "abstain_weak_signal",
        "signal": "Only weak AI-adjacent hints are present and job-post data is unavailable.",
        "evidence": [
            "No verified hiring spike is present",
            "Job-post data is unavailable",
            "Public AI hints are weak and not operational",
        ],
        "segment_confidence": 0.41,
        "ai_maturity": 1,
        "action": "abstain",
        "failure_mode_tags": ["T-02", "T-05"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["insufficient signal"],
        "must_avoid": ["calendar", "synergies", "best-in-class", "10x your pipeline"],
        "cta_required": False,
    },
    {
        "name": "abstain_context_mismatch",
        "signal": "The visible signal is too weak and the segment fit is uncertain.",
        "evidence": [
            "No clear GTM hiring momentum is verified",
            "The segment fit is uncertain",
            "Automated outreach would risk context mismatch",
        ],
        "segment_confidence": 0.39,
        "ai_maturity": 1,
        "action": "abstain",
        "failure_mode_tags": ["T-05", "T-06"],
        "banned_phrases": ["synergies", "revolutionary platform", "scaling challenges"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["insufficient signal"],
        "must_avoid": ["calendar", "synergies", "revolutionary platform", "scaling challenges"],
        "cta_required": False,
    },
    {
        "name": "abstain_missing_evidence",
        "signal": "A hiring claim would be speculative because the supporting public evidence is missing.",
        "evidence": [
            "Supporting evidence is missing",
            "The visible signals are too thin for automation",
            "The safer action is to abstain",
        ],
        "segment_confidence": 0.36,
        "ai_maturity": 0,
        "action": "abstain",
        "failure_mode_tags": ["T-01", "T-05"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["insufficient signal"],
        "must_avoid": ["calendar", "synergies", "best-in-class", "10x your pipeline"],
        "cta_required": False,
    },
    {
        "name": "review_bench_unknown",
        "signal": "The commercial signal is promising, but bench fit is still unknown.",
        "evidence": [
            "Commercial hiring suggests a live GTM need",
            "Bench fit is still unknown",
            "A human should verify fit before outreach is sent",
        ],
        "segment_confidence": 0.61,
        "ai_maturity": 2,
        "action": "review",
        "failure_mode_tags": ["T-05", "T-07"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "specific"],
        "must_include": ["manual review"],
        "must_avoid": ["calendar", "synergies", "best-in-class", "10x your pipeline"],
        "cta_required": False,
    },
    {
        "name": "review_competitor_gap_unknown",
        "signal": "The company may fit the motion, but any competitor-gap framing would be evidence-free.",
        "evidence": [
            "The top-level signal is plausible",
            "Competitor-gap evidence is missing",
            "A reviewer should validate the angle before outreach",
        ],
        "segment_confidence": 0.56,
        "ai_maturity": 2,
        "action": "review",
        "failure_mode_tags": ["T-02", "T-07"],
        "banned_phrases": ["synergies", "revolutionary platform", "scaling challenges"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["manual review"],
        "must_avoid": ["calendar", "synergies", "revolutionary platform", "scaling challenges"],
        "cta_required": False,
    },
    {
        "name": "review_signal_conflict",
        "signal": "There is some hiring motion, but the public evidence is internally conflicting.",
        "evidence": [
            "One signal suggests growth",
            "Another signal weakens confidence in that interpretation",
            "The safe path is human review before send",
        ],
        "segment_confidence": 0.47,
        "ai_maturity": 1,
        "action": "review",
        "failure_mode_tags": ["T-02", "T-05"],
        "banned_phrases": ["synergies", "best-in-class", "10x your pipeline"],
        "tone_markers": ["grounded", "respectful", "low-hype"],
        "must_include": ["manual review"],
        "must_avoid": ["calendar", "synergies", "best-in-class", "10x your pipeline"],
        "cta_required": False,
    },
]


def build_family_id(company: dict, signal: dict) -> str:
    segment = company["segment"].lower().replace(" ", "_")
    return f"fam_programmatic_{segment}_{signal['name']}_{signal['action']}"


def expected_output(company: dict, signal: dict) -> dict[str, str]:
    if signal["action"] == "abstain":
        return {
            "email_subject": "",
            "email_body": f"Insufficient signal to send outreach for {company['company_name']}. Route to manual review.",
        }
    if signal["action"] == "review":
        return {
            "email_subject": "",
            "email_body": f"Manual review required before outreach for {company['company_name']}. The signal is promising but not safe enough to automate yet.",
        }

    subject = f"Context: {company['company_name']} hiring signal"
    body = (
        f"Hi {company['company_name']} team, "
        f"I noticed {signal['signal'].lower()} "
        f"Teams in {company['segment']} often feel pressure when growth signals appear faster than internal coverage can keep up. "
        "Tenacious helps add flexible bench support without forcing a long-term hiring decision. "
        "If helpful, I can send a calendar link for a brief 15-minute conversation next week."
    )
    return {
        "email_subject": subject,
        "email_body": body,
    }


def build_task(index: int, company: dict, signal: dict) -> dict:
    action = signal["action"]
    return {
        "task_id": f"tb_prog_{index:04d}",
        "split": "unsplit",
        "source_mode": "programmatic",
        "difficulty": "hard" if action in {"abstain", "review"} else "medium",
        "task_type": "abstain_decision" if action in {"abstain", "review"} else "generate_outreach",
        "failure_mode_tags": signal["failure_mode_tags"],
        "input": {
            "company_name": company["company_name"],
            "company_context": company["company_context"],
            "hiring_signal": signal["signal"],
            "evidence": signal["evidence"],
            "segment": company["segment"],
            "segment_confidence": signal["segment_confidence"],
            "ai_maturity": signal["ai_maturity"],
            "guardrails": [
                "Use only the supplied evidence.",
                "Do not claim verified bench fit when unknown.",
                "Do not invent internal metrics.",
            ],
            "banned_phrases": signal["banned_phrases"],
            "disallowed_claims": [
                "unsupported competitor gap",
                "invented hiring momentum",
                "claiming verified bench fit when unknown",
            ],
            "tone_markers": signal["tone_markers"],
        },
        "expected_behavior": {
            "action": action,
            "must_include": signal["must_include"],
            "must_avoid": signal["must_avoid"],
            "cta_required": signal["cta_required"],
            "decision_rationale": f"Programmatic task generated from the {signal['name']} signal pattern.",
            "expected_output": expected_output(company, signal),
        },
        "rubric": {
            "signal_grounding": 0.25,
            "hallucination_control": 0.20,
            "tone_style": 0.15,
            "cta": 0.10,
            "decision_correctness": 0.20,
            "segment_fit": 0.05,
            "banned_phrase_control": 0.05,
        },
        "metadata": {
            "version": "0.3",
            "source_notes": "Programmatic task generated from a structured parameter sweep.",
            "family_id": build_family_id(company, signal),
            "authoring_revision": "programmatic_generation_v1",
            "tags": ["programmatic", company["segment"], signal["name"], action],
        },
    }


def summarize(tasks: list[dict]) -> dict:
    actions = {}
    families = {}
    for task in tasks:
        action = task["expected_behavior"]["action"]
        actions[action] = actions.get(action, 0) + 1
        family_id = task["metadata"]["family_id"]
        families[family_id] = families.get(family_id, 0) + 1
    return {
        "task_count": len(tasks),
        "actions": dict(sorted(actions.items())),
        "families": dict(sorted(families.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact programmatic Tenacious-Bench tasks as JSONL.")
    parser.add_argument("--output-jsonl", default="tenacious_bench_v0.1/programmatic_pool_unsplit.jsonl", help="Output JSONL file.")
    parser.add_argument("--summary-out", default="tenacious_bench_v0.1/programmatic_pool_unsplit_summary.json", help="Summary JSON file.")
    args = parser.parse_args()

    combos = list(product(COMPANIES, SCENARIOS))
    tasks = [build_task(i, company, signal) for i, (company, signal) in enumerate(combos, start=1)]
    summary = summarize(tasks)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
