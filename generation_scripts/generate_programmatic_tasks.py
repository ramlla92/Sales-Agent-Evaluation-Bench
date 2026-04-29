import argparse
import json
from pathlib import Path


COMPANIES = [
    ("Northwind Health", "mid-market healthcare", "201-500", "healthcare"),
    ("Aster Cloud", "SMB SaaS", "51-200", "software"),
    ("Granite Freight", "mid-market logistics", "201-500", "logistics"),
    ("Lattice Works", "enterprise manufacturing", "1000+", "manufacturing"),
]

SIGNALS = [
    ("hiring_spike", "The company opened 6 new AE and SDR roles in the last 30 days."),
    ("funding", "The company announced a recent Series B and is expanding GTM hiring."),
    ("new_market", "The company appears to be entering a new geographic market."),
    ("ops_strain", "The team is hiring RevOps support alongside SDR hiring."),
]

PAIN_POINTS = [
    ["slow rep ramp", "bench utilization", "pipeline coverage"],
    ["fragmented outreach ops", "manual prospect research", "handoff delays"],
    ["ramp consistency", "hiring pressure", "signal triage"],
]

BANNED = [
    "best-in-class",
    "guarantee results",
    "10x your pipeline",
    "revolutionary platform"
]

TONE_MARKERS = ["concise", "grounded", "respectful", "specific", "low-hype"]


def make_task(index: int, split: str) -> dict:
    company_name, segment, company_size, industry = COMPANIES[index % len(COMPANIES)]
    signal_type, signal_summary = SIGNALS[index % len(SIGNALS)]
    pain_points = PAIN_POINTS[index % len(PAIN_POINTS)]
    signal_points = [
        signal_summary,
        f"{company_name} is in {segment}.",
        f"Pain points may include {pain_points[0]} and {pain_points[1]}.",
    ]

    return {
        "task_id": f"task_programmatic_{index:03d}",
        "split": split,
        "source_mode": "programmatic",
        "difficulty": "medium",
        "prospect_context": {
            "company_name": company_name,
            "segment": segment,
            "company_size": company_size,
            "industry": industry,
            "hq_region": "US",
            "tech_stack": ["Salesforce", "HubSpot"],
            "pain_points": pain_points
        },
        "hiring_signal_brief": {
            "signal_type": signal_type,
            "signal_summary": signal_summary,
            "signal_date": "2026-04-15",
            "allowed_grounding_points": signal_points,
            "disallowed_claims": [
                "we worked with your team already",
                "we know your exact conversion rate"
            ]
        },
        "bench_summary": {
            "core_value_props": [
                "help teams cover pipeline gaps without over-hiring",
                "support faster and steadier ramp"
            ],
            "proof_points": [
                "bench flexibility",
                "sales execution support"
            ],
            "banned_phrases": BANNED,
            "tone_markers": TONE_MARKERS,
            "calendar_cta_examples": [
                "Open to a 15-minute intro next week?",
                "Happy to send a calendar link if useful."
            ]
        },
        "prior_thread": [],
        "instructions_to_agent": "Write a short cold outbound email grounded in the supplied hiring signal. Keep it concrete, low-hype, and include a calendar-oriented CTA.",
        "ground_truth": {
            "reference_output": (
                f"Hi there — I noticed {signal_summary.lower()} "
                f"Teams in {segment} often feel pressure around {pain_points[0]} and {pain_points[1]} when hiring expands quickly. "
                "Tenacious helps add bench capacity without forcing a long-term headcount decision. "
                "If helpful, I can send a calendar link for a brief 15-minute conversation next week."
            ),
            "must_include": [
                signal_summary,
                pain_points[0],
                "calendar"
            ],
            "must_avoid": BANNED,
            "target_segment": segment
        },
        "rubric": {
            "signal_grounding": "Reference at least one allowed grounding point from the brief.",
            "tenacious_tone_style": "Tone should be concise, specific, respectful, and low-hype.",
            "banned_phrases": "Avoid all banned phrases.",
            "calendar_cta_presence": "Include a calendar-oriented CTA.",
            "hallucination_unsupported_claims": "Avoid unsupported claims about results or prior knowledge.",
            "segment_fit": "Message should fit the target segment and likely pains.",
            "overall_score": "Weighted average of the six dimensions."
        },
        "metadata": {
            "version": "0.1",
            "authoring_notes": "Programmatic starter task generated from a parameter sweep template.",
            "tags": ["programmatic", segment, signal_type]
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate programmatic Tenacious-Bench starter tasks.")
    parser.add_argument("--count", type=int, default=10, help="Number of tasks to generate.")
    parser.add_argument("--split", default="train", choices=["train", "dev", "held_out"], help="Target split.")
    parser.add_argument("--output-dir", required=True, help="Directory to write task JSON files to.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.count + 1):
        task = make_task(i, args.split)
        out_path = output_dir / f"{task['task_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)

    print(f"Generated {args.count} tasks in {output_dir}")


if __name__ == "__main__":
    main()
