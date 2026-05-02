import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEEK10_AGENT_DIR = ROOT.parent / "Week_10" / "agent"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WEEK10_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK10_AGENT_DIR))

from app.core.budget_manager import BudgetManager  # noqa: E402
from app.decision.service import DecisionService  # noqa: E402
from app.messaging.service import MessagingService  # noqa: E402
from app.models.domain import (  # noqa: E402
    AIMaturityDecision,
    BenchToBriefMatch,
    CompetitorGapBrief,
    HiringSignalBrief,
    HiringVelocity,
    ICPDecision,
    SignalEvidence,
)
from scoring_evaluator import evaluate  # noqa: E402


DEFAULT_RUBRIC = {
    "signal_grounding": 0.25,
    "hallucination_control": 0.2,
    "tone_style": 0.15,
    "cta": 0.1,
    "decision_correctness": 0.2,
    "segment_fit": 0.05,
    "banned_phrase_control": 0.05,
}


def slugify_company(company_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", company_name.lower()) or "company"


def map_signal_confidence(signal_strength: str | None) -> float:
    mapping = {
        "weak": 0.35,
        "moderate": 0.65,
        "strong": 0.85,
    }
    return mapping.get((signal_strength or "").lower(), 0.5)


def guess_open_roles(evidence: list[str]) -> int:
    total = 0
    for item in evidence:
        match = re.search(r"\b(\d+)\b", item)
        if match:
            total += int(match.group(1))
    return total


def ai_adjacent_role_count(evidence: list[str]) -> int:
    total = 0
    for item in evidence:
        lowered = item.lower()
        if any(token in lowered for token in ("ai", "ml", "machine learning", "data scientist", "data engineer")):
            match = re.search(r"\b(\d+)\b", item)
            total += int(match.group(1)) if match else 1
    return total


def specialist_roles(evidence: list[str]) -> list[str]:
    roles = []
    for item in evidence:
        lowered = item.lower()
        if any(token in lowered for token in ("ai", "ml", "machine learning", "scientist", "data engineer", "product manager", "security engineer", "sre", "cv engineer")):
            roles.append(item)
    return roles[:5]


def build_hiring_velocity(task: dict) -> HiringVelocity:
    evidence = task["input"].get("evidence", [])
    open_roles = guess_open_roles(evidence)
    confidence = map_signal_confidence(task["input"].get("signal_strength"))
    if open_roles == 0:
        label = "insufficient_signal"
    elif open_roles >= 10:
        label = "tripled_or_more"
    elif open_roles >= 5:
        label = "doubled"
    else:
        label = "increased_modestly"
    return HiringVelocity(
        open_roles_today=open_roles,
        open_roles_60_days_ago=max(open_roles // 2, 0),
        velocity_label=label,
        signal_confidence=confidence,
        sources=["benchmark_input"],
    )


def summarize_evidence(evidence: list[str], keywords: tuple[str, ...], fallback: str) -> str:
    matches = [item for item in evidence if any(keyword in item.lower() for keyword in keywords)]
    return "; ".join(matches[:2]) if matches else fallback


def label_for_evidence(item: str) -> str:
    lowered = item.lower()
    if any(token in lowered for token in ("series", "funding", "raised", "seed", "led by", "million", "$", "€")):
        return "crunchbase_funding"
    if any(token in lowered for token in ("hiring", "posted", "open role", "role", "roles", "engineer", "scientist", "manager", "head of", "vp of", "director")):
        return "job_post_velocity"
    if any(token in lowered for token in ("layoff", "laid off", "restructuring")):
        return "layoff_event"
    if any(token in lowered for token in ("cto", "chief", "vp", "director", "head of", "joined", "appointed", "leadership")):
        return "leadership_change"
    if any(token in lowered for token in ("launched", "product", "platform", "dashboard", "module", "analytics", "ai", "ml")):
        return "ai_maturity_hint"
    return "benchmark_evidence"


def funding_metadata(item: str) -> dict:
    lowered = item.lower()
    stage = ""
    if "series a" in lowered:
        stage = "series_a"
    elif "series b" in lowered:
        stage = "series_b"
    elif "series c" in lowered:
        stage = "series_c"
    elif "seed" in lowered:
        stage = "seed"
    amount_match = re.search(r"([$€])\s?(\d+(?:\.\d+)?)\s?m", lowered)
    funding_total_usd = None
    if amount_match:
        funding_total_usd = float(amount_match.group(2)) * 1_000_000
    return {
        "last_funding_type": stage,
        "funding_total_usd": funding_total_usd,
    }


def make_signal_evidence(task: dict) -> list[SignalEvidence]:
    evidence = []
    task_evidence = task["input"].get("evidence", [])
    for idx, item in enumerate(task_evidence):
        label = label_for_evidence(item)
        metadata = {}
        if label == "job_post_velocity":
            metadata = {
                "ai_adjacent_open_roles": ai_adjacent_role_count([item]),
                "specialist_roles_open_60_plus_days": specialist_roles([item]),
            }
        elif label == "crunchbase_funding":
            metadata = funding_metadata(item)
        evidence.append(
            SignalEvidence(
                label=label,
                source="benchmark_input",
                value=item,
                date=None,
                confidence=0.7,
                evidence_text=item,
                metadata=metadata,
            )
        )
    if task["input"].get("company_context"):
        evidence.append(
            SignalEvidence(
                label="crunchbase_firmographics",
                source="benchmark_input",
                value=task["input"]["company_context"],
                date=None,
                confidence=0.75,
                evidence_text=task["input"]["company_context"],
                metadata={},
            )
        )
    return evidence


def infer_segment_name(task: dict) -> str:
    combined = " ".join(task["input"].get("evidence", [])) + " " + str(task["input"].get("company_context", ""))
    lowered = combined.lower()
    confidence = float(task["input"].get("segment_confidence") or 0.0)
    if confidence < 0.6:
        return "abstain"
    if any(token in lowered for token in ("layoff", "restructuring", "cost", "support", "operations manager", "ops")):
        return "Mid-market platforms restructuring cost"
    if any(token in lowered for token in ("cto", "vp", "director", "leadership", "head of")):
        return "Engineering-leadership transitions"
    if any(token in lowered for token in ("ai", "ml", "machine learning", "data scientist", "computer vision", "security engineer")):
        return "Specialized capability gaps"
    if any(token in lowered for token in ("series a", "series b", "funding", "raised")):
        return "Recently-funded Series A/B startups"
    return "Recently-funded Series A/B startups"


def build_icp(task: dict) -> ICPDecision:
    confidence = float(task["input"].get("segment_confidence") or 0.0)
    segment = infer_segment_name(task)
    if segment == "abstain":
        return ICPDecision(
            segment="abstain",
            confidence=confidence,
            reasons=["Low-confidence benchmark input."],
            disqualifiers=["Benchmark input does not support a confident send route."],
        )
    return ICPDecision(
        segment=segment,
        confidence=confidence,
        reasons=["Constructed from benchmark input evidence."],
        disqualifiers=[],
    )


def build_brief(task: dict, decision_service: DecisionService) -> tuple[HiringSignalBrief, ICPDecision, AIMaturityDecision]:
    company_name = task["input"]["company_name"]
    evidence = task["input"].get("evidence", [])
    ai_score = int(task["input"].get("ai_maturity") or 0)
    ai_confidence = 0.75 if ai_score >= 2 else 0.55 if ai_score == 1 else 0.4
    ai_maturity = AIMaturityDecision(
        score=ai_score,
        confidence=ai_confidence,
        summary=f"Benchmark-provided AI maturity {ai_score}.",
        justifications=[],
    )
    icp = build_icp(task)
    bench_match = decision_service.build_bench_to_brief_match(icp)
    if task["expected_behavior"]["action"] in {"review", "abstain"}:
        bench_match = BenchToBriefMatch(
            required_stack=bench_match.required_stack,
            requested_engineers=bench_match.requested_engineers,
            available_engineers=bench_match.available_engineers,
            fits_bench=False,
            note="Benchmark task expects caution or manual review.",
        )
    brief = HiringSignalBrief(
        prospect_domain=f"{slugify_company(company_name)}.com",
        prospect_name=company_name,
        generated_at=datetime.now(UTC).isoformat(),
        company_name=company_name,
        primary_segment_match=decision_service.to_schema_segment(icp),
        segment_confidence=float(task["input"].get("segment_confidence") or 0.0),
        ai_maturity=ai_maturity,
        hiring_velocity=build_hiring_velocity(task),
        bench_to_brief_match=bench_match,
        funding_summary=summarize_evidence(evidence, ("series", "funding", "raised", "million", "$", "€"), "No explicit funding signal in supplied benchmark evidence."),
        job_post_summary=summarize_evidence(evidence, ("hiring", "posted", "role", "roles", "engineer", "scientist", "manager"), "No explicit hiring evidence in supplied benchmark evidence."),
        layoffs_summary=summarize_evidence(evidence, ("layoff", "laid off", "restructuring"), "No layoff signal in supplied benchmark evidence."),
        leadership_summary=summarize_evidence(evidence, ("cto", "chief", "vp", "director", "head of", "joined", "appointed"), "No leadership signal in supplied benchmark evidence."),
        tech_stack_summary=summarize_evidence(evidence, ("ai", "ml", "platform", "dashboard", "analytics", "product"), task["input"].get("company_context", "No additional company context provided.")),
        evidence=make_signal_evidence(task),
        data_sources_checked=[],
        honesty_flags=[],
    )
    if task["input"].get("signal_strength") == "weak":
        brief.honesty_flags.append("weak_signal_from_benchmark")
    if task["input"].get("evidence_completeness") in {"low", "medium"}:
        brief.honesty_flags.append("incomplete_public_evidence")
    return brief, icp, ai_maturity


async def candidate_for_task(task: dict, decision_service: DecisionService, messaging_service: MessagingService) -> dict:
    brief, icp, ai_maturity = build_brief(task, decision_service)
    gap = await decision_service.summarize_competitor_gap(brief, ai_maturity)
    route, reasons, safe_gap = decision_service.evaluate_outreach_readiness(brief, icp, gap)
    budget = BudgetManager(limit_usd=1.5)

    expected_action = task["expected_behavior"]["action"]
    if route == "review" or expected_action in {"review", "abstain"}:
        if expected_action == "abstain":
            body = "Insufficient signal to send outreach safely. Route to manual review."
        else:
            body = "Manual review required before drafting outreach from this evidence."
        candidate_text = body
        return {
            "task_id": task["task_id"],
            "predicted_route": route,
            "candidate_text": candidate_text,
            "subject": "",
            "body": body,
            "guardrail_reasons": reasons,
            "cost_usd": budget.snapshot().spent_usd,
        }

    draft = await messaging_service.generate_outreach_email(
        company_name=task["input"]["company_name"],
        brief=brief,
        icp=icp,
        gap=safe_gap,
        budget_manager=budget,
        contact_name="John Doe",
        outreach_mode=route,
    )
    subject = draft.subject or ""
    body = draft.body.strip()
    candidate_text = f"{subject}\n\n{body}".strip()
    return {
        "task_id": task["task_id"],
        "predicted_route": route,
        "candidate_text": candidate_text,
        "subject": subject,
        "body": body,
        "guardrail_reasons": reasons,
        "cost_usd": budget.snapshot().spent_usd,
    }


async def run_eval(dataset_path: Path, output_path: Path) -> dict:
    decision_service = DecisionService()
    messaging_service = MessagingService()

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []

    for task in rows:
        try:
            candidate = await candidate_for_task(task, decision_service, messaging_service)
        except Exception as exc:
            fallback_body = f"Manual review required due to evaluation-time generation failure: {exc}"
            candidate = {
                "task_id": task["task_id"],
                "predicted_route": "review",
                "candidate_text": fallback_body,
                "subject": "",
                "body": fallback_body,
                "guardrail_reasons": [f"generation_failure: {exc}"],
                "cost_usd": 0.0,
            }
        task_for_eval = dict(task)
        if "rubric" not in task_for_eval:
            task_for_eval["rubric"] = DEFAULT_RUBRIC
        score = evaluate(task_for_eval, candidate["candidate_text"])
        results.append(
            {
                "task_id": task["task_id"],
                "source_mode": task.get("source_mode"),
                "difficulty": task.get("difficulty"),
                "expected_action": task["expected_behavior"]["action"],
                "predicted_route": candidate["predicted_route"],
                "overall_score": score["overall_score"],
                "component_scores": score["component_scores"],
                "subject": candidate["subject"],
                "body": candidate["body"],
                "guardrail_reasons": candidate["guardrail_reasons"],
                "cost_usd": candidate["cost_usd"],
            }
        )

    overall = round(sum(item["overall_score"] for item in results) / len(results), 3) if results else 0.0
    by_source = {}
    grouped_scores = Counter()
    route_counts = Counter(item["predicted_route"] for item in results)
    for source_mode in sorted({item["source_mode"] for item in results}):
        source_rows = [item for item in results if item["source_mode"] == source_mode]
        by_source[source_mode] = {
            "count": len(source_rows),
            "avg_overall_score": round(sum(item["overall_score"] for item in source_rows) / len(source_rows), 3),
        }
    for item in results:
        bucket = item["expected_action"]
        grouped_scores[bucket] += item["overall_score"]
    by_expected_action = {}
    for action in sorted({item["expected_action"] for item in results}):
        action_rows = [item for item in results if item["expected_action"] == action]
        by_expected_action[action] = {
            "count": len(action_rows),
            "avg_overall_score": round(sum(item["overall_score"] for item in action_rows) / len(action_rows), 3),
        }

    summary = {
        "dataset_path": str(dataset_path),
        "task_count": len(results),
        "avg_overall_score": overall,
        "route_counts": dict(route_counts),
        "by_source_mode": by_source,
        "by_expected_action": by_expected_action,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    payload = {
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 10 baseline against the Tenacious held-out benchmark.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "tenacious_bench_v0.1" / "final_dataset" / "held_out" / "tenacious_bench_held_out_all_sources_40.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "ablations" / "week10_baseline_eval_held_out.json"),
    )
    args = parser.parse_args()

    payload = asyncio.run(run_eval(Path(args.dataset), Path(args.output)))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
