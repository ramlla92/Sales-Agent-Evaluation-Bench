import argparse
import asyncio
import json
import os
import sys
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
from app.integrations.llm_client import LLMClient  # noqa: E402
from scoring_evaluator import evaluate  # noqa: E402

from ablations.run_week10_baseline_eval import (  # noqa: E402
    DEFAULT_RUBRIC,
    build_brief,
    slugify_company,
)


def short_evidence(evidence: list[str]) -> list[str]:
    cleaned = []
    for item in evidence[:3]:
        text = " ".join(item.split())
        cleaned.append(text[:140])
    return cleaned


def build_subject(task: dict, route: str) -> str:
    company = task["input"]["company_name"]
    evidence = task["input"].get("evidence", [])
    anchor = evidence[0] if evidence else task["input"].get("company_context", "current priorities")
    anchor = anchor.replace(company, "").strip(" -,:")
    anchor = " ".join(anchor.split())
    if len(anchor) > 32:
        anchor = anchor[:32].rsplit(" ", 1)[0]
    prefix = "Question:" if route == "exploratory" else "Context:"
    subject = f"{prefix} {company} {anchor}".strip()
    return subject[:58]


def offline_body(task: dict, route: str) -> str:
    company = task["input"]["company_name"]
    evidence = short_evidence(task["input"].get("evidence", []))
    first_signal = evidence[0] if evidence else task["input"].get("hiring_signal", "the public signal in your brief")
    context = task["input"].get("company_context", "")
    ai_maturity = int(task["input"].get("ai_maturity") or 0)
    signal_strength = (task["input"].get("signal_strength") or "").lower()

    if route == "review":
        return f"Insufficient signal to automate outreach safely for {company} from the current evidence. Route this case to manual review."

    opener = f"I noticed {first_signal.lower().rstrip('.')}" if first_signal else f"I noticed the recent signal around {company}"
    if route == "exploratory":
        if signal_strength == "weak" or ai_maturity <= 1:
            return (
                f"{opener}. Given the outside picture is still early, I wanted to ask one specific question rather than assume too much. "
                f"Is this tied to a broader team or product priority right now? "
                f"If useful, I can send over a calendar link and hold 15 minutes next week."
            )
        return (
            f"{opener}. That seemed specific enough to ask a short question rather than send a generic note. "
            f"Is this connected to a larger priority for the team right now? "
            f"If useful, I can send over a calendar link and hold 15 minutes next week."
        )

    context_tail = ""
    if context:
        context_tail = f" From the outside, that looks relevant to {context.lower().rstrip('.')}."
    return (
        f"{opener}.{context_tail} "
        f"I wanted to keep this grounded to the public signal rather than make a broad pitch. "
        f"If this is worth a conversation, I can send a calendar link and hold 15 minutes next week."
    )


def format_prior_thread(prior_thread: list[dict]) -> str:
    if not prior_thread:
        return "None"
    lines = []
    for item in prior_thread[:3]:
        speaker = item.get("speaker") or item.get("sender") or "unknown"
        message = item.get("message") or item.get("body") or ""
        lines.append(f"- {speaker}: {message}")
    return "\n".join(lines)


def prompt_for_task(task: dict, route: str) -> str:
    company = task["input"]["company_name"]
    context = task["input"].get("company_context", "")
    evidence_lines = "\n".join(f"- {item}" for item in short_evidence(task["input"].get("evidence", [])))
    banned = ", ".join(task["input"].get("banned_phrases", [])) or "None"
    guardrails = "; ".join(task["input"].get("guardrails", [])) or "Use only supplied evidence."
    prior_thread = format_prior_thread(task["input"].get("prior_thread", []))
    tone = ", ".join(task["input"].get("tone_markers", [])) or "grounded, respectful, low-hype"
    signal_strength = task["input"].get("signal_strength", "unknown")
    ai_maturity = task["input"].get("ai_maturity", "unknown")
    engagement_type = task["input"].get("engagement_type", "outreach")

    route_instruction = {
        "direct": "Write a grounded outreach email with one clear ask for a short call next week.",
        "exploratory": "Write a low-pressure exploratory email that asks one specific clarifying question and offers a short call next week.",
        "review": "Write a one- to two-sentence internal-style note explaining that outreach should pause for manual review.",
    }.get(route, "Write the safest correct response.")

    return f"""You are writing in the Tenacious outbound style.

Task:
{route_instruction}

Company: {company}
Context: {context}
Engagement type: {engagement_type}
Signal strength: {signal_strength}
AI maturity: {ai_maturity}
Prior thread:
{prior_thread}

Supplied evidence:
{evidence_lines if evidence_lines else "- None"}

Guardrails:
{guardrails}

Banned phrases:
{banned}

Style requirements:
- Direct, grounded, honest, professional, non-condescending.
- Use only supplied evidence.
- Cold outreach body max 120 words.
- One ask only.
- Do not use the word bench.
- Do not say Quick, Just, or Hey.
- If the signal is weak, ask rather than assert.
- Mention at least one concrete evidence item from the list above.
- Avoid hype, invented urgency, unsupported competitor claims, and generic filler.
- If writing outreach, end with a non-pushy ask for 15 minutes next week or a calendar-based equivalent.

Return only the email body text. No subject line. No bullets. No JSON.
"""


async def generate_candidate(task: dict, decision_service: DecisionService, llm: LLMClient | None, generator: str) -> dict:
    brief, icp, ai_maturity = build_brief(task, decision_service)
    gap = await decision_service.summarize_competitor_gap(brief, ai_maturity)
    route, reasons, _safe_gap = decision_service.evaluate_outreach_readiness(brief, icp, gap)
    budget = BudgetManager(limit_usd=1.5)

    if route == "review":
        body = "Insufficient signal to automate outreach safely from the current evidence. Route to manual review."
        subject = ""
        candidate_text = body
        return {
            "task_id": task["task_id"],
            "predicted_route": route,
            "subject": subject,
            "body": body,
            "candidate_text": candidate_text,
            "guardrail_reasons": reasons,
            "cost_usd": 0.0,
        }

    subject = build_subject(task, route)
    if generator == "offline":
        body = offline_body(task, route)
    else:
        prompt = prompt_for_task(task, route)
        response = await llm.complete_json(model=os.getenv("DEFAULT_MODEL") or os.getenv("CHEAP_MODEL") or "openrouter/qwen/qwen3-next-80b-a3b-thinking", prompt=prompt)
        budget.record(response.estimated_cost_usd)
        body = response.content.strip()
    candidate_text = f"{subject}\n\n{body}".strip()
    return {
        "task_id": task["task_id"],
        "predicted_route": route,
        "subject": subject,
        "body": body,
        "candidate_text": candidate_text,
        "guardrail_reasons": reasons,
        "cost_usd": budget.snapshot().spent_usd,
    }


async def run_eval(dataset_path: Path, output_path: Path, generator: str) -> dict:
    decision_service = DecisionService()
    llm = None if generator == "offline" else LLMClient()
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []

    for task in rows:
        candidate = await generate_candidate(task, decision_service, llm, generator)
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

    task_count = len(results)
    avg_overall = round(sum(item["overall_score"] for item in results) / task_count, 3) if task_count else 0.0
    total_cost = round(sum(item["cost_usd"] for item in results), 4)

    by_expected_action = {}
    for action in sorted({item["expected_action"] for item in results}):
        subset = [item for item in results if item["expected_action"] == action]
        by_expected_action[action] = {
            "count": len(subset),
            "avg_overall_score": round(sum(item["overall_score"] for item in subset) / len(subset), 3),
        }

    summary = {
        "dataset_path": str(dataset_path),
        "task_count": task_count,
        "avg_overall_score": avg_overall,
        "total_cost_usd": total_cost,
        "avg_cost_per_task_usd": round(total_cost / task_count, 4) if task_count else 0.0,
        "by_expected_action": by_expected_action,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    payload = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-only Week 10 comparator on a Tenacious benchmark split.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator", choices=["offline", "llm"], default="offline")
    args = parser.parse_args()
    payload = asyncio.run(run_eval(Path(args.dataset), Path(args.output), args.generator))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
