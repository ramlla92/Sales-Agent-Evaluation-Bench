import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llm_synthesis_core import append_cost_log, extract_cost, extract_usage

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


ACTION_OVERRIDES = {
    "tb_llm_0009": "review",
    "tb_llm_0010": "review",
    "tb_llm_0027": "review",
    "tb_llm_0028": "review",
    "tb_llm_0040": "review",
}

GENERIC_PHRASES = [
    "our solutions",
    "our platform",
    "our solution",
    "support your growth",
    "explore how we might support",
    "potential collaboration",
    "would love to connect",
    "measurable efficiency gains",
    "flexible pricing models",
]

STOPWORDS = {
    "a", "an", "and", "announced", "at", "by", "for", "from", "in", "into", "is",
    "it", "its", "last", "new", "of", "on", "or", "q1", "q2", "q3", "q4", "the",
    "to", "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize(text: str) -> str: return re.sub(r"\s+", " ", text).strip()


def short_company_name(name: str) -> str:
    suffixes = {"Inc.", "Inc", "Ltd.", "Ltd", "Solutions", "Analytics", "Technologies"}
    tokens = name.split()
    if len(tokens) > 1 and tokens[-1] in suffixes:
        return " ".join(tokens[:-1])
    return name


def possessive(name: str) -> str: return f"{name}'" if name.endswith("s") else f"{name}'s"


def extract_name_token(task: dict[str, Any]) -> str: return "[Name]" if task["input"]["engagement_type"] == "outreach" else "[First Name]"


def infer_intent(task: dict[str, Any]) -> str:
    return {
        "objection_handling": "handle objection and reopen the conversation",
        "follow_up": "earn a reply from a prior thread",
        "discovery_call": "book a discovery conversation",
    }.get(task["input"]["engagement_type"], "start a relevant conversation")


def infer_relationship(task: dict[str, Any]) -> str:
    return "warm" if task["input"]["prior_thread"] else "cold"
def infer_tone(task: dict[str, Any]) -> str: return "polite, confident, low-hype"


def sync_action_fields(task: dict[str, Any], action: str) -> None:
    expected = task["expected_behavior"]
    expected["action"] = action
    tags = task["metadata"].get("tags", [])
    if tags and tags[-1] in {"send", "exploratory_send", "review", "abstain"}:
        tags[-1] = action
    if action == "review":
        expected["cta_required"] = False
        expected["must_include"] = ["manual review"]
        expected["must_avoid"] = sorted(set(expected["must_avoid"]) | {"calendar"})
    if action == "abstain":
        expected["cta_required"] = False
        expected["must_avoid"] = sorted(set(expected["must_avoid"]) | {"calendar"})


def review_or_abstain_body(task: dict[str, Any]) -> str:
    company = task["input"]["company_name"]
    family = task["metadata"]["template_family_id"]
    action = task["expected_behavior"]["action"]
    review_reasons = {
        "review_signal_conflict": "the current signal set mixes expansion-style evidence with an objection that requires judgment rather than an automated reply",
        "review_competitor_gap_unknown": "the available evidence does not support a clean persuasion angle without risking unsupported positioning claims",
        "review_bench_unknown": "the commercial fit remains unresolved from the current evidence bundle",
        "review_schema_fault": "the record includes incomplete or internally inconsistent evidence that should be checked before any response is drafted",
    }
    abstain_reasons = {
        "abstain_missing_evidence": "the available evidence does not establish a strong enough outreach signal",
        "abstain_context_mismatch": "the available evidence does not match a clear outreach angle closely enough",
        "abstain_schema_fault_review": "the available evidence is too incomplete or indirect to automate safely",
    }
    if action == "review":
        reason = review_reasons.get(family, "the current record needs human review before any response is drafted")
        return (
            f"Manual review is required for {company} because {reason}. "
            "Use only the supplied evidence when deciding whether to reply or hold."
        )
    reason = abstain_reasons.get(family, "the available evidence is not strong enough")
    return f"Insufficient signal to send outreach for {company}; {reason}. Route to manual review."


def clean_tokens(text: str) -> set[str]:
    text = normalize(text)
    text = re.sub(r"\bai\b", "AI", text, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z0-9$€+]+", text)
    kept = set()
    for token in tokens:
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        if len(lower) >= 4 or any(ch.isdigit() for ch in token) or "$" in token or "€" in token:
            kept.add(lower)
    return kept


def sentence(text: str) -> str:
    text = normalize(text).rstrip(".")
    if not text:
        return text
    return text[0].upper() + text[1:] + "."


def evidence_categories(text: str) -> set[str]:
    lower = text.lower()
    categories: set[str] = set()
    if any(token in lower for token in ["hired", "hiring", "posted", "roles", "role", "openings", "headcount"]):
        categories.add("hiring")
    if any(token in lower for token in ["raised", "funding", "series", "$", "€", "round"]):
        categories.add("funding")
    if any(token in lower for token in ["launch", "launched", "product", "module", "tool", "platform", "dashboard"]):
        categories.add("launch")
    if any(token in lower for token in ["market", "office", "expansion", "expanded", "europe", "emea", "berlin", "amsterdam"]):
        categories.add("expansion")
    if any(token in lower for token in ["operations", "support", "ops", "coordinator"]):
        categories.add("operations")
    if "ai" in lower or "machine learning" in lower or "data scientist" in lower:
        categories.add("ai")
    return categories


def evidence_clause(company: str, item: str) -> str:
    text = normalize(item).rstrip(".")
    patterns = [
        (r"^Series ([A-Za-z]) funding of (.+)$", lambda m: f"{company} raised a Series {m.group(1).upper()} round of {m.group(2)}"),
        (r"^Series ([A-Za-z]) funding: (.+)$", lambda m: f"{company} announced a Series {m.group(1).upper()} round of {m.group(2)}"),
        (r"^funding round of (.+)$", lambda m: f"{company} closed a funding round of {m.group(1)}"),
        (r"^hiring of (.+)$", lambda m: f"{company} is hiring {m.group(1)}"),
        (r"^hiring (.+)$", lambda m: f"{company} is hiring {m.group(1)}"),
        (r"^hired (.+)$", lambda m: f"{company} hired {m.group(1)}"),
        (r"^recently hired (.+)$", lambda m: f"{company} recently hired {m.group(1)}"),
        (r"^posted (.+)$", lambda m: f"{company} posted {m.group(1)}"),
        (r"^opened (.+)$", lambda m: f"{company} opened {m.group(1)}"),
        (r"^launched (.+)$", lambda m: f"{company} launched {m.group(1)}"),
        (r"^announced (.+)$", lambda m: f"{company} announced {m.group(1)}"),
        (r"^secured (.+)$", lambda m: f"{company} secured {m.group(1)}"),
        (r"^raised (.+)$", lambda m: f"{company} raised {m.group(1)}"),
        (r"^added (.+)$", lambda m: f"{company} added {m.group(1)}"),
        (r"^expanded (.+)$", lambda m: f"{company} expanded {m.group(1)}"),
        (r"^planning to onboard (.+)$", lambda m: f"{company} plans to onboard {m.group(1)}"),
        (r"^webinar transcript: cto mentions ai in supply chain$", lambda m: f"{possessive(company)} CTO mentioned AI in the supply chain during a recent webinar"),
        (r"^ai roles listed on careers page$", lambda m: f"{company} listed AI roles on its careers page"),
        (r"^20 cloud engineers hired in emea region$", lambda m: f"{company} hired 20 cloud engineers in the EMEA region"),
        (r"^expansion into southeast asia market announced in q3$", lambda m: f"{company} announced an expansion into Southeast Asia in Q3"),
        (r"^the company announced (.+)$", lambda m: f"{company} announced {m.group(1)}"),
    ]
    for pattern, builder in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return sentence(builder(match))
    if text.lower().startswith(company.lower()):
        return sentence(text)
    return sentence(f"{company} {text}")


def pick_facts(task: dict[str, Any]) -> list[str]:
    evidence = task["input"]["evidence"]
    family = task["metadata"]["template_family_id"]
    preferred = {
        "send_market_entry_specific": ["expansion", "hiring", "funding"],
        "send_hiring_spike_grounded": ["hiring", "funding", "expansion"],
        "send_grounded_but_generic": ["hiring", "funding", "launch"],
        "send_overblock_risk_clear": ["hiring", "expansion", "funding"],
        "discovery_call_grounded_send": ["hiring", "funding", "launch"],
        "exploratory_ai_hint": ["ai", "hiring", "funding"],
        "exploratory_ops_strain": ["operations", "hiring", "funding"],
        "exploratory_sparse_signal": ["hiring", "launch", "expansion"],
        "outreach_context_repair": ["launch", "hiring", "funding"],
        "objection_budget_pressure": ["expansion", "funding", "launch"],
        "objection_discovery_followup": ["hiring", "funding", "launch"],
    }.get(family, ["hiring", "funding", "launch"])
    chosen: list[str] = []
    for category in preferred:
        for item in evidence:
            if item not in chosen and category in evidence_categories(item):
                chosen.append(item)
                break
        if len(chosen) == 2:
            return chosen
    for item in evidence:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) == 2:
            break
    return chosen


def fallback_subject(task: dict[str, Any]) -> str:
    short = short_company_name(task["input"]["company_name"])
    family = task["metadata"]["template_family_id"]
    subjects = {
        "send_hiring_spike_grounded": f"Quick note on {possessive(short)} recent hiring push",
        "send_grounded_but_generic": f"Following up on {possessive(short)} recent expansion",
        "send_market_entry_specific": f"Quick note on {possessive(short)} market expansion",
        "send_overblock_risk_clear": f"Quick note on {possessive(short)} recent growth signals",
        "discovery_call_grounded_send": f"15-minute chat about {possessive(short)} recent momentum?",
        "exploratory_ai_hint": f"Quick question on {possessive(short)} AI work",
        "exploratory_ops_strain": f"Following up on {possessive(short)} operations buildout",
        "exploratory_sparse_signal": f"Quick follow-up on {possessive(short)} recent updates",
        "outreach_context_repair": f"Quick follow-up on {possessive(short)} upcoming launch",
        "objection_budget_pressure": f"Re: priorities at {short}",
        "objection_discovery_followup": f"Quick follow-up on {possessive(short)} recent expansion",
    }
    return subjects.get(family, f"Quick note on {short}")


def fallback_body(task: dict[str, Any]) -> str:
    company = task["input"]["company_name"]
    action = task["expected_behavior"]["action"]
    family = task["metadata"]["template_family_id"]
    facts = pick_facts(task)
    fact_lines = [evidence_clause(company, fact) for fact in facts]
    pov_map = {
        "send_hiring_spike_grounded": f"That reads like a real hiring signal tied to {possessive(company)} current priorities.",
        "send_grounded_but_generic": "Those are concrete signals of momentum, so I wanted to keep the follow-up specific.",
        "send_market_entry_specific": "That combination points to active market expansion rather than a routine update.",
        "send_overblock_risk_clear": "Taken together, those signals make the timing look real rather than speculative.",
        "discovery_call_grounded_send": "The mix of hiring and company updates looked specific enough to justify a short discovery call.",
        "exploratory_ai_hint": "The outside picture is still early, but the AI signal looked specific enough to ask a short question.",
        "exploratory_ops_strain": "From the outside, that looks like the kind of operational buildout that can justify a low-pressure conversation.",
        "exploratory_sparse_signal": "The picture is still incomplete from the outside, so I’m keeping this brief and exploratory.",
        "outreach_context_repair": "I wanted the follow-up to stay anchored to that context rather than drift into a generic check-in.",
        "objection_budget_pressure": "I’m only following up because those public signals suggest the timing may be worth revisiting later.",
        "objection_discovery_followup": "I’m keeping this concrete and tied only to the public signals already in view.",
    }
    cta = (
        "If this is worth a conversation, I can send a calendar link and hold 15 minutes next week."
        if action == "send"
        else "If you'd like, I can send a calendar link and hold 15 minutes next week."
    )
    lines = [f"Hi {extract_name_token(task)},"]
    if task["input"]["engagement_type"] == "follow_up":
        lines.append("Following up on my earlier note and keeping this specific.")
    elif task["input"]["engagement_type"] == "objection_handling":
        last = task["input"]["prior_thread"][-1]["message"].lower() if task["input"]["prior_thread"] else ""
        lines.append("Understood on the budget pressure this quarter." if "budget" in last else "Thanks for the candid reply on the last note.")
    if len(fact_lines) == 2:
        clause_one = fact_lines[0].rstrip(".")
        clause_two = fact_lines[1].rstrip(".")
        if clause_two.startswith(company):
            clause_two = "it" + clause_two[len(company):]
        lines.append(f"I noticed that {clause_one}, and that {clause_two}.")
    else:
        lines.append(f"I noticed that {fact_lines[0]}")
    lines.append(pov_map.get(family, "The public signals looked specific enough to justify a short note."))
    lines.append(cta)
    lines.append("Best regards,")
    lines.append("[Your Name]")
    return "\n".join(lines)


def build_prompt(task: dict[str, Any]) -> str:
    action = task["expected_behavior"]["action"]
    expected = task["expected_behavior"]
    input_data = task["input"]
    evidence = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(input_data["evidence"]))
    return f"""Rewrite the benchmark gold email for this scenario.

This is a benchmark expected output, not a marketing blast.

Write a fresh subject and body from scratch.

Hard requirements:
- Use only the supplied evidence.
- Mention exactly 1 or 2 evidence items.
- Make the note feel written for this company only.
- Include a clear point of view about why the note is being sent.
- Keep the tone grounded, respectful, and low-hype.
- No fluff, no generic sales filler, no invented pain points, no product claims.
- The result must fail this test: "Could I swap the company name and reuse this?"
- Start with "Hi {extract_name_token(task)},".
- End with:
Best regards,
[Your Name]
- Include a low-pressure calendar CTA.
- Think through this hidden workflow before writing:
  signal -> interpretation -> strategy -> rewrite

Avoid:
{json.dumps(input_data["banned_phrases"], ensure_ascii=False)}

Company: {input_data["company_name"]}
Context: {input_data["company_context"]}
Intent: {infer_intent(task)}
Relationship: {infer_relationship(task)}
Tone constraint: {infer_tone(task)}
Engagement type: {input_data["engagement_type"]}
Signal strength: {input_data["signal_strength"]}
Hiring signal: {input_data["hiring_signal"]}
Prior thread: {json.dumps(input_data["prior_thread"], ensure_ascii=False)}
Must include: {json.dumps(expected["must_include"], ensure_ascii=False)}
Evidence:
{evidence}

Return strict JSON:
{{
  "email_subject": "...",
  "email_body": "...",
  "used_evidence_indices": [1, 2],
  "swap_test_passed": true,
  "why_not_swappable": "one short sentence"
}}
"""


def validate_candidate(task: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    subject = normalize(str(candidate.get("email_subject", "")))
    body = str(candidate.get("email_body", "")).strip()
    body_lower = body.lower()
    full_text = f"{subject.lower()}\n{body_lower}"
    used = candidate.get("used_evidence_indices", [])
    company = task["input"]["company_name"]
    short = short_company_name(company).lower()
    greeting = f"Hi {extract_name_token(task)},"

    if not subject:
        errors.append("missing_subject")
    if not body:
        errors.append("missing_body")
    if not body.startswith(greeting):
        errors.append("bad_greeting")
    if not body.endswith("Best regards,\n[Your Name]"):
        errors.append("bad_signature")
    if not isinstance(used, list) or not (1 <= len(used) <= 2):
        errors.append("bad_used_evidence")
    elif any(not isinstance(idx, int) or idx < 1 or idx > len(task["input"]["evidence"]) for idx in used):
        errors.append("used_evidence_out_of_range")
    if not candidate.get("swap_test_passed"):
        errors.append("swap_test_failed")
    if company.lower() not in body_lower and short not in body_lower:
        errors.append("company_not_mentioned")
    if task["expected_behavior"]["cta_required"] and "calendar" not in body_lower:
        errors.append("missing_calendar")

    for phrase in task["input"]["banned_phrases"]:
        if phrase.lower() in full_text:
            errors.append(f"banned_phrase:{phrase}")
    for phrase in GENERIC_PHRASES:
        if phrase in full_text:
            errors.append(f"generic_phrase:{phrase}")

    if isinstance(used, list) and used:
        matched = 0
        company_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", company)}
        for idx in used:
            if not isinstance(idx, int) or idx < 1 or idx > len(task["input"]["evidence"]):
                continue
            anchors = clean_tokens(task["input"]["evidence"][idx - 1]) - company_tokens
            if sum(1 for token in anchors if token in body_lower) >= 2:
                matched += 1
        if matched == 0:
            errors.append("not_grounded_in_selected_evidence")
    return errors


def request_candidate(
    client: Any,
    model: str,
    prompt: str,
    repair_errors: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    user_prompt = prompt
    if repair_errors:
        user_prompt += (
            "\n\nYour previous attempt failed validation. Return fresh JSON only and fix these issues: "
            + ", ".join(repair_errors)
        )
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You write benchmark gold emails. Output JSON only."},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content or "{}"), response.model_dump()


def regenerate_email(task: dict[str, Any], client: Any, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_prompt(task)
    candidate: dict[str, Any] = {}
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost_usd = 0.0
    for attempt_no in range(2):
        candidate, response_json = request_candidate(client, model, prompt, errors if attempt_no else None)
        attempt_usage = extract_usage(response_json)
        usage = {key: usage[key] + attempt_usage[key] for key in usage}
        cost_usd += extract_cost(response_json)
        errors = validate_candidate(task, candidate)
        attempts.append(
            {
                "attempt": attempt_no + 1,
                "errors": errors,
                "used_evidence_indices": candidate.get("used_evidence_indices", []),
                "swap_test_passed": bool(candidate.get("swap_test_passed")),
            }
        )
        if not errors:
            break

    report = {
        "task_id": task["task_id"],
        "accepted": not errors,
        "errors": errors,
        "attempts": attempts,
        "used_evidence_indices": candidate.get("used_evidence_indices", []),
        "usage": usage,
        "cost_usd": round(cost_usd, 6),
    }
    if errors:
        return task, report

    revised = json.loads(json.dumps(task))
    revised["expected_behavior"]["expected_output"]["email_subject"] = normalize(candidate["email_subject"])
    revised["expected_behavior"]["expected_output"]["email_body"] = str(candidate["email_body"]).strip()
    return revised, report


def resolve_api_client(provider: str, model_override: str | None) -> tuple[Any, str]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    load_dotenv()
    chosen = provider
    if provider == "auto":
        chosen = "openai" if os.getenv("OPENAI_API_KEY") else "openrouter"
    if chosen == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        model = model_override or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        return OpenAI(api_key=api_key), model
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    model = model_override or os.getenv("SYNTHESIS_PRIMARY_MODEL") or "openai/gpt-4.1-mini"
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), model


def revise_task(task: dict[str, Any], client: Any | None, model: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    revised = json.loads(json.dumps(task))
    override = ACTION_OVERRIDES.get(revised["task_id"])
    if override:
        sync_action_fields(revised, override)

    action = revised["expected_behavior"]["action"]
    if action in {"review", "abstain"}:
        revised["expected_behavior"]["expected_output"]["email_subject"] = ""
        revised["expected_behavior"]["expected_output"]["email_body"] = review_or_abstain_body(revised)
        return revised, None

    revised["expected_behavior"]["expected_output"]["email_subject"] = fallback_subject(revised)
    revised["expected_behavior"]["expected_output"]["email_body"] = fallback_body(revised)
    if client is None or model is None:
        return revised, None

    api_revised, report = regenerate_email(revised, client, model)
    if report and report["accepted"]:
        return api_revised, report
    return revised, report


def build_report(tasks: list[dict[str, Any]], api_reports: list[dict[str, Any]]) -> dict[str, Any]:
    actions = {"send": 0, "exploratory_send": 0, "review": 0, "abstain": 0}
    for task in tasks:
        action = task["expected_behavior"]["action"]
        if action in actions:
            actions[action] += 1
    return {
        "task_count": len(tasks),
        "actions": actions,
        "preference_pair_ready_tasks": sum(1 for task in tasks if task["expected_behavior"]["action"] in {"send", "exploratory_send"}),
        "api_attempts": len(api_reports),
        "api_accepted": sum(1 for item in api_reports if item["accepted"]),
        "api_rejected": sum(1 for item in api_reports if not item["accepted"]),
        "api_reports": api_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Revise benchmark gold outputs.")
    parser.add_argument("--input", default="tenacious_bench_v0.1/llm_pool_unsplit.jsonl")
    parser.add_argument("--output", default="tenacious_bench_v0.1/llm_pool_unsplit.jsonl")
    parser.add_argument("--report", default="tenacious_bench_v0.1/llm_pool_revision_report.json")
    parser.add_argument("--cost-log", default="cost_log.csv")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-provider", choices=["auto", "openai", "openrouter"], default="auto")
    parser.add_argument("--api-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-ids", nargs="*")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    tasks = load_jsonl(input_path)

    selected = set(args.task_ids or [])
    client = model = None
    if args.use_api:
        client, model = resolve_api_client(args.api_provider, args.api_model)

    revised_tasks: list[dict[str, Any]] = []
    api_reports: list[dict[str, Any]] = []
    api_done = 0
    for task in tasks:
        should_try_api = (
            args.use_api
            and task["expected_behavior"]["action"] in {"send", "exploratory_send"}
            and (not selected or task["task_id"] in selected)
            and (args.limit is None or api_done < args.limit)
        )
        revised, api_report = revise_task(task, client if should_try_api else None, model if should_try_api else None)
        if should_try_api:
            api_done += 1
        if api_report:
            api_reports.append(api_report)
            append_cost_log(Path(args.cost_log), model or "", api_report["usage"], api_report["cost_usd"], int(api_report["accepted"]), int(not api_report["accepted"]), f"stage=llm_pool_revision;task_id={task['task_id']};status={'accepted' if api_report['accepted'] else 'rejected'}")
        revised_tasks.append(revised)

    write_jsonl(output_path, revised_tasks)
    report_path.write_text(json.dumps(build_report(revised_tasks, api_reports), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
