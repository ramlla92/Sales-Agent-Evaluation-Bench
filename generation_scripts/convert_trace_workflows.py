import argparse
import json
import re
from pathlib import Path
from typing import Any


GENERIC_BANNED_PHRASES = [
    "synergies",
    "best-in-class",
    "10x your pipeline",
    "revolutionary platform",
    "scaling challenges",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_email(record: dict[str, Any]) -> dict[str, str]:
    email = record.get("outreach_email") or {}
    subject = normalize(email.get("subject", ""))
    raw_body = email.get("body", "") or ""
    body = re.sub(r"^\s*Subject:\s*[^\n]+\n*", "", raw_body, flags=re.IGNORECASE)
    body = normalize(body)
    return {
        "email_subject": subject,
        "email_body": body,
    }


def clip(text: str, limit: int = 120) -> str:
    text = normalize(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def primary_firmographic_info(record: dict[str, Any]) -> tuple[str, str, str]:
    brief = record.get("hiring_signal_brief") or {}
    for item in (brief.get("evidence") or []):
        if item.get("label") == "crunchbase_firmographics":
            metadata = item.get("metadata") or {}
            country = slugify(str(metadata.get("country_code", "") or "xx")) or "xx"
            industries_raw = str(metadata.get("industries", "") or "")
            industry_match = re.search(r'"value":"([^"]+)"', industries_raw)
            industry = slugify(industry_match.group(1)) if industry_match else "unknown"
            return "crunchbase_firmographics", country, industry
    if (brief.get("evidence") or []):
        label = slugify(str((brief.get("evidence") or [])[0].get("label", "unknown")))
        return label or "unknown", "xx", "unknown"
    return "unknown", "xx", "unknown"


def honesty_bucket(record: dict[str, Any]) -> str:
    count = len(record.get("honesty_flags") or [])
    if count >= 4:
        return "hf4p"
    if count == 3:
        return "hf3"
    if count == 2:
        return "hf2"
    if count == 1:
        return "hf1"
    return "hf0"


def build_evidence(record: dict[str, Any]) -> list[str]:
    brief = record.get("hiring_signal_brief") or {}
    evidence_items: list[str] = []

    for item in (brief.get("evidence") or [])[:3]:
        label = normalize(item.get("label", ""))
        value = normalize(item.get("value", ""))
        evidence_text = normalize(item.get("evidence_text", ""))
        if label and value:
            evidence_items.append(clip(f"{label}: {value}", 140))
        elif evidence_text:
            evidence_items.append(clip(evidence_text, 140))

    if not evidence_items:
        summaries = [
            brief.get("job_post_summary", ""),
            brief.get("funding_summary", ""),
            brief.get("tech_stack_summary", ""),
            brief.get("layoffs_summary", ""),
        ]
        for summary in summaries:
            summary = normalize(summary)
            if summary and not summary.lower().startswith("no "):
                evidence_items.append(clip(summary, 140))
            if len(evidence_items) >= 3:
                break

    return evidence_items[:3]


def infer_failure_tags(record: dict[str, Any], email_body: str) -> list[str]:
    tags: list[str] = []
    body_norm = email_body.lower()
    signal_grounding_score = record.get("signal_grounding_score")
    factual_accuracy_score = record.get("factual_accuracy_score")

    if (signal_grounding_score if signal_grounding_score is not None else 0.0) < 0.75:
        tags.append("T-01")
    if (factual_accuracy_score if factual_accuracy_score is not None else 1.0) < 0.6 or any(
        flag in (record.get("honesty_flags") or [])
        for flag in ["layoff_overrides_funding", "tech_stack_inferred_not_confirmed", "weak_ai_maturity_signal"]
    ):
        tags.append("T-02")
    if any(phrase in body_norm for phrase in ["synergies", "scaling challenges", "how are you currently", "explore potential"]):
        tags.append("T-03")
    if "calendar" not in body_norm and "15-minute" not in body_norm and "15 minute" not in body_norm and "next week" not in body_norm:
        tags.append("T-04")
    if record.get("email_sent") and record.get("ai_maturity", 0) <= 1 and record.get("segment_confidence", 1) < 0.5:
        tags.append("T-05")
    if (record.get("company_name") or "").lower() == "wise":
        tags.append("T-06")
    return sorted(set(tags)) or ["T-01"]


def infer_task_type(record: dict[str, Any]) -> str:
    prior_thread = record.get("prior_thread") or []
    if prior_thread:
        return "reply_to_objection"
    if record.get("email_sent") is False or not record.get("outreach_email"):
        return "abstain_decision"
    return "generate_outreach"


def infer_action(record: dict[str, Any], failure_tags: list[str]) -> str:
    segment_confidence = float(record.get("segment_confidence") or 0.0)
    ai_maturity = int(record.get("ai_maturity") or 0)
    honesty_flags = record.get("honesty_flags") or []
    evidence_count = len(build_evidence(record))
    factual_accuracy_score = record.get("factual_accuracy_score")
    factual_accuracy_score = 1.0 if factual_accuracy_score is None else float(factual_accuracy_score)

    if record.get("email_sent") is False or not record.get("outreach_email"):
        return "abstain"

    # Only abstain on the clearest thin-signal cases.
    if (
        segment_confidence < 0.42
        and ai_maturity <= 1
        and len(honesty_flags) >= 3
        and evidence_count <= 2
        and factual_accuracy_score < 0.55
    ):
        return "abstain"

    # Weak but non-empty evidence should generally become exploratory, not abstain.
    if record.get("outreach_route") == "exploratory":
        return "exploratory_send"
    if segment_confidence < 0.55 or ai_maturity <= 1:
        return "exploratory_send"
    return "send"


def infer_observed_action(record: dict[str, Any]) -> str:
    if record.get("email_sent") is False or not record.get("outreach_email"):
        return "abstain"
    if record.get("outreach_route") == "exploratory":
        return "exploratory_send"
    return "send"


def infer_family_id(
    record: dict[str, Any],
    task_type: str,
    observed_action: str,
    expected_action: str,
    failure_tags: list[str],
) -> str:
    ai_maturity = record.get("ai_maturity", 0)
    segment_confidence = record.get("segment_confidence", 0)
    if segment_confidence < 0.5:
        confidence_bucket = "low"
    elif segment_confidence < 0.75:
        confidence_bucket = "mid"
    else:
        confidence_bucket = "high"
    primary_tags = "_".join(failure_tags[:2]) if failure_tags else "none"
    source_label, country, industry = primary_firmographic_info(record)
    hf_bucket = honesty_bucket(record)
    return (
        f"fam_{task_type}_{observed_action}_to_{expected_action}_{primary_tags}"
        f"_ai{ai_maturity}_{confidence_bucket}_{source_label}_{country}_{industry}_{hf_bucket}"
    )


def infer_conversion_confidence(record: dict[str, Any], expected_action: str, observed_action: str) -> str:
    segment_confidence = float(record.get("segment_confidence") or 0.0)
    honesty_flags = record.get("honesty_flags") or []
    factual_accuracy_score = record.get("factual_accuracy_score")
    factual_accuracy_score = 1.0 if factual_accuracy_score is None else float(factual_accuracy_score)

    if expected_action == observed_action:
        return "high"
    if expected_action == "abstain" and segment_confidence < 0.42 and len(honesty_flags) >= 3 and factual_accuracy_score < 0.55:
        return "high"
    if expected_action == "exploratory_send" and segment_confidence < 0.55:
        return "medium"
    return "low"


def build_expected_output(action: str, observed_output: dict[str, str], company_name: str) -> dict[str, str]:
    if action == "abstain":
        return {
            "email_subject": "",
            "email_body": f"Insufficient signal to send outreach for {company_name}. Route to manual review.",
        }
    return observed_output


def build_hiring_signal(record: dict[str, Any]) -> str:
    brief = record.get("hiring_signal_brief") or {}
    candidates = [
        brief.get("job_post_summary", ""),
        brief.get("funding_summary", ""),
        brief.get("ai_maturity", {}).get("summary", "") if isinstance(brief.get("ai_maturity"), dict) else "",
        brief.get("signal_summary", ""),
    ]
    for candidate in candidates:
        candidate = normalize(candidate)
        if candidate:
            return clip(candidate, 160)
    return "Public evidence requires review."


def build_input(record: dict[str, Any], evidence: list[str]) -> dict[str, Any]:
    brief = record.get("hiring_signal_brief") or {}
    outreach_email = record.get("outreach_email") or {}
    input_payload = {
        "company_name": record.get("company_name", ""),
        "hiring_signal": build_hiring_signal(record),
        "evidence": evidence,
        "segment_confidence": float(record.get("segment_confidence") or 0.0),
        "ai_maturity": int(record.get("ai_maturity") or 0),
    }

    segment = record.get("icp_segment")
    if segment:
        input_payload["segment"] = segment

    honesty_flags = record.get("honesty_flags") or []
    if honesty_flags:
        input_payload["honesty_flags"] = honesty_flags

    safety_notes = outreach_email.get("safety_notes") or []
    if safety_notes:
        input_payload["guardrails"] = safety_notes[:4]

    company_context_bits = []
    for key in ["funding_summary", "tech_stack_summary", "leadership_summary"]:
        value = normalize(brief.get(key, ""))
        if value and not value.lower().startswith("no "):
            company_context_bits.append(clip(value, 80))
    if company_context_bits:
        input_payload["company_context"] = " | ".join(company_context_bits[:2])

    input_payload["banned_phrases"] = GENERIC_BANNED_PHRASES
    input_payload["disallowed_claims"] = [
        "unsupported competitor gap",
        "invented hiring momentum",
        "claiming verified bench fit when unknown",
    ]
    input_payload["tone_markers"] = ["grounded", "respectful", "low-hype"]
    return input_payload


def build_expected_behavior(
    record: dict[str, Any],
    expected_action: str,
    observed_action: str,
    observed_output: dict[str, str],
    evidence: list[str],
) -> dict[str, Any]:
    must_include = []
    if evidence:
        must_include.append(evidence[0])
    if expected_action != "abstain":
        must_include.append("calendar")

    behavior = {
        "action": expected_action,
        "observed_action": observed_action,
        "must_include": must_include,
        "must_avoid": GENERIC_BANNED_PHRASES + ["unsupported competitor gap"],
        "cta_required": expected_action != "abstain",
        "decision_rationale": "Auto-converted from Week 10 workflow output; review before publication.",
        "expected_output": build_expected_output(expected_action, observed_output, record.get("company_name", "this company")),
    }
    if observed_output["email_subject"] or observed_output["email_body"]:
        behavior["observed_output"] = observed_output
    return behavior


def convert_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    observed_output = extract_email(record)
    failure_tags = infer_failure_tags(record, observed_output["email_body"])
    task_type = infer_task_type(record)
    observed_action = infer_observed_action(record)
    expected_action = infer_action(record, failure_tags)
    evidence = build_evidence(record)
    conversion_confidence = infer_conversion_confidence(record, expected_action, observed_action)

    return {
        "task_id": f"tb_trace_{index:04d}",
        "split": "unsplit",
        "source_mode": "trace_derived",
        "difficulty": "hard" if ("T-05" in failure_tags or "T-02" in failure_tags or expected_action != observed_action) else "medium",
        "task_type": task_type,
        "failure_mode_tags": failure_tags,
        "input": build_input(record, evidence),
        "expected_behavior": build_expected_behavior(record, expected_action, observed_action, observed_output, evidence),
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
            "trace_id": record.get("trace_id", ""),
            "source_notes": "Auto-converted from Week 10 workflow output.",
            "family_id": infer_family_id(record, task_type, observed_action, expected_action, failure_tags),
            "conversion_confidence": conversion_confidence,
            "authoring_revision": "trace_conversion_v3",
            "tags": [slugify(record.get("company_name", "")), expected_action, task_type],
        },
    }


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}

    for task in tasks:
        family_id = task["metadata"]["family_id"]
        family_counts[family_id] = family_counts.get(family_id, 0) + 1
        action = task["expected_behavior"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
        for tag in task["failure_mode_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "task_count": len(tasks),
        "families": dict(sorted(family_counts.items())),
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "actions": dict(sorted(action_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Week 10 workflow outputs into compact Tenacious-Bench trace-derived tasks.")
    parser.add_argument("--input", default="week_10_artifacts/workflow_results_with_emails.jsonl", help="Path to workflow results JSONL.")
    parser.add_argument("--output-jsonl", default="tenacious_bench_v0.1/trace_pool_unsplit.jsonl", help="Path to write compact unsplit tasks JSONL.")
    parser.add_argument("--summary-out", default="tenacious_bench_v0.1/trace_pool_unsplit_summary.json", help="Path to write conversion summary.")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    tasks = [convert_record(record, idx) for idx, record in enumerate(records, start=1)]
    summary = summarize(tasks)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
