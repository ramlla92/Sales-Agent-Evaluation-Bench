import argparse
import json
import os
from itertools import cycle
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import load_dotenv
try:
    from generation_scripts.llm_synthesis_core import (
        GENERIC_BANNED_PHRASES,
        append_cost_log,
        build_generation_plan,
        changed_factor_names,
        coverage_from_plan,
        coverage_from_tasks,
        extract_cost,
        extract_usage,
        load_generator_models,
        normalize_text,
        print_coverage,
        prior_thread_hint,
        repair_required_cta,
        sanitize_review_output,
        reject_reason,
        slugify,
        summarize,
        validate_model_separation,
        validate_task_structure,
    )
except ModuleNotFoundError:
    from llm_synthesis_core import (
        GENERIC_BANNED_PHRASES,
        append_cost_log,
        build_generation_plan,
        changed_factor_names,
        coverage_from_plan,
        coverage_from_tasks,
        extract_cost,
        extract_usage,
        load_generator_models,
        normalize_text,
        print_coverage,
        prior_thread_hint,
        repair_required_cta,
        sanitize_review_output,
        reject_reason,
        slugify,
        summarize,
        validate_model_separation,
        validate_task_structure,
    )


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUTPUT_JSONL = "tenacious_bench_v0.1/llm_pool_unsplit.jsonl"
DEFAULT_SUMMARY_JSON = "tenacious_bench_v0.1/llm_pool_unsplit_summary.json"
DEFAULT_COST_LOG = "cost_log.csv"
DEFAULT_VERSION = "0.3"

SCENARIO_CONFIG_PATH = Path(__file__).with_name("llm_synthesis_scenarios.json")


def load_scenario_catalog() -> dict[str, Any]:
    with SCENARIO_CONFIG_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    if "scenario_specs" not in catalog or "priority_template_order" not in catalog:
        raise ValueError(f"Scenario catalog is missing required keys: {SCENARIO_CONFIG_PATH}")
    return catalog


SCENARIO_CATALOG = load_scenario_catalog()
SCENARIO_SPECS: list[dict[str, Any]] = SCENARIO_CATALOG["scenario_specs"]
PRIORITY_TEMPLATE_ORDER: list[str] = SCENARIO_CATALOG["priority_template_order"]


def build_prompt(base_spec: dict[str, Any], scenario: dict[str, Any], variant_index: int) -> str:
    changed = changed_factor_names(base_spec, scenario)
    if len(changed) > 2:
        raise ValueError(
            f"Scenario {scenario['template_family_id']} varies more than two controlled factors from its template: {changed}"
        )

    return f"""
You are authoring one synthetic benchmark example for a B2B sales-agent evaluation dataset.

Return JSON only with these keys:
- company_name
- company_context
- hiring_signal
- evidence
- segment
- segment_confidence
- ai_maturity
- prior_thread
- expected_output_subject
- expected_output_body

Controlled factors for this task:
- template_family_id: {scenario['template_family_id']}
- engagement_type: {scenario['engagement_type']}
- task_type: {scenario['task_type']}
- difficulty: {scenario['difficulty']}
- action: {scenario['action']}
- failure_mode_tags: {", ".join(scenario['failure_mode_tags'])}
- signal_strength: {scenario['signal_strength']}
- evidence_completeness: {scenario['evidence_completeness']}
- segment_confidence_band: {scenario['segment_confidence_band']}
- ai_maturity: {scenario['ai_maturity']}
- changed_factors_from_template: {", ".join(changed) if changed else "none"}
- scenario_brief: {scenario['prompt_brief']}

Generation constraints:
- use realistic public-signal style evidence
- evidence must contain exactly 3 short concrete strings
- at least 2 evidence items must include concrete anchors such as a role title, role count, funding amount/round, market/geography change, product launch, ops hiring, or named function
- do not use placeholder company names like "Acme", "Example Company", or "Sample Company"
- do not invent competitor gaps unless explicitly grounded in evidence
- do not use banned phrases: {", ".join(GENERIC_BANNED_PHRASES)}
- keep this variant distinct from others; this is variant {variant_index + 1}
- if action is send or exploratory_send, expected_output_body must read like a real email, mention a calendar, and include a specific meeting ask such as "15-minute" or "next week"
- if action is review, expected_output_body must explicitly say manual review is required and expected_output_subject should be empty
- if action is abstain, expected_output_body must explicitly say signal is insufficient and expected_output_subject should be empty
- the output email may only reference claims grounded in the evidence, company_context, hiring_signal, or prior_thread
- if engagement_type is outreach, prior_thread should be []
- if engagement_type is follow_up, prior_thread should include prior contact context
- if engagement_type is objection_handling, prior_thread should include a prospect objection
- if engagement_type is discovery_call, prior_thread should include an internal qualification note

Segment confidence guidance:
- low band: 0.30 to 0.49
- mid band: 0.50 to 0.69
- high band: 0.70 to 0.90
""".strip()


def load_existing_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def template_variant_offsets(tasks: list[dict[str, Any]]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    for task in tasks:
        template_family_id = task.get("metadata", {}).get("template_family_id")
        if not template_family_id:
            continue
        offsets[template_family_id] = offsets.get(template_family_id, 0) + 1
    return offsets


def apply_variant_offsets(plan: list[dict[str, Any]], existing_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets = template_variant_offsets(existing_tasks)
    adjusted_plan: list[dict[str, Any]] = []
    local_counts: dict[str, int] = {}
    for scenario in plan:
        template_family_id = scenario["base_template_family_id"]
        offset = offsets.get(template_family_id, 0)
        local_index = local_counts.get(template_family_id, 0)
        updated = dict(scenario)
        updated["variant_index"] = offset + local_index
        adjusted_plan.append(updated)
        local_counts[template_family_id] = local_index + 1
    return adjusted_plan


def post_openrouter(model: str, prompt: str) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing in .env")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate only valid compact JSON. No markdown. No commentary."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    raw = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OPENROUTER_BASE_URL,
        data=raw,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.tenacious-bench",
            "X-Title": "Tenacious-Bench LLM Synthesis",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
    return json.loads(body)


def extract_content(response_json: dict[str, Any]) -> dict[str, Any]:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("No choices returned by model.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content or "")
    return json.loads(text)


def normalize_prior_thread(value: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return fallback
    cleaned = []
    for item in value[:3]:
        if isinstance(item, dict):
            speaker = normalize_text(item.get("speaker"))
            message = normalize_text(item.get("message"))
            if speaker and message:
                cleaned.append({"speaker": speaker, "message": message})
    return cleaned or fallback


def confidence_from_band(band: str, raw_value: float) -> float:
    if band == "low":
        return min(max(raw_value, 0.30), 0.49)
    if band == "mid":
        return min(max(raw_value, 0.50), 0.69)
    return min(max(raw_value, 0.70), 0.90)


def build_task(index: int, base_spec: dict[str, Any], scenario: dict[str, Any], generated: dict[str, Any], model: str) -> dict[str, Any]:
    fallback_thread = prior_thread_hint(scenario["engagement_type"], scenario["template_family_id"], scenario["variant_index"])
    prior_thread = normalize_prior_thread(generated.get("prior_thread"), fallback_thread)

    raw_confidence = float(generated.get("segment_confidence", 0.5))
    segment_confidence = confidence_from_band(scenario["segment_confidence_band"], raw_confidence)
    ai_maturity = int(generated.get("ai_maturity", scenario["ai_maturity"]))
    ai_maturity = max(0, min(3, ai_maturity))

    expected_subject = normalize_text(generated.get("expected_output_subject", ""))
    expected_body = normalize_text(generated.get("expected_output_body", ""))
    if scenario["action"] in {"abstain", "review"}:
        expected_subject = ""
    if scenario["action"] == "abstain" and "insufficient signal" not in expected_body.lower():
        expected_body = f"Insufficient signal to send outreach for {generated['company_name']}. Route to manual review."
    if scenario["action"] == "review" and "manual review" not in expected_body.lower():
        expected_body = f"Manual review required before outreach for {generated['company_name']}. The signal is promising but not safe enough to automate yet."

    task = {
        "task_id": f"tb_llm_{index:04d}",
        "split": "unsplit",
        "source_mode": "multi_llm_synthesis",
        "difficulty": scenario["difficulty"],
        "task_type": scenario["task_type"],
        "failure_mode_tags": scenario["failure_mode_tags"],
        "input": {
            "company_name": normalize_text(generated["company_name"]),
            "company_context": normalize_text(generated["company_context"]),
            "hiring_signal": normalize_text(generated["hiring_signal"]),
            "evidence": [normalize_text(item) for item in generated["evidence"][:3]],
            "segment": normalize_text(generated.get("segment", "unknown")),
            "segment_confidence": segment_confidence,
            "ai_maturity": ai_maturity,
            "engagement_type": scenario["engagement_type"],
            "signal_strength": scenario["signal_strength"],
            "evidence_completeness": scenario["evidence_completeness"],
            "segment_confidence_band": scenario["segment_confidence_band"],
            "prior_thread": prior_thread,
            "guardrails": [
                "Use only the supplied evidence.",
                "Do not invent internal metrics.",
                "Do not claim verified bench fit when unknown.",
            ],
            "banned_phrases": GENERIC_BANNED_PHRASES,
            "disallowed_claims": [
                "unsupported competitor gap",
                "invented hiring momentum",
                "claiming verified bench fit when unknown",
            ],
            "tone_markers": ["grounded", "respectful", "low-hype"],
        },
        "expected_behavior": {
            "action": scenario["action"],
            "must_include": scenario["must_include"],
            "must_avoid": scenario["must_avoid"],
            "cta_required": scenario["cta_required"],
            "decision_rationale": f"LLM-synthesized task for template family {scenario['template_family_id']}.",
            "expected_output": {
                "email_subject": expected_subject,
                "email_body": expected_body,
            },
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
            "version": DEFAULT_VERSION,
            "source_notes": "LLM-synthesized from a controlled template family prompt.",
            "template_family_id": scenario["template_family_id"],
            "generation_model": model,
            "judge_model": os.getenv("SYNTHESIS_JUDGE_MODEL", "").strip(),
            "authoring_revision": "llm_synthesis_v2",
            "tags": [
                "multi_llm_synthesis",
                scenario["engagement_type"],
                scenario["template_family_id"],
                scenario["action"],
            ],
        },
    }
    validate_task_structure(task)
    return task
def run_generation(
    plan: list[dict[str, Any]],
    generator_models: list[str],
    output_jsonl: Path,
    cost_log_path: Path,
    batch_size: int,
    existing_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    accepted_tasks: list[dict[str, Any]] = list(existing_tasks)
    seen_pairs: set[tuple[str, str]] = set()
    for task in existing_tasks:
        pair = (task["metadata"]["template_family_id"], task["input"]["company_name"].lower())
        seen_pairs.add(pair)
    model_cycle = cycle(generator_models)

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for task in existing_tasks:
            out_f.write(json.dumps(task, ensure_ascii=False) + "\n")
        for batch_start in range(0, len(plan), batch_size):
            batch = plan[batch_start : batch_start + batch_size]
            for scenario in batch:
                model = next(model_cycle)
                prompt = build_prompt(next(spec for spec in SCENARIO_SPECS if spec["template_family_id"] == scenario["base_template_family_id"]), scenario, scenario["variant_index"])
                response_json = post_openrouter(model, prompt)
                usage = extract_usage(response_json)
                cost_usd = extract_cost(response_json)
                accepted = 0
                rejected = 0
                notes = f"template_family_id={scenario['template_family_id']};variant={scenario['variant_index'] + 1}"
                try:
                    generated = extract_content(response_json)
                    task = build_task(len(accepted_tasks) + 1, next(spec for spec in SCENARIO_SPECS if spec["template_family_id"] == scenario["base_template_family_id"]), scenario, generated, model)
                    repair_required_cta(task)
                    sanitize_review_output(task)
                    reason = reject_reason(task, accepted_tasks, seen_pairs)
                    if reason is not None:
                        rejected = 1
                        notes = f"{notes};status=rejected;reason={reason}"
                    else:
                        pair = (task["metadata"]["template_family_id"], task["input"]["company_name"].lower())
                        seen_pairs.add(pair)
                        accepted_tasks.append(task)
                        out_f.write(json.dumps(task, ensure_ascii=False) + "\n")
                        accepted = 1
                        notes = f"{notes};status=accepted"
                except Exception as exc:
                    rejected = 1
                    notes = f"{notes};status=rejected;reason={slugify(str(exc))[:80]}"
                append_cost_log(cost_log_path, model, usage, cost_usd, accepted, rejected, notes)
    return accepted_tasks


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate Tenacious-Bench multi-LLM synthesis tasks.")
    parser.add_argument("--count", type=int, default=16, help="Total number of tasks to generate. Must be divisible by the number of base scenario specs.")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_JSON, help="Summary JSON path.")
    parser.add_argument("--cost-log", default=DEFAULT_COST_LOG, help="CSV log file for API-call costs.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("SYNTHESIS_BATCH_SIZE", "10")), help="Generation batch size.")
    parser.add_argument("--seed-jsonl", default="", help="Existing canonical JSONL pool to retain and extend without duplication.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned distribution without calling APIs.")
    args = parser.parse_args()

    generator_models = load_generator_models()
    validate_model_separation(generator_models)

    existing_tasks = load_existing_tasks(Path(args.seed_jsonl)) if args.seed_jsonl else []
    plan = build_generation_plan(args.count, SCENARIO_SPECS, PRIORITY_TEMPLATE_ORDER)
    plan = apply_variant_offsets(plan, existing_tasks)
    planned_coverage = coverage_from_tasks(existing_tasks) if existing_tasks else {
        "engagement_type": {},
        "action": {},
        "difficulty": {},
        "failure_mode_tags": {},
        "template_family_id": {},
    }
    plan_only_coverage = coverage_from_plan(plan)
    for key, counts in plan_only_coverage.items():
        merged = dict(planned_coverage.get(key, {}))
        for label, value in counts.items():
            merged[label] = merged.get(label, 0) + value
        planned_coverage[key] = dict(sorted(merged.items()))
    print_coverage("planned_coverage", planned_coverage)

    if args.dry_run:
        return

    accepted_tasks = run_generation(
        plan,
        generator_models,
        Path(args.output_jsonl),
        Path(args.cost_log),
        args.batch_size,
        existing_tasks,
    )

    actual_coverage = coverage_from_tasks(accepted_tasks)
    print_coverage("actual_coverage", actual_coverage)

    summary = summarize(accepted_tasks, planned_coverage)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
