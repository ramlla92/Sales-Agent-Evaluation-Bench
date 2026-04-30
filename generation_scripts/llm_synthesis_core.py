import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


GENERIC_BANNED_PHRASES = [
    "synergies",
    "best-in-class",
    "10x your pipeline",
    "revolutionary platform",
    "scaling challenges",
]

CONCRETE_SIGNAL_HINTS = [
    "sdr",
    "ae",
    "revops",
    "series",
    "funding",
    "market",
    "expansion",
    "layoff",
    "headcount",
    "role",
    "hiring",
    "job post",
    "calendar",
    "discovery",
    "pipeline",
    "operations",
    "territory",
    "geo",
    "launch",
]

REQUIRED_CTA_HINTS = [
    "calendar",
    "15-minute",
    "15 minute",
    "next week",
    "brief call",
]

CLAIM_CHECK_TERMS = [
    "competitor",
    "funding",
    "layoff",
    "revops",
    "sdr",
    "ae",
    "market expansion",
    "geographic expansion",
    "ai investment",
    "hiring spike",
    "bench fit",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def load_generator_models() -> list[str]:
    models = [
        os.getenv("SYNTHESIS_PRIMARY_MODEL", "").strip(),
        os.getenv("SYNTHESIS_VARIANT_MODEL", "").strip(),
        os.getenv("SYNTHESIS_CHEAP_MODEL", "").strip(),
    ]
    models = [model for model in models if model]
    unique_models = list(dict.fromkeys(models))
    if len(unique_models) < 2:
        raise ValueError("Set at least two distinct generator models in .env for multi-LLM synthesis.")
    return unique_models


def validate_model_separation(generator_models: list[str]) -> None:
    reserved_judge_models = {
        os.getenv("SYNTHESIS_JUDGE_MODEL", "").strip(),
        os.getenv("TRAINING_JUDGE_MODEL", "").strip(),
    }
    reserved_judge_models = {model for model in reserved_judge_models if model}
    overlap = reserved_judge_models.intersection(generator_models)
    if overlap:
        raise ValueError(f"Generator models must differ from judge models. Overlap: {sorted(overlap)}")


def variation_overrides(spec: dict[str, Any], variant_index: int) -> dict[str, Any]:
    allowed = spec.get("allowed_variations") or []
    if not allowed:
        return {}
    return dict(allowed[variant_index % len(allowed)])


def apply_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(spec)
    scenario.update(overrides)
    return scenario


def changed_factor_names(base_spec: dict[str, Any], scenario: dict[str, Any]) -> list[str]:
    factor_keys = [
        "engagement_type",
        "task_type",
        "difficulty",
        "action",
        "failure_mode_tags",
        "signal_strength",
        "evidence_completeness",
        "segment_confidence_band",
        "ai_maturity",
        "template_family_id",
    ]
    return [key for key in factor_keys if scenario.get(key) != base_spec.get(key)]


def prior_thread_hint(engagement_type: str, template_family_id: str, variant_index: int) -> list[dict[str, str]]:
    if engagement_type == "follow_up":
        return [
            {"speaker": "sales_agent", "message": "Shared a short note tied to a recent public signal last week."},
            {"speaker": "prospect", "message": f"Follow-up pass {variant_index + 1}; keep the next note specific and low-pressure."},
        ]
    if engagement_type == "objection_handling":
        objection = "Budget is tight this quarter." if "budget" in template_family_id else "We need clearer differentiation before we continue."
        return [
            {"speaker": "sales_agent", "message": "Sent a grounded note tied to a public hiring or GTM signal."},
            {"speaker": "prospect", "message": objection},
        ]
    if engagement_type == "discovery_call":
        return [
            {"speaker": "internal_note", "message": f"Qualification pass {variant_index + 1}: decide whether to automate a discovery-call invitation or route to review."}
        ]
    return []


def validate_task_structure(task: dict[str, Any]) -> None:
    required_top = [
        "task_id",
        "split",
        "source_mode",
        "difficulty",
        "task_type",
        "failure_mode_tags",
        "input",
        "expected_behavior",
        "rubric",
        "metadata",
    ]
    for key in required_top:
        if key not in task:
            raise ValueError(f"Missing top-level key: {key}")
    if not task["metadata"].get("template_family_id"):
        raise ValueError("Missing metadata.template_family_id")


def is_concrete_evidence(item: str) -> bool:
    text = normalize_text(item).lower()
    if not text or len(text.split()) < 3:
        return False
    return any(char.isdigit() for char in text) or any(hint in text for hint in CONCRETE_SIGNAL_HINTS)


def has_banned_phrase(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(phrase in lowered for phrase in GENERIC_BANNED_PHRASES)


def missing_required_cta(task: dict[str, Any]) -> bool:
    if not task["expected_behavior"]["cta_required"]:
        return False
    body = task["expected_behavior"]["expected_output"]["email_body"].lower()
    return not ("calendar" in body and any(hint in body for hint in REQUIRED_CTA_HINTS))


def repair_required_cta(task: dict[str, Any]) -> bool:
    if not task["expected_behavior"]["cta_required"]:
        return False
    if not missing_required_cta(task):
        return False
    company_name = task["input"]["company_name"]
    body = normalize_text(task["expected_behavior"]["expected_output"]["email_body"])
    repair = (
        f" If helpful, I can send a calendar link and hold a 15-minute slot next week "
        f"to compare whether this is relevant for {company_name}."
    )
    task["expected_behavior"]["expected_output"]["email_body"] = f"{body}{repair}".strip()
    return not missing_required_cta(task)


def sanitize_review_output(task: dict[str, Any]) -> bool:
    if task["expected_behavior"]["action"] != "review":
        return False
    body = normalize_text(task["expected_behavior"]["expected_output"]["email_body"])
    original = body
    replacements = [
        ("competitor gap", "positioning gap"),
        ("competitor", "positioning"),
        ("bench fit", "fit"),
        ("verified bench fit", "verified fit"),
    ]
    lowered = body.lower()
    for src, dst in replacements:
        if src in lowered:
            pattern = re.compile(re.escape(src), re.IGNORECASE)
            body = pattern.sub(dst, body)
            lowered = body.lower()
    if "manual review" not in lowered:
        body = f"Manual review required before outreach. {body}".strip()
    task["expected_behavior"]["expected_output"]["email_body"] = body
    return body != original


def unsupported_claims(task: dict[str, Any]) -> list[str]:
    allowed_corpus = " ".join(
        [
            task["input"]["company_context"],
            task["input"]["hiring_signal"],
            " ".join(task["input"]["evidence"]),
            " ".join(item["message"] for item in task["input"].get("prior_thread", [])),
        ]
    ).lower()
    body = task["expected_behavior"]["expected_output"]["email_body"].lower()
    return [term for term in CLAIM_CHECK_TERMS if term in body and term not in allowed_corpus]


def generic_company_name(task: dict[str, Any]) -> bool:
    name = task["input"]["company_name"].strip().lower()
    blocked = {
        "company",
        "sample company",
        "example company",
        "test company",
        "unknown company",
        "acme",
    }
    return not name or name in blocked or len(name) < 3


def similarity_text(task: dict[str, Any]) -> str:
    return normalize_text(
        f"{task['input']['hiring_signal']} {task['expected_behavior']['expected_output']['email_subject']} {task['expected_behavior']['expected_output']['email_body']}"
    ).lower()


def is_near_duplicate(task: dict[str, Any], accepted_tasks: list[dict[str, Any]], threshold: float = 0.88) -> bool:
    template_family_id = task["metadata"]["template_family_id"]
    candidate = similarity_text(task)
    for existing in accepted_tasks:
        if existing["metadata"]["template_family_id"] != template_family_id:
            continue
        if SequenceMatcher(None, candidate, similarity_text(existing)).ratio() >= threshold:
            return True
    return False


def reject_reason(task: dict[str, Any], accepted_tasks: list[dict[str, Any]], seen_pairs: set[tuple[str, str]]) -> str | None:
    evidence = task["input"]["evidence"]
    if generic_company_name(task):
        return "generic_company_name"
    if len(evidence) < 2:
        return "fewer_than_two_evidence_items"
    if sum(1 for item in evidence if is_concrete_evidence(item)) < 2:
        return "evidence_too_vague"
    output = task["expected_behavior"]["expected_output"]
    if has_banned_phrase(output["email_subject"]) or has_banned_phrase(output["email_body"]):
        return "banned_generic_phrase"
    if missing_required_cta(task):
        return "missing_required_cta"
    pair = (task["metadata"]["template_family_id"], task["input"]["company_name"].lower())
    if pair in seen_pairs:
        return "duplicate_company_template_pair"
    unsupported = unsupported_claims(task)
    if unsupported:
        return f"unsupported_claims:{','.join(sorted(unsupported))}"
    if is_near_duplicate(task, accepted_tasks):
        return "near_duplicate_same_template"
    return None


def extract_usage(response_json: dict[str, Any]) -> dict[str, int]:
    usage = response_json.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def extract_cost(response_json: dict[str, Any]) -> float:
    usage = response_json.get("usage") or {}
    for key in ("cost", "total_cost"):
        if key in usage and usage[key] is not None:
            return float(usage[key])
    for key in ("cost", "total_cost"):
        if key in response_json and response_json[key] is not None:
            return float(response_json[key])
    return 0.0


def append_cost_log(path: Path, model: str, usage: dict[str, int], cost_usd: float, accepted: int, rejected: int, notes: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                now_iso(),
                "dataset_llm_synthesis",
                "LLM task generation",
                model,
                f"prompt={usage['prompt_tokens']};completion={usage['completion_tokens']};total={usage['total_tokens']}",
                f"{cost_usd:.6f}",
                f"accepted={accepted};rejected={rejected};{notes}",
            ]
        )


def ordered_base_specs(scenario_specs: list[dict[str, Any]], priority_template_order: list[str]) -> list[dict[str, Any]]:
    by_id = {spec["template_family_id"]: spec for spec in scenario_specs}
    ordered = [by_id[template_id] for template_id in priority_template_order if template_id in by_id]
    missing = [spec for spec in scenario_specs if spec["template_family_id"] not in priority_template_order]
    return ordered + missing


def build_generation_plan(total_count: int, scenario_specs: list[dict[str, Any]], priority_template_order: list[str]) -> list[dict[str, Any]]:
    base_specs = ordered_base_specs(scenario_specs, priority_template_order)
    if total_count <= 0:
        raise ValueError("Count must be positive.")
    selected_specs = base_specs[: min(total_count, len(base_specs))]
    plan: list[dict[str, Any]] = []
    full_rounds = total_count // len(selected_specs)
    remainder = total_count % len(selected_specs)
    for repeat_index in range(full_rounds):
        for base_spec in selected_specs:
            scenario = apply_overrides(base_spec, variation_overrides(base_spec, repeat_index))
            scenario["variant_index"] = repeat_index
            scenario["base_template_family_id"] = base_spec["template_family_id"]
            plan.append(scenario)
    if remainder:
        repeat_index = full_rounds
        for base_spec in selected_specs[:remainder]:
            scenario = apply_overrides(base_spec, variation_overrides(base_spec, repeat_index))
            scenario["variant_index"] = repeat_index
            scenario["base_template_family_id"] = base_spec["template_family_id"]
            plan.append(scenario)
    return plan


def counter_dict(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items()))


def coverage_from_tasks(tasks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    engagement_types = Counter()
    actions = Counter()
    difficulties = Counter()
    failure_tags = Counter()
    template_families = Counter()
    for task in tasks:
        engagement_types[task["input"]["engagement_type"]] += 1
        actions[task["expected_behavior"]["action"]] += 1
        difficulties[task["difficulty"]] += 1
        template_families[task["metadata"]["template_family_id"]] += 1
        for tag in task["failure_mode_tags"]:
            failure_tags[tag] += 1
    return {
        "engagement_type": counter_dict(engagement_types),
        "action": counter_dict(actions),
        "difficulty": counter_dict(difficulties),
        "failure_mode_tags": counter_dict(failure_tags),
        "template_family_id": counter_dict(template_families),
    }


def coverage_from_plan(plan: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    engagement_types = Counter()
    actions = Counter()
    difficulties = Counter()
    failure_tags = Counter()
    template_families = Counter()
    for scenario in plan:
        engagement_types[scenario["engagement_type"]] += 1
        actions[scenario["action"]] += 1
        difficulties[scenario["difficulty"]] += 1
        template_families[scenario["template_family_id"]] += 1
        for tag in scenario["failure_mode_tags"]:
            failure_tags[tag] += 1
    return {
        "engagement_type": counter_dict(engagement_types),
        "action": counter_dict(actions),
        "difficulty": counter_dict(difficulties),
        "failure_mode_tags": counter_dict(failure_tags),
        "template_family_id": counter_dict(template_families),
    }


def summarize(tasks: list[dict[str, Any]], planned_coverage: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "task_count": len(tasks),
        "grouping_fields": {"split_family_field": "template_family_id"},
        "planned_coverage": planned_coverage,
        "actual_coverage": coverage_from_tasks(tasks),
        "generation_models": counter_dict(Counter(task["metadata"]["generation_model"] for task in tasks)),
    }


def print_coverage(label: str, coverage: dict[str, dict[str, int]]) -> None:
    print(json.dumps({label: coverage}, indent=2))
