import argparse
import json
import re
from typing import Any


def load_task(task_path: str) -> dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_text(candidate_path: str) -> str:
    with open(candidate_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "at",
        "by", "if", "it", "is", "are", "be", "as", "that", "this", "your", "you",
        "we", "our", "i", "can", "will", "when", "while", "into", "from"
    }
    tokens = re.findall(r"[a-z0-9]+", normalize(text))
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def contains_any(text: str, phrases: list[str]) -> list[str]:
    normalized = normalize(text)
    return [phrase for phrase in phrases if normalize(phrase) in normalized]


def score_signal_grounding(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    evidence = task["input"].get("evidence", [])
    candidate_tokens = content_tokens(candidate)
    exact_matches = []
    fuzzy_matches = []

    for point in evidence:
        point_norm = normalize(point)
        if point_norm and point_norm in normalize(candidate):
            exact_matches.append(point)
            continue

        point_tokens = content_tokens(point)
        if not point_tokens:
            continue
        overlap_ratio = len(candidate_tokens.intersection(point_tokens)) / len(point_tokens)
        if overlap_ratio >= 0.4:
            fuzzy_matches.append(point)

    total_matches = len(exact_matches) + len(fuzzy_matches)
    details = {
        "exact_grounding_matches": exact_matches,
        "fuzzy_grounding_matches": fuzzy_matches
    }
    if total_matches >= 2:
        return 5, details
    if total_matches == 1:
        return 3, details
    return 0, details


def score_tone(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    tone_markers = task["input"].get("tone_markers", [])
    candidate_norm = normalize(candidate)
    hits = 0

    marker_keywords = {
        "concise": ["brief", "quick", "short", "concise"],
        "grounded": ["noticed", "saw", "based on", "from your", "signal"],
        "respectful": ["if helpful", "open to", "worth a look", "happy to", "no worries"],
        "specific": ["team", "hiring", "bench", "ramp", "pipeline"],
        "low-hype": ["not a silver bullet", "no pressure", "practical", "measured"]
    }

    for marker in tone_markers:
        options = marker_keywords.get(marker.lower(), [marker])
        if any(option in candidate_norm for option in options):
            hits += 1

    if hits >= 4:
        return 5, {"tone_marker_hits": hits, "tone_markers": tone_markers}
    if hits >= 2:
        return 3, {"tone_marker_hits": hits, "tone_markers": tone_markers}
    return 1, {"tone_marker_hits": hits, "tone_markers": tone_markers}


def score_banned_phrases(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    banned = task["input"].get("banned_phrases", [])
    matched = contains_any(candidate, banned)
    if not matched:
        return 5, {"matched_banned_phrases": []}
    if len(matched) == 1:
        return 2, {"matched_banned_phrases": matched}
    return 0, {"matched_banned_phrases": matched}


def score_calendar_cta(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    cta_required = task["expected_behavior"].get("cta_required", True)
    cta_patterns = [
        r"calendar",
        r"book time",
        r"pick a time",
        r"schedule",
        r"15[- ]?min",
        r"20[- ]?min",
        r"next week",
        r"link"
    ]
    hits = [pattern for pattern in cta_patterns if re.search(pattern, normalize(candidate))]

    if not cta_required:
        if hits:
            return 5, {"cta_required": cta_required, "cta_evidence": hits}
        return 5, {"cta_required": cta_required, "cta_evidence": []}

    if len(hits) >= 2:
        return 5, {"cta_required": cta_required, "cta_evidence": hits}
    if len(hits) == 1:
        return 3, {"cta_required": cta_required, "cta_evidence": hits}
    return 0, {"cta_required": cta_required, "cta_evidence": []}


def score_hallucination(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    disallowed = task["input"].get("disallowed_claims", [])
    must_avoid = task["expected_behavior"].get("must_avoid", [])
    bad_matches = contains_any(candidate, disallowed + must_avoid)
    unsupported_patterns = [
        r"\bguarantee\b",
        r"\b100%\b",
        r"\bdouble your\b",
        r"\btripled\b",
        r"\bwe already know\b",
        r"\bdefinitely\b"
    ]
    pattern_hits = [pattern for pattern in unsupported_patterns if re.search(pattern, normalize(candidate))]
    if not bad_matches and not pattern_hits:
        return 5, {"matched_unsupported_claims": [], "pattern_hits": []}
    if len(bad_matches) + len(pattern_hits) == 1:
        return 2, {"matched_unsupported_claims": bad_matches, "pattern_hits": pattern_hits}
    return 0, {"matched_unsupported_claims": bad_matches, "pattern_hits": pattern_hits}


def score_segment_fit(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    segment_terms = []
    if task["input"].get("segment"):
        segment_terms.append(task["input"]["segment"])
    if task["input"].get("company_context"):
        segment_terms.append(task["input"]["company_context"])

    count = 0
    matched = []
    candidate_norm = normalize(candidate)
    for term in segment_terms:
        if normalize(term) in candidate_norm:
            matched.append(term)
            count += 1

    if count >= 2:
        return 5, {"matched_segment_terms": matched}
    if count == 1:
        return 3, {"matched_segment_terms": matched}
    return 1, {"matched_segment_terms": matched}


def score_send_decision(task: dict[str, Any], candidate: str) -> tuple[int, dict[str, Any]]:
    expected_action = task["expected_behavior"].get("action", "send")
    candidate_norm = normalize(candidate)
    abstain_patterns = [
        "not enough signal",
        "insufficient signal",
        "should not send",
        "hold off",
        "route to review",
        "manual review",
        "do not send",
        "need more evidence",
    ]
    review_patterns = [
        "manual review",
        "route to review",
        "needs review",
        "review before send",
        "human review",
    ]
    send_patterns = [
        "open to",
        "15-minute",
        "15 minute",
        "calendar",
        "quick chat",
        "next week",
        "send a calendar link",
    ]

    abstain_hits = [pattern for pattern in abstain_patterns if pattern in candidate_norm]
    review_hits = [pattern for pattern in review_patterns if pattern in candidate_norm]
    send_hits = [pattern for pattern in send_patterns if pattern in candidate_norm]

    if expected_action == "abstain":
        if abstain_hits:
            return 5, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        if send_hits:
            return 0, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        return 2, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}

    if expected_action == "review":
        if review_hits:
            return 5, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        if send_hits:
            return 0, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        return 2, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}

    if expected_action in {"send", "exploratory_send"}:
        if send_hits:
            return 5, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        if abstain_hits or review_hits:
            return 0, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}
        return 2, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}

    return 2, {"expected_action": expected_action, "matched_abstain_signals": abstain_hits, "matched_review_signals": review_hits, "matched_send_signals": send_hits}


def compute_overall(task: dict[str, Any], component_scores: dict[str, int]) -> float:
    rubric = task["rubric"]
    weights = {
        "signal_grounding": rubric.get("signal_grounding", 0.0),
        "hallucination_unsupported_claims": rubric.get("hallucination_control", 0.0),
        "tenacious_tone_style": rubric.get("tone_style", 0.0),
        "calendar_cta_presence": rubric.get("cta", 0.0),
        "send_decision_confidence_match": rubric.get("decision_correctness", 0.0),
        "segment_fit": rubric.get("segment_fit", 0.0),
        "banned_phrases": rubric.get("banned_phrase_control", 0.0),
    }
    total = 0.0
    for key, weight in weights.items():
        total += component_scores[key] * weight
    return round(total, 2)


def evaluate(task: dict[str, Any], candidate: str) -> dict[str, Any]:
    signal_score, signal_info = score_signal_grounding(task, candidate)
    tone_score, tone_info = score_tone(task, candidate)
    banned_score, banned_info = score_banned_phrases(task, candidate)
    cta_score, cta_info = score_calendar_cta(task, candidate)
    hallucination_score, hallucination_info = score_hallucination(task, candidate)
    segment_score, segment_info = score_segment_fit(task, candidate)
    decision_score, decision_info = score_send_decision(task, candidate)

    component_scores = {
        "signal_grounding": signal_score,
        "tenacious_tone_style": tone_score,
        "banned_phrases": banned_score,
        "calendar_cta_presence": cta_score,
        "hallucination_unsupported_claims": hallucination_score,
        "segment_fit": segment_score,
        "send_decision_confidence_match": decision_score,
    }
    overall = compute_overall(task, component_scores)

    return {
        "task_id": task["task_id"],
        "component_scores": component_scores,
        "overall_score": overall,
        "details": {
            "signal_grounding": signal_info,
            "tenacious_tone_style": tone_info,
            "banned_phrases": banned_info,
            "calendar_cta_presence": cta_info,
            "hallucination_unsupported_claims": hallucination_info,
            "segment_fit": segment_info,
            "send_decision_confidence_match": decision_info,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Tenacious-Bench task against a candidate output.")
    parser.add_argument("task_json", help="Path to a task JSON file.")
    parser.add_argument("candidate_output", nargs="?", help="Path to a candidate output text file.")
    parser.add_argument("--use-ground-truth", action="store_true", help="Score the task's expected_output instead of a file.")
    args = parser.parse_args()

    task = load_task(args.task_json)

    if args.use_ground_truth:
        expected_output = task["expected_behavior"]["expected_output"]
        candidate = f'{expected_output["email_subject"]}\n\n{expected_output["email_body"]}'
    else:
        if not args.candidate_output:
            raise SystemExit("Provide candidate_output or use --use-ground-truth")
        candidate = load_candidate_text(args.candidate_output)

    result = evaluate(task, candidate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
