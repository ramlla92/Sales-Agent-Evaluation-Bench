import argparse
import json
from collections import Counter
from pathlib import Path
import re
from hashlib import md5


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_INPUT = ROOT / "tenacious_bench_v0.1" / "final_dataset" / "train" / "tenacious_bench_train_all_sources_100.jsonl"
DEFAULT_DEV_INPUT = ROOT / "tenacious_bench_v0.1" / "final_dataset" / "dev" / "tenacious_bench_dev_all_sources_60.jsonl"
DEFAULT_TRAIN_OUTPUT = ROOT / "training_data" / "path_b_train_preferences.jsonl"
DEFAULT_DEV_OUTPUT = ROOT / "training_data" / "path_b_dev_preferences.jsonl"
DEFAULT_SUMMARY_OUTPUT = ROOT / "training_data" / "path_b_preferences_summary.json"

STYLE_GUIDE_BANNED_PHRASES = [
    "world-class",
    "top talent",
    "a-players",
    "rockstar",
    "ninja",
    "wizard",
    "skyrocket",
    "supercharge",
    "10x",
    "i hope this email finds you well",
    "just following up",
    "circling back",
    "quick question",
    "quick chat",
    "synergize",
    "synergy",
    "leverage",
    "ecosystem",
    "game-changer",
    "disruptor",
    "paradigm shift",
    "per my last email",
    "bench",
]

PERSPECTIVE_DRIFT_MARKERS = [
    "we're excited to",
    "we are excited to",
    "we recently",
    "our team",
    "our product",
    "join our team",
    "apply now",
    "stay tuned",
    "we would like to",
    "we have recently",
    "we're actively hiring",
    "we are actively hiring",
]

TENACIOUS_TONE_MARKERS = [
    "noticed",
    "saw",
    "came across",
    "from the outside",
    "curious",
    "would it be useful",
    "would a short conversation",
    "compare notes",
    "next week",
    "open to",
    "worth a conversation",
    "did not want to overread",
    "if this is",
    "if useful",
    "if it is relevant",
]

EXTERNAL_SENDER_MARKERS = [
    "you're",
    "you are",
    "your",
    "would it be useful",
    "would a short conversation",
    "if this is",
    "if useful",
    "if it is relevant",
    "compare notes",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def render_input_block(task: dict) -> str:
    input_data = task["input"]
    expected = task.get("expected_behavior", {})
    engagement_type = str(input_data.get("engagement_type", "") or "").strip()
    company_context = str(input_data.get("company_context", "") or "").strip()
    scenario = " | ".join(part for part in [engagement_type, company_context] if part)
    lines = [
        f"Company: {input_data.get('company_name', '')}",
        f"Scenario: {scenario}",
        f"Signal: {input_data.get('hiring_signal', '')}",
        "Evidence:",
    ]
    evidence = input_data.get("evidence", [])
    if evidence:
        lines.extend(f"- {item}" for item in evidence)
    else:
        lines.append("- None")

    prior_thread = input_data.get("prior_thread", [])
    if prior_thread:
        lines.extend(["", "Thread:"])
        for item in prior_thread:
            speaker = item.get("speaker") or item.get("sender") or "unknown"
            message = item.get("message") or item.get("body") or ""
            lines.append(f"- {speaker}: {message}")

    lines.extend(
        [
            "",
            f"Target action: {expected.get('action', '')}",
            "Return the safest grounded final response.",
        ]
    )
    return "\n".join(lines).strip()


def render_output(output: dict, action: str) -> str:
    subject = (output or {}).get("email_subject", "") or ""
    body = (output or {}).get("email_body", "") or ""
    parts = [f"action: {action}"]
    if subject:
        parts.append(f"subject: {subject}")
    parts.append(f"body: {body}")
    return "\n".join(parts).strip()


def first_evidence_line(task: dict) -> str:
    evidence = task.get("input", {}).get("evidence", [])
    if evidence:
        return str(evidence[0]).strip()
    hiring_signal = task.get("input", {}).get("hiring_signal", "")
    return str(hiring_signal).strip() or "the available public signal"


def humanize_structured_signal(text: str, company: str) -> str:
    raw = re.sub(r"\s+", " ", text or "").strip(" .")
    lowered = raw.lower()
    if not raw:
        return ""
    if "no local job-post snapshot matched this company" in lowered:
        return f"there is not a clear hiring signal for {company} from the outside"
    if lowered.startswith("job_post_velocity:"):
        if "insufficient_signal" in lowered:
            return f"the hiring picture for {company} is still inconclusive from the outside"
        return f"the public hiring picture for {company} looks active"
    if lowered.startswith("crunchbase_funding:"):
        amount_match = re.search(r"USD\s*([\d,]+)", raw, flags=re.IGNORECASE)
        if amount_match:
            return f"Crunchbase lists a funding event for {company} of USD {amount_match.group(1)}"
        if "funding rounds" in lowered:
            return f"Crunchbase lists at least one funding event for {company}"
        return f"Crunchbase lists a funding event for {company}"
    if lowered.startswith("crunchbase_firmographics:"):
        return f"Crunchbase includes basic company profile data for {company}"
    if lowered.startswith("tech_stack:"):
        return f"public web signals show a visible tech stack for {company}"
    if lowered.startswith("layoff_event:"):
        return f"the public brief includes a workforce change signal for {company}"
    if lowered.startswith("leadership_change:"):
        return f"the public brief includes a leadership change at {company}"
    if lowered.startswith("ai_maturity_hint:"):
        return f"the public brief hints at possible AI relevance for {company}"
    return ""


def evidence_phrase(task: dict) -> str:
    company = task.get("input", {}).get("company_name", "the company")
    text = humanize_structured_signal(first_evidence_line(task), company) or first_evidence_line(task)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "the available public signal"


def banned_phrases_for_task(task: dict) -> list[str]:
    return [str(item).strip() for item in task.get("input", {}).get("banned_phrases", []) if str(item).strip()]


def contains_banned_phrase(text: str, banned_phrases: list[str]) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in banned_phrases if phrase.lower() in lowered]


def has_bad_trace_markers(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "hi john doe",
        "would love to connect",
        "**subject:**",
        "crunchbase_funding:",
        "crunchbase_firmographics:",
        "tech_stack:",
        "layoff_event:",
        "job_post_velocity:",
        "no local job-post snapshot matched this company",
        "as you scale",
        "must be feeling the pain",
        "best regards",
        "hi there,",
        "[name]",
    ]
    return any(marker in lowered for marker in markers)


def style_guide_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in STYLE_GUIDE_BANNED_PHRASES if phrase in lowered]


def perspective_drift_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [marker for marker in PERSPECTIVE_DRIFT_MARKERS if marker in lowered]


def has_weak_tenacious_tone(text: str, action: str) -> bool:
    lowered = text.lower()
    if action in {"abstain", "review"}:
        if any(marker in lowered for marker in ["let us know", "thank you for considering", "stay tuned", "we appreciate your interest"]):
            return True
        if len(lowered.split()) > 28:
            return True
        return False
    marker_hits = sum(1 for marker in TENACIOUS_TONE_MARKERS if marker in lowered)
    if marker_hits == 0:
        return True
    if "!" in text:
        return True
    if any(marker in lowered for marker in ["congrats on", "excited to share", "apply now", "looking forward to continuing", "happy to help accelerate", "stay tuned"]):
        return True
    if not any(marker in lowered for marker in EXTERNAL_SENDER_MARKERS):
        return True
    return False


def is_company_side_output(text: str, company: str) -> bool:
    lowered = text.lower()
    company_lower = (company or "").lower()
    if any(marker in lowered for marker in PERSPECTIVE_DRIFT_MARKERS):
        return True
    if company_lower and any(
        phrase in lowered
        for phrase in [
            f"{company_lower} has recently received",
            f"{company_lower} recently received",
            f"{company_lower} has recently",
            f"{company_lower}'s recent funding news",
        ]
    ):
        return True
    if "body: \"" in lowered and ("we " in lowered or "our " in lowered):
        return True
    return False


def normalize_snippet(text: str, limit: int = 90) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip(" .")
    return clean[:limit].rstrip(" ,;:") if clean else ""


def sanitize_output_text(task: dict, text: str) -> str:
    company = task.get("input", {}).get("company_name", "the company")
    sanitized = text
    candidates = [task.get("input", {}).get("hiring_signal", "")]
    candidates.extend(task.get("input", {}).get("evidence", []))
    candidates.append(first_evidence_line(task))

    for raw in candidates:
        raw_text = str(raw or "").strip()
        humanized = humanize_structured_signal(raw_text, company)
        if raw_text and humanized:
            sanitized = sanitized.replace(raw_text, humanized)

    sanitized = sanitized.replace("I noticed There is", "I noticed there is")
    sanitized = sanitized.replace("I came across There is", "I came across there is")
    sanitized = sanitized.replace("I would still treat There is", "I would still treat there is")

    return sanitized


def stable_index(task: dict, size: int) -> int:
    key = str(task.get("task_id") or task.get("id") or "")
    digest = md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def pick(task: dict, options: list[str]) -> str:
    return options[stable_index(task, len(options))]


def sentence_case(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip(" .")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def clause_case(text: str) -> str:
    cleaned = sentence_case(text)
    if not cleaned:
        return ""
    lowercase_leads = (
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
        "these ",
        "those ",
        "there ",
        "from ",
        "public ",
        "job ",
        "hiring ",
        "posted ",
        "recent ",
        "moderate ",
        "possible ",
        "only ",
        "no ",
    )
    if not cleaned.lower().startswith(lowercase_leads):
        return cleaned
    return cleaned[0].lower() + cleaned[1:]


def strip_structured_prefix(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip(" .")
    cleaned = re.sub(r"^[a-z_]+:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
    return cleaned.strip(" .")


def personalize_company_lead(text: str, company: str) -> str:
    structured = humanize_structured_signal(text, company)
    if structured:
        return structured
    cleaned = strip_structured_prefix(text)
    if cleaned.lower().startswith("the company "):
        return f"{company} {cleaned[len('the company '):]}"
    return cleaned


def summarize_signal(task: dict) -> str:
    company = task.get("input", {}).get("company_name", "The company")
    hiring_signal = personalize_company_lead(task.get("input", {}).get("hiring_signal", ""), company)
    evidence_items = [personalize_company_lead(item, company) for item in task.get("input", {}).get("evidence", []) if strip_structured_prefix(item)]

    candidates = []
    if hiring_signal and hiring_signal.lower() not in {"none", "public evidence requires review"}:
        candidates.append(hiring_signal)
    candidates.extend(evidence_items[:2])

    for candidate in candidates:
        lowered = candidate.lower()
        if any(bad in lowered for bad in ["job_post_velocity", "crunchbase_firmographics", "tech_stack", "layoff_event"]):
            continue
        return sentence_case(candidate)

    context = strip_structured_prefix(task.get("input", {}).get("company_context", ""))
    if context:
        return sentence_case(context)

    company = task.get("input", {}).get("company_name", "the company")
    return f"{company} shows a public signal worth a careful look"


def exploratory_signal_line(task: dict) -> str:
    company = task.get("input", {}).get("company_name", "The company")
    evidence_items = [personalize_company_lead(item, company) for item in task.get("input", {}).get("evidence", []) if strip_structured_prefix(item)]
    hiring_signal = personalize_company_lead(task.get("input", {}).get("hiring_signal", ""), company)
    context = personalize_company_lead(task.get("input", {}).get("company_context", ""), company)

    meaningful = []
    for item in [hiring_signal, *evidence_items, context]:
        lowered = item.lower()
        if not item or lowered in {"none", "public evidence requires review"}:
            continue
        if any(bad in lowered for bad in ["job_post_velocity", "crunchbase_firmographics", "tech_stack", "layoff_event"]):
            continue
        meaningful.append(item)

    if meaningful:
        return sentence_case(meaningful[0])

    if "ai angle exploratory" in hiring_signal.lower() or "ai angle exploratory" in " ".join(evidence_items).lower():
        return "The public brief hints at adjacent AI relevance, but not enough for a confident claim"

    return "The available public brief is still thin"


def preserved_output_needs_refresh(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "best regards",
        "hi there,",
        "[name]",
        "calendar link",
        "pick a time on my calendar",
        "as you scale",
        "must be feeling the pain",
        "clearly indicate",
        "that reads like a real hiring signal",
        "that combination points to active market expansion",
        "this increase suggests",
    ]
    return any(marker in lowered for marker in markers)


def compact_company_name(company: str, max_len: int = 36) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", company or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip(" ,;:")


def chosen_subject(task: dict, action: str) -> str:
    company = compact_company_name(task.get("input", {}).get("company_name", "the company"))
    engagement_type = str(task.get("input", {}).get("engagement_type", "") or "").lower()
    if action == "send":
        if engagement_type == "follow_up":
            return f"Following up on {company}'s recent signal"
        return f"Context: {company} hiring signal"
    if action == "exploratory_send":
        return f"Question: {company} signal"
    return ""


def synthesize_safe_output(task: dict, action: str) -> dict:
    company = task.get("input", {}).get("company_name", "the company")
    signal = summarize_signal(task)
    evidence = sentence_case(evidence_phrase(task))
    engagement_type = str(task.get("input", {}).get("engagement_type", "") or "").lower()
    confidence = str(task.get("input", {}).get("signal_strength", "") or "").lower()
    greeting = "Hi [First Name],"
    has_funding = "funding" in " ".join(str(item).lower() for item in task.get("input", {}).get("evidence", []))
    has_leadership = "leadership" in " ".join(str(item).lower() for item in task.get("input", {}).get("evidence", []))
    soft_read = has_funding or has_leadership

    if action == "abstain":
        body = pick(
            task,
            [
                f"Insufficient signal to justify outreach for {company}. The current evidence is too thin to act on safely.",
                f"Not enough evidence to send outreach for {company}. This record should stay out of automation for now.",
                f"Recommend holding off on outreach to {company}. The available signal does not clear the bar for a grounded send.",
                f"Low-confidence lead for {company}. The public evidence is too limited for an automated message.",
                f"No clear hiring trigger is visible for {company}. I would hold this back rather than force outreach.",
                f"The signals around {company} are still too weak to justify outreach at this stage.",
                f"There is not enough evidence of a timely need at {company}. Better to defer than automate a weak send.",
                f"{company} does not clear the evidence bar for automated outreach yet.",
            ],
        )
        return {
            "email_subject": "",
            "email_body": body,
        }
    if action == "review":
        body = pick(
            task,
            [
                f"Manual review is required for {company} because the current evidence is not safe enough to automate confidently.",
                f"Route {company} to manual review. The signal may be relevant, but the angle is not grounded enough for automation yet.",
                f"Hold this record for human review. The evidence for {company} is still too ambiguous for a clean automated decision.",
                f"Review before any outreach to {company}. The current brief leaves too much room for unsupported positioning.",
                f"Defer {company} to human review. The available facts do not support a clean automated angle yet.",
                f"A person should review {company} before any send decision. The current brief is directionally interesting but not settled.",
                f"The current signal for {company} sits in a gray zone. Human review is the safer next step.",
                f"{company} needs a reviewer to validate the angle before automation goes any further.",
            ],
        )
        return {
            "email_subject": "",
            "email_body": body,
        }

    if action == "exploratory_send":
        subject = chosen_subject(task, action)
        signal_line = exploratory_signal_line(task)
        if engagement_type == "follow_up":
            body = pick(
                task,
                [
                    (
                        f"{greeting}\n"
                        f"Following up on my earlier note and keeping this specific.\n"
                        f"I noticed {clause_case(signal_line)}.\n"
                        f"From the outside, that could point to something broader, but I did not want to assume too much. "
                        f"Would it be useful to compare notes for 10 to 15 minutes next week?"
                    ),
                    (
                        f"{greeting}\n"
                        f"Following up on the note I sent earlier.\n"
                        f"The public signal I saw was {clause_case(signal_line)}.\n"
                        f"I am treating it as directional rather than definitive. If it is relevant on your side, "
                        f"would a short conversation next week be useful?"
                    ),
                    (
                        f"{greeting}\n"
                        f"When I came back to your earlier note, {clause_case(signal_line)} stood out.\n"
                        f"I did not want to read too much into a partial signal, but it seemed worth a direct question. "
                        f"Would it be useful to compare notes briefly next week?"
                    ),
                ],
            )
        elif confidence in {"weak", "low", "medium", "moderate"}:
            body = pick(
                task,
                [
                    (
                        f"{greeting}\n"
                        f"I noticed {clause_case(signal_line)}.\n"
                        f"I cannot tell from the outside whether that reflects a broader initiative or a narrower update. "
                        f"If this is on your radar, would it be useful to compare notes for 10 to 15 minutes next week?"
                    ),
                    (
                        f"{greeting}\n"
                        f"I came across {clause_case(signal_line)}.\n"
                        f"It looked specific enough to ask about, but not strong enough to treat as a firm expansion story. "
                        f"Would a brief conversation next week be useful?"
                    ),
                    (
                        f"{greeting}\n"
                        f"From the outside, {clause_case(signal_line)}.\n"
                        f"I wanted to keep the note exploratory rather than read too much into a partial signal. "
                        f"Would it make sense to compare notes for 10 to 15 minutes next week?"
                    ),
                    (
                        f"{greeting}\n"
                        f"I noticed {clause_case(signal_line)}.\n"
                        f"That may or may not reflect a broader initiative, so I wanted to ask rather than assume. "
                        f"If useful, I can make time for a short conversation next week."
                    ),
                ],
            )
        else:
            body = pick(
                task,
                [
                    (
                        f"{greeting}\n"
                        f"I noticed {clause_case(signal_line)}.\n"
                        f"The outside picture is still incomplete, so I did not want to overread the signal. "
                        f"Would a short conversation next week be useful?"
                    ),
                    (
                        f"{greeting}\n"
                        f"I came across {clause_case(signal_line)}.\n"
                        f"I still see this as exploratory rather than definitive. "
                        f"If it is relevant, I can make time next week."
                    ),
                    (
                        f"{greeting}\n"
                        f"I noticed {clause_case(signal_line)}.\n"
                        f"It looked specific enough to ask about, but not strong enough to turn into a bigger claim. "
                        f"Would a short conversation next week be useful?"
                    ),
                ],
            )
        return {
            "email_subject": subject,
            "email_body": body,
        }

    subject = chosen_subject(task, action)
    if engagement_type == "follow_up":
        body = pick(
            task,
            [
                (
                    f"{greeting}\n"
                    f"Following up on my earlier note and keeping this specific.\n"
                    f"I noticed {signal}.\n"
                    f"{'That could be tied to a current priority.' if soft_read else 'That looks like a concrete public signal tied to current priorities.'} "
                    f"Would 15 minutes next week be useful?"
                ),
                (
                    f"{greeting}\n"
                    f"Following up on the note I sent last week.\n"
                    f"I wanted to come back to {clause_case(signal)}.\n"
                    f"{'It may be relevant, although I do not want to overread it.' if soft_read else 'That timing looks real rather than speculative.'} "
                    f"If useful, I can make time next week."
                ),
            ],
        )
    else:
        body = pick(
            task,
            [
                (
                    f"{greeting}\n"
                    f"I noticed {signal}.\n"
                    f"{'That could be relevant to a current priority.' if soft_read else 'That looks like a concrete public signal tied to current priorities.'} "
                    f"Would 15 minutes next week be useful?"
                ),
                (
                    f"{greeting}\n"
                    f"I came across {clause_case(signal)}.\n"
                    f"{'It may point to something worth discussing, even if the outside picture is still partial.' if soft_read else 'It looked specific enough to merit a direct note.'} "
                    f"Would a short conversation next week be worth it?"
                ),
                (
                    f"{greeting}\n"
                    f"From the outside, {clause_case(signal)}.\n"
                    f"{'I may be reading only part of the story, but it seemed worth asking about.' if soft_read else 'That looked specific enough to justify a direct note.'} "
                    f"If this is an active priority, I can make time next week."
                ),
                (
                    f"{greeting}\n"
                    f"Saw {clause_case(signal)}.\n"
                    f"{'I may only be seeing part of the picture, but it seemed worth a direct question.' if soft_read else 'That often creates pressure on the team even when the outside signal is only one piece of the story.'} "
                    f"Would it be useful to compare notes next week?"
                ),
            ],
        )
    return {
        "email_subject": subject,
        "email_body": body,
    }


def synthesize_rejected_output(task: dict, action: str) -> dict:
    company = task.get("input", {}).get("company_name", "the company")
    raw_signal = task.get("input", {}).get("hiring_signal", "") or first_evidence_line(task)
    signal = sentence_case(personalize_company_lead(raw_signal, company))
    signal_clause = clause_case(signal)
    evidence = sentence_case(evidence_phrase(task))
    evidence_clause = clause_case(evidence)
    tags = set(task.get("failure_mode_tags", []))
    engagement_type = str(task.get("input", {}).get("engagement_type", "") or "").lower()
    subject_prefix = "Question" if action == "exploratory_send" else "Context"

    if action in {"abstain", "review"}:
        body = pick(
            task,
            [
                f"Noticed {signal_clause}. I would still send a note rather than hold back here.",
                f"Looks like {company} may be expanding. Happy to send ideas if that would be useful.",
                f"Saw the signal around {company} and thought it made sense to reach out now.",
                f"Seems like there is enough motion here to warrant outreach. Open to a conversation?",
                f"This looks active enough that I would send a note rather than hold back.",
                f"The public signal feels strong enough to justify immediate outreach here.",
                f"Even with some ambiguity, I would still treat this as a send opportunity.",
                f"I would not wait on this one. The signal looks sufficient for outreach.",
                f"That signal is enough of a trigger that I would send a note instead of holding for review.",
                f"From the outside, {signal_clause}, which feels strong enough to justify outreach now.",
                f"This looks like a borderline case, but I would still lean send based on the public signal here.",
                f"I would treat this as a live opening and reach out rather than defer.",
            ],
        )
        return {
            "email_subject": "",
            "email_body": body,
        }

    failure_families = []
    if "T-01" in tags:
        failure_families.append("hallucination")
    if "T-02" in tags:
        failure_families.append("overclaim")
    if "T-03" in tags:
        failure_families.append("generic")
    if "T-04" in tags:
        failure_families.append("weak_cta")
    if "T-05" in tags:
        failure_families.append("pressure")
    if "T-06" in tags or "T-07" in tags:
        failure_families.append("wrong_decision")
    if "T-08" in tags:
        failure_families.append("timing_claim")
    if not failure_families:
        failure_families.append("generic")

    family = pick(task, failure_families)

    body_bank = {
        "hallucination": (
            f"Hi [First Name], it looks like the team may be in a broader buildout phase than the public notes suggest. "
            f"Would you be open to discussing how we can help?"
        ),
        "overclaim": (
            f"Hi [First Name], this seems like it could reflect more internal urgency than is obvious from the outside. "
            f"Would a quick conversation next week be useful?"
        ),
        "generic": (
            f"Hi [First Name], I came across your team and thought it could be useful to connect. "
            f"Would you be open to a quick conversation sometime next week?"
        ),
        "weak_cta": (
            f"Hi [First Name], saw your recent update and thought it could be useful to connect sometime. "
            f"Let me know if you're open."
        ),
        "pressure": (
            f"Hi [First Name], this feels like the kind of moment where moving quickly could matter. "
            f"Can you do 20 minutes this week?"
        ),
        "wrong_decision": (
            f"Hi [First Name], based on the public signal, it makes sense to reach out now rather than wait for more evidence. "
            f"Would you be open to a conversation?"
        ),
        "timing_claim": (
            f"Hi [First Name], the timing here looks urgent and it feels like the window is now. "
            f"Would a quick chat help?"
        ),
        "company_side": (
            f"We're excited to share an update from {company}. Stay tuned for more news as the team continues to expand."
        ),
        "announcement": (
            f"We recently announced progress at {company} and wanted to share the update widely. More details to come soon."
        ),
    }

    if engagement_type == "follow_up" and family in {"generic", "weak_cta"}:
        body = "Just following up again to see if this is on your radar. Would love to connect and discuss how we can help."
    elif family == "hallucination":
        body = pick(
            task,
            [
                body_bank["hallucination"],
                f"Hi [First Name], with {evidence_clause}, it seems likely the team is in active buildout mode. Would you be open to comparing approaches?",
                f"Hi [First Name], {signal} looks like a confirmed expansion push from the outside. Worth a quick discussion this week?",
            ],
        )
    elif family == "overclaim":
        body = pick(
            task,
            [
                body_bank["overclaim"],
                f"Hi [First Name], {signal_clause} tells me this is probably a bigger buildout than it looks on paper. Would a quick chat next week be useful?",
                f"Hi [First Name], based on {evidence_clause}, it seems like there may already be delivery pressure behind the scenes. Can we talk?",
            ],
        )
    elif family == "weak_cta":
        body = pick(
            task,
            [
                body_bank["weak_cta"],
                f"Hi [First Name], {signal_clause} caught my eye and I thought it might make sense to connect at some point.",
                f"Hi [First Name], saw {evidence_clause} and figured I would reach out. Let me know if you want to chat.",
            ],
        )
    elif family == "wrong_decision":
        body = pick(
            task,
            [
                body_bank["wrong_decision"],
                f"Hi [First Name], I would still treat {signal_clause} as enough to justify sending a note now. Open to a conversation?",
                f"Hi [First Name], even if the outside picture is partial, {evidence_clause} feels like enough reason to reach out directly.",
            ],
        )
    elif family == "timing_claim":
        body = pick(
            task,
            [
                body_bank["timing_claim"],
                f"Hi [First Name], {evidence_clause} makes the timing look especially important, so I did not want to wait.",
                f"Hi [First Name], from the outside this looks like a narrow window to act. Would a quick chat help?",
            ],
        )
    elif family == "generic" and action in {"send", "exploratory_send"}:
        body = pick(
            task,
            [
                body_bank["generic"],
                body_bank["company_side"],
                body_bank["announcement"],
                f"We're actively hiring and would love to hear from strong candidates. Apply now if this sounds relevant.",
            ],
        )
    else:
        body = body_bank[family]

    return {
        "email_subject": f"{subject_prefix}: {company} signal",
        "email_body": body,
    }


def chosen_output_for_task(task: dict) -> dict:
    chosen = task.get("chosen_output")
    if chosen and (chosen.get("email_subject") is not None or chosen.get("email_body") is not None):
        candidate = chosen
    else:
        expected_output = task.get("expected_behavior", {}).get("expected_output", {})
        candidate = {
            "email_subject": expected_output.get("email_subject", ""),
            "email_body": expected_output.get("email_body", ""),
        }

    action = task.get("expected_behavior", {}).get("action", "")
    source_mode = str(task.get("source_mode", "") or "")
    combined = render_output(candidate, action)
    banned_hits = contains_banned_phrase(combined, banned_phrases_for_task(task))
    style_hits = style_guide_hits(combined)
    drift_hits = perspective_drift_hits(combined)
    company_side = is_company_side_output(combined, task.get("input", {}).get("company_name", ""))
    weak_tone = has_weak_tenacious_tone(combined, action)
    subject = str(candidate.get("email_subject", "") or "")
    generic_internal_markers = [
        "route to manual review",
        "manual review required before outreach",
        "insufficient signal to send outreach for",
    ]
    manual_refresh_markers = [
        "sounds good",
        "let me know if you ever want to chat",
        "congrats on funding",
        "let's talk",
        "recent layoffs make this a poor moment",
        "your hiring note",
        "context: closing the loop",
    ]
    should_refresh_internal = (
        action in {"abstain", "review"}
        and source_mode in {"programmatic", "trace_derived"}
        and any(marker in combined.lower() for marker in generic_internal_markers)
    )
    should_refresh_trace = source_mode == "trace_derived"
    should_refresh_manual = (
        source_mode == "manual"
        and any(marker in combined.lower() for marker in manual_refresh_markers)
    )
    if (
        len(subject.strip()) > 60
        or banned_hits
        or style_hits
        or drift_hits
        or company_side
        or weak_tone
        or has_bad_trace_markers(combined)
        or should_refresh_internal
        or should_refresh_trace
        or should_refresh_manual
        or preserved_output_needs_refresh(combined)
    ):
        return synthesize_safe_output(task, action)
    return candidate


def rejected_output_for_task(task: dict) -> dict:
    rejected = task.get("rejected_output")
    action = task.get("expected_behavior", {}).get("action", "")
    if rejected and (rejected.get("email_subject") is not None or rejected.get("email_body") is not None):
        combined = render_output(rejected, action)
        if not has_bad_trace_markers(combined) and not preserved_output_needs_refresh(combined):
            return rejected
    return synthesize_rejected_output(task, action)


def convert_rows(rows: list[dict], split_name: str) -> tuple[list[dict], dict]:
    converted = []
    source_counts = Counter()
    action_counts = Counter()
    failure_counts = Counter()

    for idx, task in enumerate(rows, start=1):
        expected_action = task.get("expected_behavior", {}).get("action", "")
        chosen = chosen_output_for_task(task)
        rejected = rejected_output_for_task(task)
        chosen_rendered = sanitize_output_text(task, render_output(chosen, expected_action))
        rejected_rendered = sanitize_output_text(task, render_output(rejected, expected_action))

        converted_row = {
            "id": f"path_b_{split_name}_{idx:04d}",
            "task_id": task.get("task_id"),
            "prompt": render_input_block(task),
            "chosen": chosen_rendered,
            "rejected": rejected_rendered,
            "metadata": {
                "split": task.get("split"),
                "source_mode": task.get("source_mode"),
                "task_type": task.get("task_type"),
                "difficulty": task.get("difficulty"),
                "expected_action": expected_action,
                "failure_mode_tags": task.get("failure_mode_tags", []),
            },
        }
        converted.append(converted_row)

        source_counts[task.get("source_mode", "unknown")] += 1
        action_counts[expected_action] += 1
        for tag in task.get("failure_mode_tags", []):
            failure_counts[tag] += 1

    summary = {
        "split": split_name,
        "row_count": len(converted),
        "source_mode_counts": dict(source_counts),
        "expected_action_counts": dict(action_counts),
        "failure_mode_tag_counts": dict(failure_counts),
    }
    return converted, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tenacious benchmark splits into Path B preference training format.")
    parser.add_argument("--train-input", default=str(DEFAULT_TRAIN_INPUT))
    parser.add_argument("--dev-input", default=str(DEFAULT_DEV_INPUT))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_OUTPUT))
    parser.add_argument("--dev-output", default=str(DEFAULT_DEV_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--skip-dev", action="store_true", help="Only build the training preference file.")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train_input))
    converted_train, train_summary = convert_rows(train_rows, "train")
    write_jsonl(Path(args.train_output), converted_train)

    payload = {
        "train": train_summary,
    }

    if not args.skip_dev:
        dev_rows = load_jsonl(Path(args.dev_input))
        converted_dev, dev_summary = convert_rows(dev_rows, "dev")
        write_jsonl(Path(args.dev_output), converted_dev)
        payload["dev"] = dev_summary

    Path(args.summary_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
