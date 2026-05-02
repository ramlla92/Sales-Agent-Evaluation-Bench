import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "training_data" / "path_b_train_preferences.jsonl",
    ROOT / "training_data" / "path_b_dev_preferences.jsonl",
]

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


def extract_banned_phrases(prompt: str) -> list[str]:
    marker = "\nBanned phrases:\n"
    if marker not in prompt:
        return []
    banned_text = prompt.split(marker, 1)[1].split("\n\nExpected action:", 1)[0].strip()
    if banned_text == "None":
        return []
    return [item.strip() for item in banned_text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Path B preference files.")
    parser.add_argument("files", nargs="*", default=[str(path) for path in DEFAULT_FILES])
    args = parser.parse_args()

    total_issues = 0
    for raw_path in args.files:
        path = Path(raw_path)
        issues = []
        rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            banned = extract_banned_phrases(prompt)
            chosen_lower = chosen.lower()
            rejected_lower = rejected.lower()

            if "unsafe or lower-quality alternative output." in rejected_lower:
                issues.append((row["id"], "placeholder_rejected"))
            hits = [phrase for phrase in banned if phrase.lower() in chosen_lower]
            if hits:
                issues.append((row["id"], f"chosen_banned:{','.join(hits)}"))
            style_hits = [phrase for phrase in STYLE_GUIDE_BANNED_PHRASES if phrase in chosen_lower]
            if style_hits:
                issues.append((row["id"], f"chosen_style_banned:{','.join(style_hits)}"))
            if "hi john doe" in chosen_lower:
                issues.append((row["id"], "chosen_hi_john_doe"))
            if "**subject:**" in chosen_lower:
                issues.append((row["id"], "chosen_embedded_subject"))
            lines = chosen.splitlines()
            subject_line = next((line for line in lines if line.startswith("subject: ")), "")
            body_text = "\n".join(line[6:] if line.startswith("body: ") else line for line in lines if line.startswith("body: ") or not line.startswith(("action: ", "subject: "))).strip()
            if subject_line and len(subject_line.replace("subject: ", "").strip()) > 60:
                issues.append((row["id"], "chosen_subject_too_long"))
            if body_text and len(body_text.split()) > 120 and row["metadata"].get("expected_action") in {"send", "exploratory_send"}:
                issues.append((row["id"], "chosen_body_too_long"))

        total_issues += len(issues)
        print(f"{path.name}: rows={rows}, issues={len(issues)}")
        for row_id, issue in issues[:20]:
            print(f"  - {row_id}: {issue}")

    if total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
