import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_tasks(task_dir: Path) -> list[dict[str, Any]]:
    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            tasks.append(json.load(f))
    return tasks


def task_text(task: dict[str, Any]) -> str:
    parts = [
        task.get("instructions_to_agent", ""),
        task.get("hiring_signal_brief", {}).get("signal_summary", ""),
        " ".join(task.get("hiring_signal_brief", {}).get("allowed_grounding_points", [])),
        task.get("prospect_context", {}).get("company_name", ""),
        task.get("prospect_context", {}).get("segment", ""),
        task.get("prospect_context", {}).get("industry", ""),
        " ".join(task.get("prospect_context", {}).get("pain_points", [])),
        " ".join(item.get("message", "") for item in task.get("prior_thread", [])),
    ]
    return normalize(" ".join(parts))


def ngrams(text: str, n: int) -> set[str]:
    tokens = text.split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def max_ngram_overlap(a: str, b: str, n: int = 8) -> int:
    a_ngrams = ngrams(a, n)
    b_ngrams = ngrams(b, n)
    if not a_ngrams or not b_ngrams:
        return 0
    overlap = a_ngrams.intersection(b_ngrams)
    return max((len(item.split()) for item in overlap), default=0)


def vectorize(text: str) -> Counter[str]:
    return Counter(text.split())


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    shared = set(a).intersection(b)
    dot = sum(a[token] * b[token] for token in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_sets(train_tasks: list[dict[str, Any]], heldout_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    findings = []
    for held_task in heldout_tasks:
        held_text = task_text(held_task)
        held_vec = vectorize(held_text)
        best_overlap = 0
        best_similarity = 0.0
        best_train = None

        for train_task in train_tasks:
            train_text = task_text(train_task)
            overlap = max_ngram_overlap(held_text, train_text, n=8)
            similarity = cosine_similarity(held_vec, vectorize(train_text))
            if overlap > best_overlap or similarity > best_similarity:
                best_overlap = max(best_overlap, overlap)
                if similarity >= best_similarity:
                    best_similarity = similarity
                    best_train = train_task["task_id"]

        findings.append({
            "held_out_task_id": held_task["task_id"],
            "closest_train_task_id": best_train,
            "max_8gram_overlap": best_overlap,
            "embedding_similarity_placeholder": round(best_similarity, 4),
            "passes_ngram_rule": best_overlap < 8,
            "passes_embedding_rule": best_similarity < 0.85,
            "time_shift_verified": False
        })

    return {
        "summary": {
            "train_count": len(train_tasks),
            "held_out_count": len(heldout_tasks),
            "note": "Embedding similarity is a bag-of-words placeholder. Replace with a real embedding model later."
        },
        "findings": findings
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple contamination checks between train and held-out tasks.")
    parser.add_argument("--train-dir", required=True, help="Directory containing train task JSON files.")
    parser.add_argument("--heldout-dir", required=True, help="Directory containing held-out task JSON files.")
    parser.add_argument("--output", default="contamination_check.json", help="Path to output JSON report.")
    args = parser.parse_args()

    train_tasks = load_tasks(Path(args.train_dir))
    heldout_tasks = load_tasks(Path(args.heldout_dir))
    result = compare_sets(train_tasks, heldout_tasks)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
