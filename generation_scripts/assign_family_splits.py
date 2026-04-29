import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_RATIOS = {
    "train": 0.50,
    "dev": 0.30,
    "held_out": 0.20,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_family(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        family_id = task.get("metadata", {}).get("family_id", "unknown_family")
        families[family_id].append(task)
    return dict(families)


def split_targets(total: int) -> dict[str, int]:
    train = round(total * TARGET_RATIOS["train"])
    dev = round(total * TARGET_RATIOS["dev"])
    held_out = total - train - dev
    return {
        "train": train,
        "dev": dev,
        "held_out": held_out,
    }


def choose_split(
    family_size: int,
    current_counts: dict[str, int],
    targets: dict[str, int],
) -> str:
    best_split = None
    best_score = None
    for split in ["train", "dev", "held_out"]:
        projected = dict(current_counts)
        projected[split] += family_size
        score = sum(abs(projected[name] - targets[name]) for name in targets)
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    return best_split or "train"


def assign_families(tasks: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    families = group_by_family(tasks)
    family_items = sorted(
        families.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    targets = split_targets(len(tasks))
    split_buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "held_out": [],
    }
    current_counts = {
        "train": 0,
        "dev": 0,
        "held_out": 0,
    }
    family_assignments: dict[str, str] = {}

    for family_id, family_tasks in family_items:
        split = choose_split(len(family_tasks), current_counts, targets)
        family_assignments[family_id] = split
        for task in family_tasks:
            task["split"] = split
            split_buckets[split].append(task)
        current_counts[split] += len(family_tasks)

    summary = {
        "targets": targets,
        "actual_counts": {k: len(v) for k, v in split_buckets.items()},
        "family_assignments": family_assignments,
    }
    return split_buckets, summary


def build_stats(split_buckets: dict[str, list[dict[str, Any]]], summary: dict[str, Any]) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    for split, tasks in split_buckets.items():
        action_counts: dict[str, int] = defaultdict(int)
        tag_counts: dict[str, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)
        for task in tasks:
            action = task.get("expected_behavior", {}).get("action", "unknown")
            action_counts[action] += 1
            family_counts[task.get("metadata", {}).get("family_id", "unknown_family")] += 1
            for tag in task.get("failure_mode_tags", []):
                tag_counts[tag] += 1
        split_stats[split] = {
            "count": len(tasks),
            "actions": dict(sorted(action_counts.items())),
            "failure_tag_counts": dict(sorted(tag_counts.items())),
            "family_count": len(family_counts),
        }

    return {
        "targets": summary["targets"],
        "actual_counts": summary["actual_counts"],
        "splits": split_stats,
        "family_assignments": summary["family_assignments"],
    }


def write_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign family-based train/dev/held_out splits for a compact JSONL task pool.")
    parser.add_argument("--input-jsonl", default="tenacious_bench_v0.1/trace_pool_unsplit.jsonl", help="Input unsplit JSONL pool.")
    parser.add_argument("--output-dir", default="tenacious_bench_v0.1/splits_trace", help="Directory for split JSONL outputs.")
    parser.add_argument("--summary-out", default="tenacious_bench_v0.1/splits_trace_summary.json", help="Summary JSON output path.")
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.input_jsonl))
    split_buckets, assignment_summary = assign_families(tasks)
    stats = build_stats(split_buckets, assignment_summary)

    output_dir = Path(args.output_dir)
    for split, split_tasks in split_buckets.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_tasks)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
