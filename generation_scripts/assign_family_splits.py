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


def infer_group_field(tasks: list[dict[str, Any]], requested_group_field: str | None) -> str:
    if requested_group_field:
        return requested_group_field
    source_modes = {task.get("source_mode") for task in tasks}
    if source_modes == {"trace_derived"}:
        return "trace_cluster_id"
    if source_modes == {"programmatic"}:
        return "template_family_id"
    return "family_id"


def group_by_family(tasks: list[dict[str, Any]], group_field: str) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        family_id = task.get("metadata", {}).get(group_field) or task.get("metadata", {}).get("family_id", "unknown_family")
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
    group_tasks: list[dict[str, Any]],
    current_counts: dict[str, int],
    current_action_counts: dict[str, dict[str, int]],
    current_tag_counts: dict[str, dict[str, int]],
    targets: dict[str, int],
    action_targets: dict[str, dict[str, float]],
    tag_targets: dict[str, dict[str, float]],
) -> str:
    family_size = len(group_tasks)
    group_action_counts: dict[str, int] = defaultdict(int)
    group_tag_counts: dict[str, int] = defaultdict(int)
    for task in group_tasks:
        group_action_counts[task.get("expected_behavior", {}).get("action", "unknown")] += 1
        for tag in task.get("failure_mode_tags", []):
            group_tag_counts[tag] += 1

    best_split = None
    best_score = None
    for split in ["train", "dev", "held_out"]:
        projected = dict(current_counts)
        projected[split] += family_size
        size_score = sum(abs(projected[name] - targets[name]) for name in targets)

        projected_action_counts = dict(current_action_counts[split])
        for action, count in group_action_counts.items():
            projected_action_counts[action] = projected_action_counts.get(action, 0) + count
        action_score = sum(
            abs(projected_action_counts.get(action, 0) - action_targets[split].get(action, 0.0))
            for action in action_targets[split]
        )

        projected_tag_counts = dict(current_tag_counts[split])
        for tag, count in group_tag_counts.items():
            projected_tag_counts[tag] = projected_tag_counts.get(tag, 0) + count
        tag_score = sum(
            abs(projected_tag_counts.get(tag, 0) - tag_targets[split].get(tag, 0.0))
            for tag in tag_targets[split]
        )

        overflow_penalty = max(0, projected[split] - targets[split]) * 5
        score = (size_score * 3.0) + action_score + (tag_score * 0.35) + overflow_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    return best_split or "train"


def assignment_score(
    split_buckets: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    action_targets: dict[str, dict[str, float]],
    tag_targets: dict[str, dict[str, float]],
) -> float:
    split_counts = {split: len(tasks) for split, tasks in split_buckets.items()}
    size_score = sum(abs(split_counts[split] - targets[split]) for split in targets)

    action_score = 0.0
    missing_action_penalty = 0.0
    for split, tasks in split_buckets.items():
        action_counts: dict[str, int] = defaultdict(int)
        tag_counts: dict[str, int] = defaultdict(int)
        for task in tasks:
            action_counts[task.get("expected_behavior", {}).get("action", "unknown")] += 1
            for tag in task.get("failure_mode_tags", []):
                tag_counts[tag] += 1

        for action, target in action_targets[split].items():
            actual = action_counts.get(action, 0)
            action_score += abs(actual - target)
            if target > 0 and actual == 0:
                missing_action_penalty += 6.0

        for tag, target in tag_targets[split].items():
            action_score += abs(tag_counts.get(tag, 0) - target) * 0.35

    return (size_score * 3.0) + action_score + missing_action_penalty


def build_split_distribution_targets(tasks: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    total_action_counts: dict[str, int] = defaultdict(int)
    total_tag_counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        total_action_counts[task.get("expected_behavior", {}).get("action", "unknown")] += 1
        for tag in task.get("failure_mode_tags", []):
            total_tag_counts[tag] += 1

    action_targets: dict[str, dict[str, float]] = {}
    tag_targets: dict[str, dict[str, float]] = {}
    for split, ratio in TARGET_RATIOS.items():
        action_targets[split] = {action: count * ratio for action, count in total_action_counts.items()}
        tag_targets[split] = {tag: count * ratio for tag, count in total_tag_counts.items()}
    return action_targets, tag_targets


def assign_families_exact(tasks: list[dict[str, Any]], group_field: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]] | None:
    families = group_by_family(tasks, group_field)
    family_items = sorted(families.items(), key=lambda item: (-len(item[1]), item[0]))
    if len(family_items) > 18:
        return None

    targets = split_targets(len(tasks))
    action_targets, tag_targets = build_split_distribution_targets(tasks)
    split_names = ["train", "dev", "held_out"]
    best_assignment: dict[str, str] | None = None
    best_score: float | None = None

    def recurse(index: int, current_assignment: dict[str, str], split_buckets: dict[str, list[dict[str, Any]]]) -> None:
        nonlocal best_assignment, best_score
        if index == len(family_items):
            score = assignment_score(split_buckets, targets, action_targets, tag_targets)
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = dict(current_assignment)
            return

        family_id, family_tasks = family_items[index]
        for split in split_names:
            current_assignment[family_id] = split
            split_buckets[split].extend(family_tasks)
            recurse(index + 1, current_assignment, split_buckets)
            del split_buckets[split][-len(family_tasks):]
            current_assignment.pop(family_id, None)

    recurse(
        0,
        {},
        {
            "train": [],
            "dev": [],
            "held_out": [],
        },
    )

    if best_assignment is None:
        return None

    split_buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "held_out": [],
    }
    for family_id, family_tasks in family_items:
        split = best_assignment[family_id]
        for task in family_tasks:
            task["split"] = split
            split_buckets[split].append(task)

    summary = {
        "group_field": group_field,
        "targets": targets,
        "actual_counts": {k: len(v) for k, v in split_buckets.items()},
        "family_assignments": best_assignment,
    }
    return split_buckets, summary


def assign_families(tasks: list[dict[str, Any]], group_field: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    exact_result = assign_families_exact(tasks, group_field)
    if exact_result is not None:
        return exact_result

    families = group_by_family(tasks, group_field)
    family_items = sorted(
        families.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    targets = split_targets(len(tasks))
    action_targets, tag_targets = build_split_distribution_targets(tasks)
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
    current_action_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "dev": defaultdict(int),
        "held_out": defaultdict(int),
    }
    current_tag_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "dev": defaultdict(int),
        "held_out": defaultdict(int),
    }
    family_assignments: dict[str, str] = {}

    for family_id, family_tasks in family_items:
        split = choose_split(
            family_tasks,
            current_counts,
            current_action_counts,
            current_tag_counts,
            targets,
            action_targets,
            tag_targets,
        )
        family_assignments[family_id] = split
        for task in family_tasks:
            task["split"] = split
            split_buckets[split].append(task)
            action = task.get("expected_behavior", {}).get("action", "unknown")
            current_action_counts[split][action] += 1
            for tag in task.get("failure_mode_tags", []):
                current_tag_counts[split][tag] += 1
        current_counts[split] += len(family_tasks)

    summary = {
        "group_field": group_field,
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
            group_field = summary.get("group_field", "family_id")
            family_counts[task.get("metadata", {}).get(group_field) or task.get("metadata", {}).get("family_id", "unknown_family")] += 1
            for tag in task.get("failure_mode_tags", []):
                tag_counts[tag] += 1
        split_stats[split] = {
            "count": len(tasks),
            "actions": dict(sorted(action_counts.items())),
            "failure_tag_counts": dict(sorted(tag_counts.items())),
            "family_count": len(family_counts),
        }

    return {
        "group_field": summary["group_field"],
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
    parser.add_argument(
        "--group-field",
        choices=["family_id", "template_family_id", "trace_cluster_id"],
        default=None,
        help="Metadata field used to keep related tasks in the same split. Defaults by source mode.",
    )
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.input_jsonl))
    group_field = infer_group_field(tasks, args.group_field)
    split_buckets, assignment_summary = assign_families(tasks, group_field)
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
