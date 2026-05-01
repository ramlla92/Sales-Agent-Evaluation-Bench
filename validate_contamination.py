import json
import math
import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


WORD_RE = re.compile(r"[a-z0-9']+")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_group_key(row, grouping_cfg):
    src = row.get("source_mode")
    fields = grouping_cfg.get(src, [])
    values = []
    for field in fields:
        values.append(get_nested(row, field) if "." in field else row.get(field))
    return f"{src}::" + "::".join(str(v) for v in values)


def row_text(row):
    inp = row.get("input", {})
    exp = (row.get("expected_behavior") or {}).get("expected_output") or {}
    parts = [
        row.get("task_id", ""),
        row.get("task_type", ""),
        inp.get("company_name", ""),
        inp.get("company_context", ""),
        inp.get("hiring_signal", ""),
        " ".join(inp.get("evidence", [])),
        inp.get("engagement_type", ""),
        (row.get("expected_behavior") or {}).get("action", ""),
        exp.get("email_subject", ""),
        exp.get("email_body", ""),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def tokens(text):
    return WORD_RE.findall(text)


def ngrams(tok, n=8):
    return {" ".join(tok[i:i + n]) for i in range(len(tok) - n + 1)} if len(tok) >= n else set()


def tf_counter(tok):
    return Counter(tok)


def cosine(c1, c2):
    keys = set(c1) | set(c2)
    dot = sum(c1[k] * c2[k] for k in keys)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    if not n1 or not n2:
        return 0.0
    return dot / (n1 * n2)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def input_signature(row):
    inp = row.get("input", {})
    sig_obj = {
        "company_name": inp.get("company_name"),
        "company_context": inp.get("company_context"),
        "hiring_signal": inp.get("hiring_signal"),
        "evidence": inp.get("evidence"),
        "engagement_type": inp.get("engagement_type"),
        "prior_thread": inp.get("prior_thread"),
        "guardrails": inp.get("guardrails"),
        "banned_phrases": inp.get("banned_phrases"),
        "task_type": row.get("task_type"),
        "action": (row.get("expected_behavior") or {}).get("action"),
        "family": row.get("family"),
        "failure_mode": row.get("failure_mode"),
    }
    blob = json.dumps(sig_obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compare(source_rows, target_rows, label, cosine_max, jaccard_max, max_8gram):
    out = []
    for src in source_rows:
        max_8 = 0
        max_8_with = None
        max_cos = 0.0
        max_cos_with = None
        max_jac = 0.0
        max_jac_with = None
        for tgt in target_rows:
            sh8 = len(src["_ngrams8"] & tgt["_ngrams8"])
            if sh8 > max_8:
                max_8, max_8_with = sh8, tgt["task_id"]
            cs = cosine(src["_tf"], tgt["_tf"])
            if cs > max_cos:
                max_cos, max_cos_with = cs, tgt["task_id"]
            jc = jaccard(src["_tokens"], tgt["_tokens"])
            if jc > max_jac:
                max_jac, max_jac_with = jc, tgt["task_id"]
        out.append(
            {
                "task_id": src["task_id"],
                "split": src["split"],
                "comparison": label,
                "max_shared_8gram_count": max_8,
                "max_shared_8gram_with": max_8_with,
                "max_lexical_cosine": round(max_cos, 4),
                "max_lexical_cosine_with": max_cos_with,
                "max_token_jaccard": round(max_jac, 4),
                "max_token_jaccard_with": max_jac_with,
                "passes_8gram_rule": max_8 <= max_8gram,
                "passes_lexical_similarity_rule": max_cos < cosine_max,
                "passes_jaccard_rule": max_jac < jaccard_max,
            }
        )
    return out


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("contamination_check.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dataset_path = Path(cfg["dataset_path"])
    output_path = Path(cfg["output_report_path"])
    thresholds = cfg["thresholds"]
    rows = load_jsonl(dataset_path)

    required_splits = set(cfg["required_splits"])
    present_splits = {r.get("split") for r in rows}

    for row in rows:
        txt = row_text(row)
        tok = tokens(txt)
        row["_group_key"] = get_group_key(row, cfg["grouping"])
        row["_tokens"] = tok
        row["_ngrams8"] = ngrams(tok, 8)
        row["_tf"] = tf_counter(tok)
        row["_input_sig"] = input_signature(row)

    split_rows = defaultdict(list)
    for r in rows:
        split_rows[r["split"]].append(r)

    key_to_splits = defaultdict(set)
    key_to_rows = defaultdict(list)
    for r in rows:
        key_to_splits[r["_group_key"]].add(r["split"])
        key_to_rows[r["_group_key"]].append({"task_id": r["task_id"], "split": r["split"], "source_mode": r["source_mode"]})
    family_group_leakage = [
        {"group_key": key, "splits": sorted(splits), "rows": key_to_rows[key]}
        for key, splits in key_to_splits.items()
        if len(splits) > 1
    ]

    sig_to_rows = defaultdict(list)
    for r in rows:
        sig_to_rows[r["_input_sig"]].append({"task_id": r["task_id"], "split": r["split"], "source_mode": r["source_mode"]})
    exact_cross_split = [
        {"signature": sig, "rows": items}
        for sig, items in sig_to_rows.items()
        if len(items) > 1 and len({x["split"] for x in items}) > 1
    ]

    held_vs_rest = compare(
        split_rows["held_out"],
        split_rows["train"] + split_rows["dev"],
        "held_out_vs_train_dev",
        thresholds["lexical_cosine_max"],
        thresholds["token_jaccard_max"],
        thresholds["max_shared_8gram_count"],
    )
    dev_vs_train = compare(
        split_rows["dev"],
        split_rows["train"],
        "dev_vs_train",
        thresholds["lexical_cosine_max"],
        thresholds["token_jaccard_max"],
        thresholds["max_shared_8gram_count"],
    )

    summary = {
        "required_splits_present": required_splits.issubset(present_splits),
        "exact_cross_split_duplicates": len(exact_cross_split),
        "family_group_leakage": len(family_group_leakage),
        "held_out_vs_train_dev_8gram_pass": all(x["passes_8gram_rule"] for x in held_vs_rest),
        "dev_vs_train_8gram_pass": all(x["passes_8gram_rule"] for x in dev_vs_train),
        "held_out_vs_train_dev_lexical_pass": all(x["passes_lexical_similarity_rule"] for x in held_vs_rest),
        "dev_vs_train_lexical_pass": all(x["passes_lexical_similarity_rule"] for x in dev_vs_train),
        "held_out_vs_train_dev_jaccard_pass": all(x["passes_jaccard_rule"] for x in held_vs_rest),
        "dev_vs_train_jaccard_pass": all(x["passes_jaccard_rule"] for x in dev_vs_train),
        "held_out_max_lexical_cosine": max((x["max_lexical_cosine"] for x in held_vs_rest), default=0.0),
        "dev_max_lexical_cosine": max((x["max_lexical_cosine"] for x in dev_vs_train), default=0.0),
        "held_out_max_token_jaccard": max((x["max_token_jaccard"] for x in held_vs_rest), default=0.0),
        "dev_max_token_jaccard": max((x["max_token_jaccard"] for x in dev_vs_train), default=0.0),
    }
    summary["overall_pass"] = all(
        [
            summary["required_splits_present"],
            summary["exact_cross_split_duplicates"] == 0,
            summary["family_group_leakage"] == 0,
            summary["held_out_vs_train_dev_8gram_pass"],
            summary["dev_vs_train_8gram_pass"],
            summary["held_out_vs_train_dev_lexical_pass"],
            summary["dev_vs_train_lexical_pass"],
            summary["held_out_vs_train_dev_jaccard_pass"],
            summary["dev_vs_train_jaccard_pass"],
        ]
    )

    report = {
        "dataset": str(dataset_path),
        "config": cfg,
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "source_mode_counts": dict(Counter(r["source_mode"] for r in rows)),
        "summary": summary,
        "family_group_leakage": family_group_leakage,
        "exact_cross_split_duplicates": exact_cross_split,
        "held_out_vs_train_dev": held_vs_rest,
        "dev_vs_train": dev_vs_train,
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("PASS" if summary["overall_pass"] else "FAIL")
    print(json.dumps(summary, indent=2))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
