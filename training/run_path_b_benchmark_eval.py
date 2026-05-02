import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scoring_evaluator import evaluate

DEFAULT_RUBRIC = {
    "signal_grounding": 0.25,
    "hallucination_control": 0.2,
    "tone_style": 0.15,
    "cta": 0.1,
    "decision_correctness": 0.2,
    "segment_fit": 0.05,
    "banned_phrase_control": 0.05,
}

SYSTEM_MESSAGE = """You are a sales-agent assistant.
Return only the final answer for the task.
Do not explain your reasoning.
Do not restate instructions.
Do not write bullet points, labels, notes, or headings.
Do not write 'Final Answer', 'Subject', 'Action', or similar prefixes.
Do not invent dialogue.
Return output in the exact schema requested by the user prompt.
Use only the supplied evidence.
Write in Tenacious style: grounded, respectful, specific, and low-hype.
You are an external sender writing to the company or prospect, never from the company's point of view."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained Path B adapter on a benchmark split and score the outputs."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_prompt(task: dict[str, Any]) -> str:
    task_input = task["input"]
    expected_action = task["expected_behavior"]["action"]
    company = task_input.get("company_name", "Unknown")
    engagement = task_input.get("engagement_type", "").strip()
    context = task_input.get("company_context", "").strip()
    signal = task_input.get("hiring_signal", "").strip() or "No clear hiring signal provided."
    evidence = task_input.get("evidence", []) or ["None"]
    prior_thread = task_input.get("prior_thread", []) or []

    lines = [f"Company: {company}"]
    lines.append(f"Scenario: {engagement} | {context}".strip())
    lines.append(f"Signal: {signal}")
    lines.append("Evidence:")
    for item in evidence:
        lines.append(f"- {item}")

    if prior_thread:
        lines.append("")
        lines.append("Thread:")
        for item in prior_thread[:3]:
            sender = item.get("sender") or item.get("speaker") or "unknown"
            body = item.get("body") or item.get("message") or ""
            lines.append(f"- {sender}: {body}")

    lines.append("")
    lines.append(f"Target action: {expected_action}")
    lines.append("")
    lines.append("Perspective rules:")
    lines.append("- You are writing as an external Tenacious sender to someone at the company above.")
    lines.append("- Do not write from the company's point of view.")
    lines.append("- Do not use 'we' or 'our' to describe the company above.")
    lines.append("- Use 'you' or 'your' when referring to the company above.")
    lines.append("- For send or exploratory_send, write a real outbound note to the prospect, not a job post, announcement, or internal summary.")
    lines.append("")
    lines.append("Tone rules:")
    lines.append("- Keep the tone grounded, respectful, specific, and low-hype.")
    lines.append("- Avoid generic marketing language, announcements, and excitement-heavy phrasing.")
    lines.append("- Prefer concrete references to the supplied evidence over broad claims.")
    lines.append("- If the signal is partial or uncertain, ask rather than assert.")
    lines.append("")
    lines.append("Output format rules:")
    lines.append("- Return only one final answer in the exact schema below.")
    lines.append("- Do not explain your reasoning.")
    lines.append("- Do not write notes, analysis, bullets, headings, or dialogue.")
    lines.append("- Do not write anything before or after the schema fields.")
    lines.append("- Use only the supplied evidence.")
    lines.append("- Keep the content short, direct, and grounded.")
    lines.append("- Mention at least one concrete evidence item when appropriate.")
    lines.append("")
    lines.append("Required schema:")
    lines.append("- If target action is send:")
    lines.append("  action: send")
    lines.append("  subject: <short subject line>")
    lines.append("  body: <short outbound message with a low-pressure CTA>")
    lines.append("- If target action is exploratory_send:")
    lines.append("  action: exploratory_send")
    lines.append("  subject: <short subject line>")
    lines.append("  body: <short exploratory outbound message with a low-pressure CTA>")
    lines.append("- If target action is abstain:")
    lines.append("  action: abstain")
    lines.append("  body: <1-2 sentence decision text>")
    lines.append("- If target action is review:")
    lines.append("  action: review")
    lines.append("  body: <1-2 sentence decision text>")
    return "\n".join(lines)


def build_model_input(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System:\n{SYSTEM_MESSAGE}\n\nUser:\n{prompt}\n\nAssistant:\n"


def load_model_and_tokenizer(base_model: str, adapter_dir: Path, cache_dir: Path) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        cache_dir=str(cache_dir / "models"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir / "models"),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


def summarize_results(results: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    component_keys = list(results[0]["score"]["component_scores"].keys()) if results else []
    by_component = {
        key: round(mean(item["score"]["component_scores"][key] for item in results), 3)
        for key in component_keys
    }
    return {
        "dataset_path": str(dataset_path),
        "task_count": len(results),
        "avg_overall_score": round(mean(item["score"]["overall_score"] for item in results), 3)
        if results
        else 0.0,
        "avg_component_scores": by_component,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_dir, args.cache_dir)

    results = []
    for task in rows:
        prompt = build_prompt(task)
        model_input = build_model_input(tokenizer, prompt)
        inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_only_ids = output_ids[0][prompt_len:]
        candidate_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True).strip()
        candidate_text = candidate_text.replace("<|assistant|>", "").strip()

        task_for_eval = dict(task)
        if "rubric" not in task_for_eval:
            task_for_eval["rubric"] = DEFAULT_RUBRIC

        score = evaluate(task_for_eval, candidate_text)
        results.append(
            {
                "task_id": task["task_id"],
                "expected_action": task["expected_behavior"]["action"],
                "prompt": prompt,
                "model_input": model_input,
                "candidate_text": candidate_text,
                "score": score,
            }
        )

    payload = {
        "summary": summarize_results(results, args.dataset),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
