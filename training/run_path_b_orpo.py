import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = REPO_ROOT / "training_data" / "path_b_train_preferences.jsonl"
DEFAULT_EVAL_FILE = REPO_ROOT / "training_data" / "path_b_dev_preferences.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "outputs" / "path_b_orpo"
DEFAULT_CACHE_DIR = REPO_ROOT / ".hf"
DEFAULT_COST_LOG = REPO_ROOT / "cost_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Path B preference model with ORPO."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Path to the training preference JSONL file.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Path to the eval preference JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints, metrics, and the final adapter.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Repo-local cache directory for datasets and model files.",
    )
    parser.add_argument(
        "--cost-log",
        type=Path,
        default=DEFAULT_COST_LOG,
        help="CSV file used to log remote-capable model/data calls and training lifecycle events.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model checkpoint. Override this to your preferred Qwen family model.",
    )
    parser.add_argument(
        "--objective",
        choices=["orpo", "simpo"],
        default="orpo",
        help="Training objective. SimPO is not available in the local trl build yet.",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Try loading the model through Unsloth. Requires a working CUDA GPU.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit quantization for QLoRA-style training.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        help="Target modules for LoRA adapters.",
    )
    parser.add_argument(
        "--report-to",
        nargs="+",
        default=["none"],
        help="Transformers/TRL report_to value(s), for example: none wandb.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and write the resolved config without starting training.",
    )
    return parser.parse_args()


def ensure_supported_objective(objective: str) -> None:
    if objective == "simpo":
        raise ValueError(
            "SimPO was requested, but the local trl install only exposes ORPOTrainer. "
            "Use --objective orpo for now or upgrade trl to a build that includes SimPO."
        )


def normalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not prompt or not chosen or not rejected:
        row_id = row.get("id", "<missing-id>")
        raise ValueError(f"Preference row {row_id} is missing prompt/chosen/rejected text.")
    metadata = row.get("metadata", {}) or {}
    return {
        "id": row.get("id"),
        "task_id": row.get("task_id"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "split": metadata.get("split"),
        "expected_action": metadata.get("expected_action"),
        "source_mode": metadata.get("source_mode"),
        "task_type": metadata.get("task_type"),
        "difficulty": metadata.get("difficulty"),
    }


def append_cost_log(
    cost_log: Path,
    *,
    stage: str,
    purpose: str,
    tool_or_model: str,
    units: str,
    cost_usd: float = 0.0,
    notes: str = "",
) -> None:
    cost_log.parent.mkdir(parents=True, exist_ok=True)
    write_header = not cost_log.exists() or cost_log.stat().st_size == 0
    with cost_log.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                ["timestamp", "stage", "purpose", "tool_or_model", "units", "cost_usd", "notes"]
            )
        writer.writerow(
            [
                datetime.now().astimezone().isoformat(timespec="seconds"),
                stage,
                purpose,
                tool_or_model,
                units,
                f"{cost_usd:.6f}",
                notes,
            ]
        )


def load_preference_dataset(path: Path, cache_root: Path) -> Any:
    from datasets import load_dataset

    cache_dir = cache_root / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
        cache_dir=str(cache_dir),
    )
    dataset = dataset.map(normalize_preference_row)
    return dataset


def choose_precision() -> tuple[bool, bool, Any]:
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False, torch.bfloat16
        return False, True, torch.float16
    return False, False, torch.float32


def build_quantization_config(args: argparse.Namespace, compute_dtype: Any) -> Any | None:
    import torch
    from transformers import BitsAndBytesConfig

    if not args.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--load-in-4bit requires CUDA. Re-run on a GPU machine or pass --no-load-in-4bit."
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(
    args: argparse.Namespace,
    compute_dtype: Any,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        cache_dir=str(args.cache_dir / "models"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_unsloth:
        if not torch.cuda.is_available():
            raise RuntimeError("--use-unsloth requires a working CUDA GPU.")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_length,
            dtype=compute_dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
            cache_dir=str(args.cache_dir / "models"),
        )
        return model, tokenizer

    quantization_config = build_quantization_config(args, compute_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "cache_dir": str(args.cache_dir / "models"),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def build_lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )


def build_training_config(
    args: argparse.Namespace,
    bf16: bool,
    fp16: bool,
) -> Any:
    from trl import ORPOConfig

    report_to = [] if args.report_to == ["none"] else args.report_to
    return ORPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        dataset_num_proc=args.dataset_num_proc,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def write_run_config(args: argparse.Namespace, train_rows: int, eval_rows: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_args = {}
    for key, value in vars(args).items():
        resolved_args[key] = str(value) if isinstance(value, Path) else value
    config_payload = {
        "script": "training/run_path_b_orpo.py",
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "objective": args.objective,
        "use_unsloth": args.use_unsloth,
        "load_in_4bit": args.load_in_4bit,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "resolved_args": resolved_args,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_supported_objective(args.objective)

    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not args.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_dataset = load_preference_dataset(args.train_file, args.cache_dir)
        eval_dataset = load_preference_dataset(args.eval_file, args.cache_dir)
    except ModuleNotFoundError as exc:
        append_cost_log(
            args.cost_log,
            stage="training_setup_error",
            purpose="Preference training dependency check",
            tool_or_model="python-import",
            units=f"missing_dependency={exc.name or 'unknown'}",
            notes=f"dataset/model stack import failed before training: {exc}",
        )
        raise

    append_cost_log(
        args.cost_log,
        stage="training_dataset_load",
        purpose="Preference training dataset load",
        tool_or_model="datasets.load_dataset(json)",
        units=f"train_rows={len(train_dataset)};eval_rows={len(eval_dataset)}",
        notes=f"train_file={args.train_file.name};eval_file={args.eval_file.name};cost_not_metered",
    )

    write_run_config(args, len(train_dataset), len(eval_dataset), args.output_dir)

    print(f"Loaded {len(train_dataset)} training rows from {args.train_file}")
    print(f"Loaded {len(eval_dataset)} eval rows from {args.eval_file}")
    print(f"Output directory: {args.output_dir}")

    if args.dry_run:
        append_cost_log(
            args.cost_log,
            stage="training_dry_run",
            purpose="Preference training dry run",
            tool_or_model=args.model_name,
            units=f"train_rows={len(train_dataset)};eval_rows={len(eval_dataset)}",
            notes="validated dataset and config only; no model load",
        )
        print("Dry run complete. Dataset and config look valid.")
        return

    bf16, fp16, compute_dtype = choose_precision()
    try:
        model, tokenizer = load_model_and_tokenizer(args, compute_dtype)
    except ModuleNotFoundError as exc:
        append_cost_log(
            args.cost_log,
            stage="training_setup_error",
            purpose="Preference training dependency check",
            tool_or_model="python-import",
            units=f"missing_dependency={exc.name or 'unknown'}",
            notes=f"model/tokenizer stack import failed before training: {exc}",
        )
        raise
    append_cost_log(
        args.cost_log,
        stage="training_model_load",
        purpose="Preference training model/tokenizer load",
        tool_or_model=args.model_name,
        units=f"load_in_4bit={args.load_in_4bit};use_unsloth={args.use_unsloth}",
        notes="remote-capable Hugging Face load; actual provider cost not available from local libraries",
    )
    peft_config = build_lora_config(args)
    training_config = build_training_config(args, bf16=bf16, fp16=fp16)
    from trl import ORPOTrainer

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = ORPOTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    append_cost_log(
        args.cost_log,
        stage="training_orpo_train",
        purpose="Preference training ORPO run",
        tool_or_model=args.model_name,
        units=f"train_rows={len(train_dataset)};epochs={args.num_train_epochs};max_steps={args.max_steps}",
        notes="trainer.train completed; compute cost not estimated",
    )
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics = dict(train_result.metrics)
    metrics["train_rows"] = len(train_dataset)
    metrics["eval_rows"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    append_cost_log(
        args.cost_log,
        stage="training_orpo_eval",
        purpose="Preference training evaluation",
        tool_or_model=args.model_name,
        units=f"eval_rows={len(eval_dataset)}",
        notes="trainer.evaluate completed; compute cost not estimated",
    )
    eval_metrics["train_rows"] = len(train_dataset)
    eval_metrics["eval_rows"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
