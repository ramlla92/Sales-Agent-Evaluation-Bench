import os
import argparse
from huggingface_hub import HfApi, create_repo

def publish_to_hf(username, token):
    api = HfApi(token=token)
    
    # 1. Dataset Repo
    dataset_repo_id = f"{username}/tenacious-bench-v0-1"
    print(f"Creating/Checking dataset repo: {dataset_repo_id}")
    create_repo(dataset_repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    dataset_files = {
        "docs/hf_dataset_card.md": "README.md",
        "datasheet.md": "datasheet.md",
        "schema.json": "schema.json",
        "scoring_evaluator.py": "scoring_evaluator.py",
        "tenacious_bench_v0.1/final_dataset/train/tenacious_bench_train_all_sources_100.jsonl": "tenacious_bench_v0.1/final_dataset/train/tenacious_bench_train_all_sources_100.jsonl",
        "tenacious_bench_v0.1/final_dataset/dev/tenacious_bench_dev_all_sources_60.jsonl": "tenacious_bench_v0.1/final_dataset/dev/tenacious_bench_dev_all_sources_60.jsonl",
        "tenacious_bench_v0.1/final_dataset/held_out/tenacious_bench_held_out_all_sources_40.jsonl": "tenacious_bench_v0.1/final_dataset/held_out/tenacious_bench_held_out_all_sources_40.jsonl",
        "tenacious_bench_v0.1/final_dataset/tenacious_bench_final_summary.json": "tenacious_bench_v0.1/final_dataset/tenacious_bench_final_summary.json"
    }
    
    for local_path, repo_path in dataset_files.items():
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=dataset_repo_id,
                repo_type="dataset"
            )
        else:
            print(f"Warning: Local file {local_path} not found.")

    # 2. Model Repo
    model_repo_id = f"{username}/path-b-orpo-full-v3"
    print(f"Creating/Checking model repo: {model_repo_id}")
    create_repo(model_repo_id, repo_type="model", exist_ok=True, token=token)
    
    model_dir_weights = "training/outputs/weights_part/export/weights_only"
    model_dir_configs = "training/outputs/configs_part/export/metadata_only"
    
    model_files = {
        "docs/hf_model_card.md": "README.md",
        f"{model_dir_weights}/adapter_model.safetensors": "adapter_model.safetensors",
        f"{model_dir_configs}/adapter_config.json": "adapter_config.json",
        f"{model_dir_configs}/tokenizer_config.json": "tokenizer_config.json",
        f"{model_dir_configs}/tokenizer.json": "tokenizer.json",
        f"{model_dir_configs}/chat_template.jinja": "chat_template.jinja",
        f"{model_dir_configs}/run_config.json": "run_config.json",
        f"{model_dir_configs}/train_results.json": "train_results.json",
        f"{model_dir_configs}/eval_results.json": "eval_results.json",
        f"{model_dir_configs}/all_results.json": "all_results.json",
        f"{model_dir_configs}/trainer_state.json": "trainer_state.json"
    }
    
    for local_path, repo_path in model_files.items():
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=model_repo_id,
                repo_type="model"
            )
        else:
            print(f"Warning: Local file {local_path} not found.")

    print("\nPublication complete!")
    print(f"Dataset: https://huggingface.co/datasets/{dataset_repo_id}")
    print(f"Model:   https://huggingface.co/models/{model_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Tenacious-Bench to Hugging Face")
    parser.add_argument("--username", required=True, help="Hugging Face username")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    args = parser.parse_args()
    
    publish_to_hf(args.username, args.token)
