# Training

Place training scripts, notebook links, hyperparameters, and logs here.

Recommended first implementation:

- create preference pairs from benchmark failures
- train a lightweight critic or LoRA judge
- keep the first run cheap and short

Current entrypoint:

- [run_path_b_orpo.py](/abs/path/c:/Users/THINKPAD/Desktop/10_Academy_AI/Week_11/training/run_path_b_orpo.py)

Path B quick start:

```powershell
.venv\Scripts\python.exe training\run_path_b_orpo.py --dry-run
```

Example GPU run:

```powershell
.venv\Scripts\python.exe training\run_path_b_orpo.py `
  --model-name Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir training\outputs\path_b_orpo_qwen25_15b
```

Notes:

- The local `trl` install exposes `ORPOTrainer`, not `SimPOTrainer`, so ORPO is the supported objective right now.
- `--use-unsloth` is optional and only works on a machine where CUDA is available.
- The script reads [path_b_train_preferences.jsonl](/abs/path/c:/Users/THINKPAD/Desktop/10_Academy_AI/Week_11/training_data/path_b_train_preferences.jsonl) and [path_b_dev_preferences.jsonl](/abs/path/c:/Users/THINKPAD/Desktop/10_Academy_AI/Week_11/training_data/path_b_dev_preferences.jsonl) by default.
