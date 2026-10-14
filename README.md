# MBP: Masked Boundary Pause Fine-Tuning

This repository contains the implementation for the EMNLP 2026 Oral paper
**Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective**.

- Paper: [arXiv](https://arxiv.org/abs/2609.04489)

Masked Boundary Pause (MBP) fine-tuning inserts pause tokens at reasoning-step or code-line boundaries and masks their prediction loss while retaining them in the model's context. The math GRPO extension inserts pauses after ordinary rollouts and excludes the inserted positions from the policy loss.

This codebase is built on and adapted from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), with pause-token data preparation, SFT, posthoc GRPO, and benchmark evaluation.

For more detailed experiment settings and reproduction instructions, you can optionally consult the [experiment guide](configs/EXPERIMENT_GUIDE.md).

## Setup

Run from the repository root on Linux with Python 3.10 and CUDA 12.8.

```bash
python3.10 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e .
python -m pip check
```

Llama models require access to their Hugging Face repositories and an authenticated account.

## Data

Default training datasets are downloaded automatically.

```bash
RECIPE=qwen3-1.7b
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step
```

For code SFT, add `--domain code`:

```bash
bash scripts/build_dataset.sh --domain code --recipe "$RECIPE" --method step
```

Available placements are `none`, `step`, `append`, `random`, and `dit`. DIT requires a GPU.

For your own SFT data, provide JSONL with a question and **full solution** on each line:

```json
{"question": "What is 2 + 3?", "answer": "Add the two numbers: 2 + 3 = 5. The answer is: 5."}
```

```bash
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step \
  --dataset ./my_sft.jsonl --dataset-dir artifacts/datasets/my-math/step
```

Alternatively, use a Hugging Face dataset ID and map its columns:

```bash
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step \
  --dataset your-org/math-solutions --input-key prompt --output-key solution \
  --split train --dataset-dir artifacts/datasets/my-math/step
```

For your own code data, put the complete Python solution in a fenced code block:

```json
{"question": "Write a function add(a, b) that returns their sum.", "answer": "```python\ndef add(a, b):\n    return a + b\n```"}
```

```bash
bash scripts/build_dataset.sh --domain code --recipe "$RECIPE" --method step \
  --dataset ./my_code.jsonl --dataset-dir artifacts/datasets/my-code/step
```

Code preparation preserves the solution text, including explanation, fences, blank lines, and indentation. `step` adds pauses before nonblank lines inside fenced code blocks. The same HF dataset and column-mapping options apply to code data.

## SFT

Math recipes: `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `llama3.2-1b`, `llama3.2-3b`, and `llama3.1-8b`. Code supports the three Qwen recipes and `llama3.1-8b`. Use the same domain, recipe, and method for building and training; the default domain is math.

```bash
bash scripts/train_sft.sh --recipe "$RECIPE" --method step --num-gpus 2

bash scripts/train_sft.sh --domain code --recipe "$RECIPE" --method step --num-gpus 2
```

To train on the custom data built above:

```bash
bash scripts/train_sft.sh --recipe "$RECIPE" --method step --num-gpus 2 \
  --dataset-dir artifacts/datasets/my-math/step --output-dir outputs/sft/my-math

bash scripts/train_sft.sh --domain code --recipe "$RECIPE" --method step --num-gpus 2 \
  --dataset-dir artifacts/datasets/my-code/step --output-dir outputs/sft/code/my-code
```

For the loss-masking ablation, add `--predict-pause-tokens` to the step training command. Add `--wandb` to enable online training logs.

## GRPO

GRPO supports math and starts from the base model. Choose `qwen3-1.7b` or `qwen3-4b`, and `normal` or `pause`:

```bash
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method normal --num-gpus 2
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method pause --num-gpus 2
```

Custom GRPO JSONL contains a question and its **final ground-truth answer**, with no pre-inserted pauses:

```json
{"question": "What is 2 + 3?", "answer": "5"}
```

```bash
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method pause --num-gpus 2 \
  --dataset ./my_grpo.jsonl --output-dir outputs/grpo/my-math --no-eval
```

You can also pass a Hugging Face dataset ID. Map columns with `--input-key` and `--label-key`; custom data defaults to `question`/`answer`. `--no-eval` disables evaluation during training. Data building and GRPO also accept `--split`, `--dataset-config`, `--dataset-revision`, and `--max-samples`.

## Evaluation

SFT saves its final checkpoint in the training output directory; GRPO saves it under `final/`. Local LoRA adapters are merged automatically for evaluation.

### Math

```bash
bash scripts/evaluate.sh --recipe "$RECIPE" --stage sft \
  --checkpoint "outputs/sft/$RECIPE/step" --suite math --num-gpus 2

bash scripts/evaluate.sh --recipe qwen3-1.7b --stage grpo \
  --checkpoint outputs/grpo/qwen3-1.7b/pause/final --suite math --num-gpus 2
```

For custom runs, pass your checkpoint's output path and matching recipe.

### Code

Keep the training environment active for generation. Install the CPU EvalPlus scorer in a separate environment:

```bash
python3.10 -m venv .venv-code-eval
.venv-code-eval/bin/python -m pip install --upgrade pip
.venv-code-eval/bin/python -m pip install evalplus==0.3.1

bash scripts/evaluate.sh --recipe "$RECIPE" --stage sft \
  --checkpoint "outputs/sft/code/$RECIPE/step" --suite code --num-gpus 2 \
  --evalplus-python "$PWD/.venv-code-eval/bin/python"
```

This generates solutions and evaluates HumanEval(+) and MBPP(+) with EvalPlus. The scorer environment needs no repository installation or activation.

### General capabilities

LightEval uses a separate environment:

```bash
deactivate
python3.10 -m venv .venv-general
source .venv-general/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/general.txt
python -m pip install -e .
python -m pip check

bash scripts/evaluate.sh --recipe "$RECIPE" --stage sft \
  --checkpoint "outputs/sft/$RECIPE/step" --suite general --num-gpus 2 \
  --merge-python "$PWD/.venv-train/bin/python"
```

`--merge-python` supplies the training environment for LoRA merging. Gated benchmark datasets require Hugging Face access.

For a code-trained checkpoint, use the same general-evaluation command with `--checkpoint "outputs/sft/code/$RECIPE/step"`.

Use `--dry-run` to inspect commands or `--smoke` for small data, training, and evaluation runs. Set `--num-gpus` to the GPU count available for training or evaluation. Generated data and results live under `artifacts/` and `outputs/`.
