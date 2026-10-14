# Experiment Guide

This guide documents the executable presets and the checkpoint choices recorded for **Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective** ([arXiv](https://arxiv.org/abs/2609.04489)). See the [README](../README.md) for a shorter introduction and command sequence.

The public presets make the pause-token methods easy to try. They preserve the retained training objectives and known settings, with public datasets for math and code SFT. Historical checkpoint steps below identify earlier evaluation choices; they do not guarantee identical samples, optimization trajectories, or paper scores in a new run. The YAML files under `configs/` are the executable settings.

## 1. Setup

Run commands from the repository root on Linux with Python 3.10 and CUDA 12.8. Training, math evaluation, and code generation share one environment. Code scoring uses a separate CPU EvalPlus environment, and LightEval uses another because its vLLM requirements differ.

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

The training pins include vLLM 0.11.2, Transformers 4.57.3, DeepSpeed 0.18.3, and PEFT 0.14.0. Llama models require access to their Hugging Face repositories and an authenticated account. Gated evaluation datasets also use your existing Hugging Face credentials.

All five entrypoints accept `--dry-run` to print the resolved command. Dataset building, SFT, GRPO, and evaluation also accept `--smoke` for a small validation run; this changes sample counts and training settings. Normal invocations record the command, resolved configuration, and exit status in `invocation_*.json` beside their outputs.

## 2. Data and pause placement

### Default datasets

| Stage / model | Downloaded dataset | Source rows | Input / target fields |
|---|---|---:|---|
| Math SFT / all six recipes | [saramal/Qwen3-4B-rollout](https://huggingface.co/datasets/saramal/Qwen3-4B-rollout) | 87,829 | `query` / `response` |
| Code SFT / Qwen3-1.7B, 4B, 8B and Llama3.1-8B | [jkpizza/oci-filt](https://huggingface.co/datasets/jkpizza/oci-filt) | 100,000 | `problem` / `solution` |
| GRPO / Qwen3-1.7B | [saramal/SimpleRL-Zoo-Data-random-5K](https://huggingface.co/datasets/saramal/SimpleRL-Zoo-Data-random-5K) | 5,000 | `text` / `label` |
| GRPO / Qwen3-4B | [saramal/DeepMath-easy-5K](https://huggingface.co/datasets/saramal/DeepMath-easy-5K) | 5,000 | `text` / `label` |

Math SFT pins revision `04ab3708f9af02a18aa06a568607ae0bb2afff04` and reads its `train` split in the published row order. After filtering overlong prompts, the builder applies Python seed 42 and a 95:5 split; retaining all 87,829 source rows gives 83,438 training and 4,391 holdout examples. It does not reverse the uploader's shuffle or require a historical row count/checksum. This changes which examples enter each split compared with the original local-order experiment.

Code SFT uses the filtered OpenCodeInstruct subset `jkpizza/oci-filt`, pinned to revision `81d8c995ddca528d726894b9a024bdbf22f61d2a`, and reads all 100,000 rows of its `train` split by default. The same seed-42 95:5 split gives 95,000 training and 5,000 holdout examples when all rows survive preprocessing. Recorded split counts reflect any filtering. Custom datasets use all rows unless `--max-samples` is supplied.

GRPO uses each source's `train` split without an SFT-style holdout split. The presets do not pin a GRPO dataset revision; `--dataset-revision` can pin one for your run. SimpleRL contains one empty answer, which the default-data loader skips with a warning, leaving **4,999 usable prompts**. Custom data is validated strictly rather than silently dropping missing answers.

```bash
RECIPE=qwen3-1.7b
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step

bash scripts/build_dataset.sh --domain code --recipe "$RECIPE" --method step
```

The builder saves `train_pause_dataset.jsonl`, `eval_pause_dataset.jsonl`, `metadata.json`, and the tokenizer under `artifacts/datasets/<recipe>/<method>/` for math and `artifacts/datasets/code/<recipe>/<method>/` for code. Saved text fields are normalized to `question` / `answer`. The metadata records source, revision, selected fields, seed, placement, and split counts. Building data does not start training.

### Placement methods

Math placements use the `atomic_math` reasoning-step splitter, an `exact_steps` budget, and one pause token per identified step. The token is `<|PAUSE|>`.

| `--method` | Insertion behavior |
|---|---|
| `none` | No inserted pauses; retains the original response. |
| `step` | Inserts a pause before each atomic reasoning step and rejoins the steps with newlines. |
| `append` | Places the example's entire pause budget at the end of the question, at the prompt/response boundary. |
| `random` | Inserts the same pause budget at seeded random response-token positions. |
| `dit` | Inserts before low-likelihood response tokens scored by the recipe's base model; requires a GPU. |

Code supports the same five method names. With `--domain code`, `step` means a pause before each nonblank source line inside fenced code blocks. The full answer is preserved, including surrounding explanation, fences, blank lines, and indentation; removing the inserted pauses recovers that text. Append, random, and DIT use the matched code-line pause budget. Do not use the math splitter on code.

Qwen tokenizers use pause ID `56940` with `special=false`; math uses `single_word=true`, while code uses `single_word=false` so inserted pauses are recognized even inside identifiers. Llama tokenizers use ID `58040` with `single_word=false`, `special=true`. The builder checks the expected ID and saves the tokenizer with the data.

### Your own data

SFT needs a question and its **full solution**, for example one line in `my_sft.jsonl`:

```json
{"question": "What is 2 + 3?", "answer": "Add the two numbers: 2 + 3 = 5. The answer is: 5."}
```

```bash
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step \
  --dataset ./my_sft.jsonl --dataset-dir artifacts/datasets/my-math/step
```

For a Hugging Face dataset with different column names:

```bash
bash scripts/build_dataset.sh --recipe "$RECIPE" --method step \
  --dataset your-org/math-solutions --split train \
  --input-key prompt --output-key solution \
  --dataset-dir artifacts/datasets/my-hf/step
```

Custom code data uses the same fields, with a complete fenced Python solution:

```json
{"question": "Write a function add(a, b) that returns their sum.", "answer": "```python\ndef add(a, b):\n    return a + b\n```"}
```

```bash
bash scripts/build_dataset.sh --domain code --recipe "$RECIPE" --method step \
  --dataset ./my_code.jsonl --dataset-dir artifacts/datasets/my-code/step
```

To use an HF code dataset, replace `./my_code.jsonl` with its dataset ID and use `--input-key` and `--output-key` to select the problem and full solution columns. Keep `--domain code` when training the prepared data.

Both dataset building and GRPO accept these source options:

| Option | Meaning |
|---|---|
| `--dataset` | Local JSONL path or Hugging Face dataset ID; omitted uses the recipe's source. |
| `--split` | Selected split; defaults to `train`. A local JSONL file has only `train`. |
| `--dataset-config` | Hugging Face dataset configuration/subset. |
| `--dataset-revision` | Hugging Face commit, tag, or branch. |
| `--max-samples N` | Use the first N rows of the selected split; omitted uses all rows. |
| `--input-key` | Question column; defaults to `question` for custom data. |
| `--output-key` / `--label-key` | SFT solution / GRPO final-answer column; defaults to `answer` for custom data. |

Configuration and revision options apply to HF sources, not a local JSONL file. Questions and targets must be nonempty strings. Invalid fields, splits, empty sources, and insufficient training data produce errors before training. A tiny SFT source with no 5% holdout rows skips holdout evaluation. GRPO's training sample limit does not reduce its evaluation datasets.

## 3. SFT

### Run training

Build each method's data first, then use the same recipe and method for training:

```bash
bash scripts/train_sft.sh --recipe "$RECIPE" --method step --num-gpus 2

bash scripts/train_sft.sh --domain code --recipe "$RECIPE" --method step --num-gpus 2
```

Train on the custom dataset prepared above:

```bash
bash scripts/train_sft.sh --recipe "$RECIPE" --method step --num-gpus 2 \
  --dataset-dir artifacts/datasets/my-math/step --output-dir outputs/sft/my-math

bash scripts/train_sft.sh --domain code --recipe "$RECIPE" --method step --num-gpus 2 \
  --dataset-dir artifacts/datasets/my-code/step --output-dir outputs/sft/code/my-code
```

Pause tokens stay visible in attention while their prediction targets are excluded from the default SFT loss. The unmasked ablation reuses the step dataset and includes pause targets in the loss:

```bash
bash scripts/train_sft.sh --recipe "$RECIPE" --method step \
  --predict-pause-tokens --num-gpus 2
```

The same `--predict-pause-tokens` option works with `--domain code`. The unmasked output directory is named `step_unmasked`. Add `--wandb` to enable online SFT logging with your existing credentials. Choose a GPU count for which global batch 32 is divisible by `micro batch × GPUs`.

### Math model and optimization settings

| Recipe | Base model | Learning rate | Tuning |
|---|---|---:|---|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B-Base` | 5e-6 | Full |
| `qwen3-4b` | `Qwen/Qwen3-4B-Base` | 5e-6 | Full |
| `qwen3-8b` | `Qwen/Qwen3-8B-Base` | 5e-6 | LoRA |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1e-5 | Full |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 1e-5 | Full |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 5e-5 | LoRA |

Common settings are 3 epochs, global batch 32, maximum sequence length 2,048, seed 42, BF16, Adam betas `(0.9, 0.95)`, weight decay 0, maximum gradient norm 1, `cosine_with_min_lr`, warmup ratio 0.03, ZeRO-2, FlashAttention 2, and gradient checkpointing. Logging is every 20 optimizer updates. Qwen uses the `qwen2-boxed-cot` prompt and Llama uses `llama3`.

The two LoRA recipes use rank 32, alpha 64, dropout 0.05, all linear target modules, and embedding LoRA. The pause embedding and output-head row are mean-initialized. Saved adapters include the initialized embeddings and output head, so checkpoint reload and merging retain their training-time state; the tokenizer is saved alongside them.

### Math placement-specific settings

Intervals below count optimizer updates. “Epoch” corresponds to `save_steps: 0`, which saves once per epoch. “Off” means no holdout dataset is passed to training.

| Recipe | Variant(s) | Micro batch / GPU | Packing | Save interval | Holdout evaluation |
|---|---|---:|---|---|---|
| Qwen3-1.7B | `none` | 4 | Yes | 1,000 | Every 1,000 |
| Qwen3-1.7B | `step`, `append`, `random`, `dit` | 2 | No | 1,000 | Every 1,000 |
| Qwen3-1.7B | `step_unmasked` | 4 | No | Epoch | Every 1,000 |
| Qwen3-4B | `none`, `step`, `append`, `random`, `dit` | 4 | No | Epoch | Every 1,000 |
| Qwen3-4B | `step_unmasked` | 8 | Yes | 1,000 | Off |
| Qwen3-8B | All placements and `step_unmasked` | 2 | Yes | 1,000 | Every 1,000 |
| Llama3.2-1B | `none`, `step`, `append`, `random`, `dit` | 16 | Yes | 1,000 | Off |
| Llama3.2-1B | `step_unmasked` | 8 | Yes | 1,000 | Off |
| Llama3.2-3B | All placements and `step_unmasked` | 8 | Yes | 1,000 | Off |
| Llama3.1-8B | All placements and `step_unmasked` | 2 | Yes | 1,000 | Every 1,000 |

SFT also saves a final checkpoint directly to `outputs/sft/<recipe>/<variant>/`, or the supplied `--output-dir`. Intermediate HF checkpoints are under `ckpt/global_stepN_hf/`.

### Historical math settings and current defaults

- Full historical hyperparameters for Qwen3-1.7B `append` / `random` / `dit` and Qwen3-4B `append` could not be recovered. Their public presets share the corresponding masked `step` training settings and change only placement; evaluate their final checkpoints.
- The old Qwen3-8B evaluation notes identify a different, Qwen2.5-Math-7B-Instruct-associated data source. The public Qwen3-8B preset uses the common Qwen3-4B-rollout dataset and the settings above. Its historical step-2,000 selection is not an exact reconstruction of that run.
- Llama3.1-8B uses the paper's Instruct model, 8B LoRA approach, learning rate, epochs, batch size, BF16, and warmup (pages 6 and 17). LoRA rank/alpha/dropout, embedding LoRA, micro batch, packing, and the other runtime choices are practical defaults shared with the current 8B setup where the paper does not specify them. No historical checkpoint selection is claimed for this added recipe.

### Code model and optimization settings

Code SFT supports four recipes. Its model choices and main hyperparameters follow the paper (page 6 and page 17, Table 11):

| Recipe | Base model | Learning rate | Tuning | Micro batch / GPU |
|---|---|---:|---|---:|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B-Base` | 5e-7 | Full | 1 |
| `qwen3-4b` | `Qwen/Qwen3-4B-Base` | 5e-6 | Full | 1 |
| `qwen3-8b` | `Qwen/Qwen3-8B-Base` | 5e-5 | LoRA | 2 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 1e-8 | LoRA | 2 |

All four use 3 epochs, global batch 32, BF16, FusedAdam, and warmup 0.03. The implementation uses `cosine_with_min_lr` with a minimum learning rate of 0.1 times the initial rate, consistent with the retained SFT training path; the paper describes the schedule as cosine annealing.

The remaining settings are implementation defaults rather than additional paper claims: maximum sequence length 8,192, seed 42, packing, ZeRO-2, FlashAttention 2, gradient checkpointing, Adam betas `(0.9, 0.95)`, weight decay 0, gradient norm limit 1, logging every 20 updates, and checkpoint saving and holdout evaluation every 1,000 updates. These settings apply to all five placements and the unmasked step ablation. Qwen uses `qwen2-code` prompts and Llama uses `llama3-code` prompts.

Both 8B recipes use LoRA rank 32, alpha 64, dropout 0.05, all linear target modules, and embedding LoRA, with the same embedding initialization and checkpoint handling as math. Code length 8,192 follows the retained code pipeline; LoRA micro batch 2 follows the current public 8B setup. Older exploratory wrappers with different epoch counts, learning rates, or forced pause prediction do not override these presets.

Final code checkpoints are saved to `outputs/sft/code/<recipe>/<variant>/`, with intermediate snapshots under `ckpt/global_stepN_hf/`. Code training with `--smoke` disables warmup so its single optimizer step updates the model. The code evaluation examples use final checkpoints. The historical Qwen3-1.7B general-evaluation choices are recorded in section 7, separately from EvalPlus results.

## 4. GRPO

### Run training

GRPO is available for math and starts from the recipe's base model, independently of SFT. It supports Qwen3-1.7B and Qwen3-4B, each with `normal` and `pause` methods:

```bash
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method normal
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method pause
bash scripts/train_grpo.sh --recipe qwen3-4b --method normal
bash scripts/train_grpo.sh --recipe qwen3-4b --method pause
```

Defaults use four GPUs for 1.7B and two for 4B. Override with `--num-gpus`; use `--smoke --num-gpus 1` for a one-update check. W&B logging is offline by default; `--wandb` enables online logging with existing credentials.

Custom GRPO data contains a question and its **final ground-truth answer**, without pre-inserted pauses. Example `my_grpo.jsonl` row:

```json
{"question": "What is 2 + 3?", "answer": "5"}
```

```bash
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method pause \
  --dataset ./my_grpo.jsonl --output-dir outputs/grpo/my-math --no-eval
```

For an HF dataset with different columns:

```bash
bash scripts/train_grpo.sh --recipe qwen3-1.7b --method pause \
  --dataset your-org/math-answers --split train \
  --input-key prompt --label-key final_answer \
  --output-dir outputs/grpo/my-hf --no-eval
```

Normal training needs at least a complete rollout batch, 64 prompts by default. `--no-eval` disables the preset's training-time evaluation. Selected training and evaluation rows are saved under the run's `data/` directory.

### Hyperparameters

| Setting | Value |
|---|---|
| Actor learning rate / scheduler / warmup | 5e-7 / `cosine_with_min_lr` / 0.03 |
| Rollout batch / responses per prompt | 64 prompts / 8 responses |
| Training batch / micro train / micro rollout | 128 / 4 / 4 |
| Episodes / update epochs | 1 / 3 optimization epochs per rollout batch |
| Objective | `dr_grpo`, PPO policy loss, clip 0.2 |
| KL | k3 estimator, coefficient 0.001, separate KL loss |
| Prompt / generation length | 1,024 / 2,048 tokens |
| Rollout temperature / top-p | 1.0 / 1.0 |
| Precision / distribution | BF16, ZeRO-2, gradient checkpointing, FlashAttention 2 |
| Optimizer | Adam betas `(0.9, 0.95)`, weight decay 0, max gradient norm 1 |
| Initialization / seed | Mean pause embedding initialization / 42 |

Actor, reference, and vLLM workers are colocated. vLLM uses one engine per GPU, tensor parallel size 1, memory utilization 0.6, eager execution, and sleep mode; DeepSpeed sleep is also enabled. These are full-tuning GRPO presets (`lora_rank: 0`).

| Model / method | GPUs | Packing | Adam offload | Save every | Pause rollout fraction |
|---|---:|---|---|---:|---|
| 1.7B / normal | 4 | Yes | Yes | 10 steps | No insertion |
| 1.7B / pause | 4 | Yes | No | 10 steps | 0.8 → 0.5 over steps 0 → 40 |
| 4B / normal | 2 | No | Yes | 20 steps | No insertion |
| 4B / pause | 2 | No | No | 10 steps | 0.9 → 0.1 over steps 25 → 40 |

Pause fractions are held at the start value before the specified interval, interpolated linearly within it, and held at the end value afterward. Each rollout batch uses a seeded selection with a rounded sample count.

### Objective and pause penalties

1. Generate ordinary responses and correctness rewards without inserted pauses.
2. Select responses according to the rollout-fraction schedule and insert one pause per reasoning step after generation, preserving the original token IDs.
3. Compute old-actor and current-actor log probabilities in the same augmented context. Exclude inserted pause positions from the policy loss and KL, while retaining them in attention for later tokens.
4. Evaluate the frozen reference on the **original, unperturbed context** and align its log probabilities to the original token positions (`pause_reference_mode: original`). This is the retained run configuration; it differs from using a pause-perturbed reference prefix.
5. Form Dr.GRPO advantages by subtracting each prompt group's mean reward, without dividing by the group's standard deviation. Apply the pause method's length/clip penalties afterward, at the last active response token.

Pause penalties use a target response length of 800, coefficient `1e-4` for each excess token, and a clipping penalty of 0.05. Only correct responses receive these penalties. Their multiplier is 0 through step 20, increases linearly to 1 at step 60 (40 ramp steps), and stays at 1 afterward. The normal method does not apply them. Invalid or overlong augmented samples remove the entire prompt group; distributed workers then align their retained group counts.

Training-time evaluation runs every five updates on GSM8K (1,319 questions from `skrishna/gsm8k_only_answer`, `test`) and bundled MATH500 (500). Each benchmark gets a deterministic pass@1 evaluation with one response at temperature 0, followed by a separate pass@4 evaluation with four responses at temperature 0.6. The final math protocol below reports deterministic pass@1 only.

## 5. Checkpoint selection and math evaluation

### Recorded evaluation choices

Use these historical math evaluation points when comparing with the retained experiment choices. Paths use the current output layout; they are locations produced by your training, not downloadable pretrained checkpoints.

| Stage / recipe | Method(s) | Checkpoint under the run directory |
|---|---|---|
| SFT / `qwen3-1.7b` | `none`, `step` | `ckpt/global_step7000_hf` |
| SFT / `llama3.2-1b` | `none`, `step` | `ckpt/global_step6000_hf` |
| SFT / `qwen3-8b` | `none`, `step`, `append`, `random`, `dit` | `ckpt/global_step2000_hf` |
| SFT / `qwen3-4b`, `llama3.2-3b` | All retained variants | Final checkpoint: run directory itself |
| SFT / `qwen3-1.7b`, `llama3.2-1b` | Remaining variants | Final checkpoint: run directory itself |
| SFT / `llama3.1-8b`; `qwen3-8b` unmasked | All Llama variants; `step_unmasked` for Qwen | Final checkpoint: current-preset default, no verified historical selection |
| GRPO / `qwen3-1.7b` | `normal`, `pause` | `ckpt/global_step70_hf` |
| GRPO / `qwen3-4b` | `normal`, `pause` | `final` |

The Qwen3-4B final-checkpoint choice follows the archived math and general evaluation records that agree, rather than the conflicting candidate paths in the working notes. The current data order, custom data size, and packing can change which update counts a new run reaches. If a historical step is unavailable, evaluate an existing saved checkpoint or the final checkpoint and report that choice. The [README](../README.md) uses final checkpoints for the simplest workflow.

Primary SFT math example, using the recorded Qwen3-1.7B step checkpoint:

```bash
CHECKPOINT=outputs/sft/qwen3-1.7b/step/ckpt/global_step7000_hf
bash scripts/evaluate.sh --recipe qwen3-1.7b --stage sft \
  --checkpoint "$CHECKPOINT" --suite math --num-gpus 2
```

Equivalent historical choices for Llama3.2-1B and Qwen3-8B:

```bash
bash scripts/evaluate.sh --recipe llama3.2-1b --stage sft \
  --checkpoint outputs/sft/llama3.2-1b/step/ckpt/global_step6000_hf \
  --suite math --num-gpus 2

bash scripts/evaluate.sh --recipe qwen3-8b --stage sft \
  --checkpoint outputs/sft/qwen3-8b/step/ckpt/global_step2000_hf \
  --suite math --num-gpus 2
```

GRPO examples:

```bash
bash scripts/evaluate.sh --recipe qwen3-1.7b --stage grpo \
  --checkpoint outputs/grpo/qwen3-1.7b/pause/ckpt/global_step70_hf \
  --suite math --num-gpus 2

bash scripts/evaluate.sh --recipe qwen3-4b --stage grpo \
  --checkpoint outputs/grpo/qwen3-4b/pause/final --suite math --num-gpus 2
```

Replace the method directory to evaluate another trained variant. For custom output directories, supply that checkpoint path and its matching recipe. Full HF checkpoints, HF model IDs, and local LoRA adapters are supported. All evaluation suites automatically merge local adapters into a temporary model; source checkpoints remain unchanged and the temporary merged model is removed afterward.

### Math protocol

| Benchmark | Questions | `--benchmarks` name |
|---|---:|---|
| GSM8K | 1,319 | `gsm8k_test` |
| MATH500 | 500 | `math_500_test` |
| MATH | 5,000 | `math_test` |
| AIME24 | 30 | `aime2024_test` |
| AMC23 | 40 | `amc23_test` |
| Minerva | 272 | `minerva_math` |

All six bundled benchmarks run by default. Math uses deterministic pass@1: one response, temperature 0, top-p 1, seed 42, and the retained answer extraction and mathematical correctness checks. No evaluation mode forces pause insertion.

| Setting | SFT checkpoint | GRPO checkpoint |
|---|---|---|
| Prompt / generation limit | 2,048 / 1,024 tokens | 1,024 / 2,048 tokens |
| Repetition penalty | 1.0 | 1.05 |
| Prompt handling | Text prompt with model-family stop rules | Tokenized, truncated prompt; total model length 3,072 |
| Spontaneous pause generation | Allowed | Pause token explicitly banned when present in tokenizer |

Multi-GPU math evaluation shards examples across workers and combines their scores. Outputs include per-benchmark generations, `evaluation_config.json`, `results.json`, and `results.csv`. The default location is `outputs/evaluation/<stage>/<recipe>/<checkpoint-label>/math/`; `--output-dir` chooses another location.

## 6. Code evaluation

Keep the training environment active: it runs model generation and merges local LoRA adapters. Install EvalPlus 0.3.1 in a separate CPU scoring environment without installing this repository there:

```bash
python3.10 -m venv .venv-code-eval
.venv-code-eval/bin/python -m pip install --upgrade pip
.venv-code-eval/bin/python -m pip install evalplus==0.3.1

bash scripts/evaluate.sh --recipe "$RECIPE" --stage sft \
  --checkpoint "outputs/sft/code/$RECIPE/step" --suite code --num-gpus 2 \
  --evalplus-python "$PWD/.venv-code-eval/bin/python"
```

`--evalplus-python` is the absolute Python path for the CPU scorer; it does not change the generation environment. Benchmark data is downloaded automatically. Code evaluation is available for the four code SFT recipes; GRPO remains math-only.

| Benchmark | Problems | Reported metrics |
|---|---:|---|
| HumanEval | 164 | HumanEval pass@1; HumanEval+ pass@1 |
| MBPP | 378 | MBPP pass@1; MBPP+ pass@1 |

The `+` metrics use the expanded EvalPlus tests. A completion must pass all applicable tests to count as correct; the `+` score includes the original tests as well. The evaluator generates one completion per problem with temperature 0, top-p 1, seed 42, a 2,048-token prompt limit, and up to 8,192 new tokens. It uses the model-family code prompt (`qwen2-code` or `llama3-code`) and EvalPlus's benchmark-specific preparation and scoring.

No pause insertion is forced at evaluation time, and spontaneous pause generation is allowed. Raw generations retain these tokens for inspection; pause markers are removed before code extraction and execution. HF checkpoints and local LoRA adapters follow the same reload and merge path as math evaluation.

The default output directory is `outputs/evaluation/sft/<recipe>/<checkpoint-label>/code/`; use `--output-dir` to override it. Outputs retain raw generations, EvalPlus samples and per-problem results, the resolved configuration, and aggregate metrics. `--smoke` evaluates two problems per benchmark with up to 256 new tokens and marks the results as a subset. These examples use final checkpoints and do not claim an exact reconstruction of the paper's selected EvalPlus checkpoints.

## 7. General capability evaluation

Create the separate general-evaluation environment after installing the training environment:

```bash
deactivate
python3.10 -m venv .venv-general
source .venv-general/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/general.txt
python -m pip install -e .
python -m pip check

bash scripts/evaluate.sh --recipe qwen3-1.7b --stage sft \
  --checkpoint outputs/sft/qwen3-1.7b/step/ckpt/global_step7000_hf \
  --suite general --num-gpus 2 \
  --merge-python "$PWD/.venv-train/bin/python"
```

This environment uses LightEval 0.13.0 and vLLM 0.10.1.1. For another checkpoint, use the same recipe, stage, and checkpoint selection as in the math section. `--merge-python` invokes PEFT from the training environment when the input is a local LoRA adapter.

General evaluation also accepts code-trained checkpoints, without a domain flag:

```bash
bash scripts/evaluate.sh --recipe "$RECIPE" --stage sft \
  --checkpoint "outputs/sft/code/$RECIPE/step" --suite general --num-gpus 2 \
  --merge-python "$PWD/.venv-train/bin/python"
```

The archived Qwen3-1.7B code-trained **general-evaluation** results identify these selections:

| Code SFT method | Historical checkpoint |
|---|---|
| `none` | `ckpt/global_step3000_hf` |
| `step` | `ckpt/global_step4000_hf` |
| `append` | `ckpt/global_step4000_hf` |
| `random`, `dit` | `ckpt/global_step2000_hf` |

Their recorded general metrics agree with the paper. These records do not establish which checkpoints produced the EvalPlus code scores, so the table applies only to general evaluation. The public examples use final code checkpoints. Other code models have no verified historical checkpoint mapping here, and the default retention of three intermediate checkpoints may remove an earlier step during a new run.

The task string is `mmlu_mc|4,gpqa:mc|4,hellaswag|4,bigbench_hard|4,piqa_parquet|4`. The suffix `4` means **four-shot context**, not four generated answers or pass@4.

| Task | Retained metric |
|---|---|
| MMLU (`mmlu_mc`, 57 subjects) | Multiple-choice log-likelihood accuracy |
| GPQA (`gpqa:mc`, main subset) | Multiple-choice log-likelihood accuracy |
| HellaSwag | Generative exact match (`em`) |
| BBH (`bigbench_hard`, 18 retained tasks) | Multiple-choice log-likelihood accuracy |
| PIQA (`piqa_parquet`) | Multiple-choice log-likelihood accuracy |

General evaluation uses temperature 0.7, BF16, maximum model length 4,096, seed 1234, GPU memory utilization 0.9, maximum batched tokens 8,192, and maximum sequences 256. It neither forces nor bans pause generation for either stage. `--tasks` can select a different LightEval task string. Reports are written under `outputs/evaluation/<stage>/<recipe>/<checkpoint-label>/general/`, or the supplied output directory.

## 8. Optional GRPO AUC analysis

Switch back to the training environment and recompute the four bundled CSV exports without a W&B account:

```bash
deactivate
source .venv-train/bin/activate
bash scripts/analyze_auc.sh
```

The exports cover training reward over steps 1–78 and GSM8K `eval/gsm8k/pass_at_1` over steps 5–75. The script sorts valid points by step, keeps the last value for duplicate steps, integrates with the trapezoidal rule, and divides by the observed step range for normalized AUC. It writes summary and comparison tables under `outputs/auc/csv_auc_<timestamp>/`.

To retrieve the configured historical runs from W&B instead, using your existing access:

```bash
bash scripts/analyze_auc.sh --source wandb
```

The canonical runs are configured in `configs/evaluation/auc.yaml`; offline CSV analysis remains usable without remote access. Generated data, model checkpoints, and results live under `artifacts/` and `outputs/`. This codebase retains the Apache-2.0 license and attribution of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
