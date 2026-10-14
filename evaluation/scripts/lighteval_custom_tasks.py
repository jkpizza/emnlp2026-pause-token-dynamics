"""Project-local LightEval tasks.

The built-in `mmlu` tasks in this LightEval version are generative exact-match
tasks. That is brittle for chat-templated Qwen checkpoints because generations
can include role-token remnants after the answer letter. The `mmlu_mc:*` tasks
keep the same prompts and few-shot splits, but score choices with
loglikelihood accuracy.

The built-in `piqa` task points at `ybisk/piqa`, which is a dataset-script-only
repository. `datasets>=4` refuses to load those scripts, so `piqa_parquet` uses
LightEval's parquet mirror instead. PIQA is a two-choice multiple-choice task,
so this local variant scores the A/B choices with loglikelihood accuracy instead
of brittle generative exact match.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.mmlu import TASKS_TABLE as BUILTIN_MMLU_TASKS_TABLE
from lighteval.tasks.tasks.mmlu import mmlu_prompt
from lighteval.tasks.tasks.piqa import piqa_prompt


def make_mmlu_mc_task(config: LightevalTaskConfig) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"mmlu_mc:{config.hf_subset}",
        prompt_function=mmlu_prompt,
        hf_repo=config.hf_repo,
        hf_subset=config.hf_subset,
        hf_avail_splits=config.hf_avail_splits,
        evaluation_splits=config.evaluation_splits,
        few_shots_split=config.few_shots_split,
        few_shots_select=config.few_shots_select,
        generation_size=1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=config.version,
    )


piqa_parquet = LightevalTaskConfig(
    name="piqa_parquet",
    prompt_function=piqa_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)


mmlu_mc_tasks = [make_mmlu_mc_task(config) for config in BUILTIN_MMLU_TASKS_TABLE]


TASKS_TABLE = [*mmlu_mc_tasks, piqa_parquet]
