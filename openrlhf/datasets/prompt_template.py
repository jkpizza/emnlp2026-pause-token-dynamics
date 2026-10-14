"""Exact prompts retained for the supported math and code SFT recipes."""

PROMPTS = {'qwen2-boxed-cot': '<|im_start|>system\n'
                    'Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n'
                    '<|im_start|>user\n'
                    '{input}<|im_end|>\n'
                    '<|im_start|>assistant\n',
 'llama3': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'
           '\n'
           'You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n'
           '\n'
           'Solve the following math problem efficiently and clearly:\n'
           '\n'
           '- For simple problems (2 steps or fewer):\n'
           'Provide a concise solution with minimal explanation.\n'
           '\n'
           '- For complex problems (3 steps or more):\n'
           'Use this step-by-step format:\n'
           '\n'
           '## Step 1: [Concise description]\n'
           '[Brief explanation and calculations]\n'
           '\n'
           '## Step 2: [Concise description]\n'
           '[Brief explanation and calculations]\n'
           '\n'
           '...\n'
           '\n'
           'Regardless of the approach, always conclude with:\n'
           '\n'
           'Therefore, the final answer is: $\\boxed{{answer}}$.\n'
           '\n'
           'Where [answer] is just the final number or expression that solves the problem.\n'
           '\n'
           'Problem: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'
           '\n'}

PROMPTS.update({
    "qwen2-code": (
        "<|im_start|>system\nYou are a helpful Python coding assistant. "
        "Reason briefly about the problem and then provide a complete, self-contained Python "
        "solution inside a single ```python ... ``` block.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3-code": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful Python coding assistant. "
        "Reason briefly about the problem and then provide a complete, self-contained Python "
        "solution inside a single ```python ... ``` block.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
})


def get_prompt_template(prompt_type):
    if prompt_type in (None, "None"):
        return None
    if prompt_type not in PROMPTS:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    return PROMPTS[prompt_type]
