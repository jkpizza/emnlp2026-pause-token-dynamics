"""Prompt strings used by the paper math experiments."""

PROMPTS = {'qwen2-boxed-cot': '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n', 'llama3': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{{answer}}$.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'}

# Coding prompts from the original SFT and benchmark protocol.
PROMPTS['qwen2-code'] = '<|im_start|>system\nYou are a helpful Python coding assistant. Reason briefly about the problem and then provide a complete, self-contained Python solution inside a single ```python ... ``` block.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
PROMPTS['llama3-code'] = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful Python coding assistant. Reason briefly about the problem and then provide a complete, self-contained Python solution inside a single ```python ... ``` block.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

def get_prompt_template(prompt_type):
    if prompt_type in (None, "None"):
        return None
    try:
        return PROMPTS[prompt_type]
    except KeyError:
        raise ValueError(f"Unsupported prompt type: {prompt_type}") from None
