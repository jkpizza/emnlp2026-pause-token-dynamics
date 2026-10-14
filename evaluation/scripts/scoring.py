"""Historical math scoring shared by SFT and GRPO benchmark evaluation."""
from evaluation.scripts.eval_utils import math_equal
from evaluation.scripts.utils import extract_all_answers, extract_last_answer, is_answer_equal, strip_latent_markers

def _extract_answer_candidates(generation: str, answer_trigger: str):
    answers = extract_all_answers(generation, answer_trigger)
    if answers is None:
        return []
    if isinstance(answers, list):
        return answers
    return [answers]

def _answers_equal(predicted_answer, answer) -> bool:
    if predicted_answer is None or answer is None:
        return False
    if is_answer_equal(predicted_answer, answer):
        return True
    try:
        return math_equal(predicted_answer, answer)
    except Exception:
        return False

def _generation_result(generation: str, answer, answer_trigger: str):
    clean_generation = strip_latent_markers(generation)
    extracted_all = _extract_answer_candidates(clean_generation, answer_trigger)
    vote_answer = extracted_all[-1] if extracted_all else None
    is_correct = any(_answers_equal(predicted_answer, answer) for predicted_answer in extracted_all)
    return {
        "extracted_last_answers": extract_last_answer(clean_generation, answer_trigger),
        "extracted_all_answers": extracted_all if extracted_all else None,
        "vote_answer": vote_answer,
        "is_correct": bool(is_correct),
    }

def score_generation_group(generations, answer, answer_trigger: str, eval_method: str):
    generation_results = [
        _generation_result(generation, answer, answer_trigger) for generation in generations
    ]

    if eval_method == "pass":
        selected_answer = None
        is_correct = any(result["is_correct"] for result in generation_results)
        for result in generation_results:
            if result["is_correct"] and result["vote_answer"] is not None:
                selected_answer = result["vote_answer"]
                break
        if selected_answer is None:
            selected_answer = next(
                (result["vote_answer"] for result in generation_results if result["vote_answer"] is not None),
                None,
            )
        return {
            "is_correct": bool(is_correct),
            "selected_answer": selected_answer,
            "generation_results": generation_results,
            "vote_counts": [],
        }

    raise ValueError(f"Unsupported eval_method: {eval_method}")
