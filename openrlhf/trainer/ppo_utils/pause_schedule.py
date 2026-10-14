"""Deterministic interpolation for the recorded post-rollout pause schedules."""


def scheduled_pause_ratio(
    start_probability, end_probability, warmup_start_step, warmup_end_step, global_step
):
    start_probability = float(start_probability)
    end_probability = float(end_probability)
    warmup_start_step = int(warmup_start_step)
    warmup_end_step = int(warmup_end_step)
    global_step = int(global_step)
    if warmup_end_step <= warmup_start_step:
        return end_probability if global_step >= warmup_end_step else start_probability
    if global_step <= warmup_start_step:
        return start_probability
    if global_step >= warmup_end_step:
        return end_probability
    ratio = (global_step - warmup_start_step) / float(
        warmup_end_step - warmup_start_step
    )
    return start_probability + ratio * (end_probability - start_probability)
