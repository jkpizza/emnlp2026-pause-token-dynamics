from openrlhf.datasets.sft_dataset import SFTDataset


def _single_token_id(tokenizer, token):
    if tokenizer is None:
        return None
    converted = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if converted is None or converted == unk_id:
        return None
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def shift_loss_mask_to_target_positions(loss_mask):
    target_loss_mask = loss_mask.new_zeros(loss_mask.shape)
    if loss_mask.size(-1) > 1:
        target_loss_mask[..., 1:] = loss_mask[..., :-1]
    return target_loss_mask


def mask_pause_token_targets(input_ids, loss_mask, pause_token_id, predict_pause_tokens=False):
    target_loss_mask = shift_loss_mask_to_target_positions(loss_mask)
    if predict_pause_tokens:
        return target_loss_mask
    if pause_token_id is None:
        return target_loss_mask

    return target_loss_mask.masked_fill(input_ids.eq(int(pause_token_id)), 0.0)


def strip_trailing_pause_tokens(text, pause_token):
    if not pause_token or not isinstance(text, str):
        return text, ""

    pause_suffix = ""
    while text.endswith(pause_token):
        text = text[: -len(pause_token)]
        pause_suffix += pause_token
    return text, pause_suffix


class PauseSFTDataset(SFTDataset):
    """
    SFT dataset variant for pause-token experiments.

    For append-style pause datasets, the pause token is stored at the end of
    the question field before template formatting. Move that trailing pause
    suffix to the beginning of the response so the assistant sees it after the
    prompt/chat template, not inside the user message.
    """

    def __init__(self, dataset, tokenizer, max_length, strategy, *args, **kwargs):
        self.pause_token = getattr(strategy.args, "pause_token", "<|PAUSE|>")
        self.predict_pause_tokens = bool(getattr(strategy.args, "predict_pause_tokens", False))
        self.pause_token_id = _single_token_id(tokenizer, self.pause_token)
        if self.pause_token_id is None:
            raise ValueError(f"Pause token {self.pause_token!r} must exist and tokenize to exactly one id.")
        super().__init__(dataset, tokenizer, max_length, strategy, *args, **kwargs)

    def process_data(self, data):
        if self.pretrain_mode or self.multiturn:
            return super().process_data(data)

        data = dict(data)
        pause_token = getattr(self.strategy.args, "pause_token", "<|PAUSE|>")
        question = data.get(self.input_key)
        data[self.input_key], pause_suffix = strip_trailing_pause_tokens(question, pause_token)

        processed = super().process_data(data)
        if pause_suffix and processed["prompt"] is not None:
            processed["response"] = pause_suffix + processed["response"]
        return processed

    def get_loss_mask(self, input_ids, idx):
        loss_mask = super().get_loss_mask(input_ids, idx)
        return mask_pause_token_targets(
            input_ids,
            loss_mask,
            self.pause_token_id,
            predict_pause_tokens=getattr(self, "predict_pause_tokens", False),
        )
