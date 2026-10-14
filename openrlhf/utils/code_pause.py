"""Pause placement in fenced code without changing source whitespace."""
from __future__ import annotations

import re


_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\r\n]*)$")


def code_line_starts(response: str) -> list[int]:
    """Return offsets before indentation for nonblank lines inside fences."""
    starts = []
    marker = None
    offset = 0
    for line in response.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        match = _FENCE.match(text)
        if marker is None and match:
            marker = match.group(1)
        elif (marker is not None and match and match.group(1)[0] == marker[0]
              and len(match.group(1)) >= len(marker) and not match.group(2).strip()):
            marker = None
        elif marker is not None and text.strip():
            starts.append(offset)
        offset += len(line)
    if marker is not None:
        raise ValueError("Unclosed fenced code block; solutions must contain complete code fences.")
    return starts


def insert_at_offsets(text: str, offsets: list[int], pause_text: str) -> str:
    """Insert at character offsets, retaining duplicates and every source byte."""
    parts = []
    previous = 0
    for offset in sorted(offsets):
        if not 0 <= offset <= len(text):
            raise ValueError(f"Invalid pause insertion offset: {offset}.")
        parts.extend((text[previous:offset], pause_text))
        previous = offset
    parts.append(text[previous:])
    return "".join(parts)


def insert_pauses_before_code_lines(response: str, pause_text: str) -> tuple[str, int]:
    starts = code_line_starts(response)
    return insert_at_offsets(response, starts, pause_text), len(starts)


def insert_at_token_boundaries(response: str, boundaries: list[int], tokenizer, pause_text: str) -> str:
    """Use token offsets so a boundary inside a multibyte character stays lossless."""
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    token_offsets = encoded["offset_mapping"]
    offsets = []
    for boundary in boundaries:
        if not 0 <= boundary <= len(token_offsets):
            raise ValueError(f"Invalid code token boundary: {boundary}.")
        offsets.append(len(response) if boundary == len(token_offsets) else token_offsets[boundary][0])
    return insert_at_offsets(response, offsets, pause_text)
