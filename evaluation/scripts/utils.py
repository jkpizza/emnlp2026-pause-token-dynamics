from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    torch = None
    F = None
    AutoTokenizer = None



def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    if AutoTokenizer is None:
        raise ModuleNotFoundError("get_tokenizer requires transformers and torch to be installed.")
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch



def extract_first_numeric_answer(text):
    matches = []

    # pattern 1: \(\\boxed{ANSWER}\)
    match1 = re.search(r'\\\(\\boxed\{(.*?)\}\\\)', text)
    if match1:
        matches.append(('boxed', match1.start(), match1.group(1).strip()))

    # pattern 2: Therefore, the answer is: ANSWER.
    match2 = re.search(r'Therefore, the answer is: ([^\.\n]+)', text)
    if match2:
        matches.append(('therefore', match2.start(), match2.group(1).strip()))

    if not matches:
        return None

    # extract first answer
    first = min(matches, key=lambda x: x[1])
    answer_text = first[2]

    # float / int extract
    num_match = re.search(r'\d+(?:\.\d+)?', answer_text)
    return float(num_match.group()) if num_match else None


# match answer include non-numeric answer

def extract_last_answer(text: str, answer_trigger: str):
    """
    Extracts the last expression following the answer_trigger in the text.
    """
    # 2. Fallback to \\boxed{...} with nested {} support
    # Match \boxed{...} where ... may include nested {...}
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    if matches_boxed:
        # Unwrap double braces if present (e.g., {{...}})
        boxed_content = matches_boxed[-1].strip()
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        return boxed_content

    pattern = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches = re.findall(pattern, text)
    if matches:
        last_answer = matches[-1].strip()
        if last_answer:
            return last_answer


    # Nothing found
    return None
import re
from typing import List, Optional, Pattern, Union
def extract_all_answers_deprecated(text: str, answer_trigger: str, extra_patterns: Optional[List[Union[str, Pattern[str]]]] = None,):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # 2. Find all \boxed{...} matches with nested {} support
    pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    matches_boxed = re.findall(pattern_boxed, text)
    for boxed_content in matches_boxed:
        boxed_content = boxed_content.strip()
        # Unwrap double braces if present (e.g., {{...}})
        if boxed_content.startswith('{') and boxed_content.endswith('}'):
            boxed_content = boxed_content[1:-1].strip()
        if boxed_content:
            results.append(boxed_content)

    # 3. Find markdown-style #### <answer>
    pattern_markdown = r'####\s*<answer>\s*([^\n\.]+?)[\n\.]'
    matches_markdown = re.findall(pattern_markdown, text)
    for match in matches_markdown:
        val = match.strip()
        if val:
            results.append(val)

    # 3. Find markdown-style #### <answer>
    pattern_markdown = r'(?m)^\s*####\s*["\']?(.+?)["\']?\s*(?:\n|$)'
    matches_markdown = re.findall(pattern_markdown, text)
    for match in matches_markdown:
        val = match.strip()
        if val:
            results.append(val)

    # 4. Add any additional patterns
    if extra_patterns:
        for pattern in extra_patterns:
            compiled = re.compile(pattern, re.MULTILINE) if isinstance(pattern, str) else pattern
            matches = compiled.findall(text)
            for match in matches:
                val = match.strip()
                if val:
                    results.append(val)
    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None

import re
from typing import List, Optional, Pattern, Union, Tuple

BOXED_TOKEN_RE = re.compile(r'\\\\boxed\{|\\boxed\{')  # matches \\boxed{ or \boxed{


def strip_latent_markers(text: str) -> str:
    text = re.sub(r"<\|eol\|>", "\n", str(text))
    text = re.sub(r"<\|bol\|>", "\n", text)
    text = re.sub(r"<\|LATENT_\d+\|>", " ", text)
    text = re.sub(r"<\|placeholder\|>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_special_tokens(text: str) -> str:
    return strip_latent_markers(text)


def _extract_boxed_candidates(text: str) -> List[Tuple[int, int, str, bool]]:
    """
    Returns boxed candidates as:
        (content_start, content_end, content, has_inner_boxed)
    """
    candidates = []

    for m in BOXED_TOKEN_RE.finditer(text):
        content_start = m.end()
        depth = 1
        i = content_start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth != 0:
            continue

        boxed_content = text[content_start:i - 1].strip()

        # unwrap one extra outer brace if needed: {{...}} -> {...}
        if boxed_content.startswith("{") and boxed_content.endswith("}"):
            boxed_content = boxed_content[1:-1].strip()

        has_inner_boxed = bool(BOXED_TOKEN_RE.search(boxed_content))
        candidates.append((content_start, i - 1, boxed_content, has_inner_boxed))

    return candidates


def extract_all_answers(
    text: str,
    answer_trigger: str,
    extra_patterns: Optional[List[Union[str, Pattern[str]]]] = None,
    dedup: bool = True,
) -> Optional[List[str]]:
    """
    Extract all parsed answer candidates in text order.

    Priority is no longer used to select a single answer.
    Instead, every parsed candidate is returned.

    Sources:
    1) \\boxed{...}, \boxed{...}   (outer suspicious boxes with inner boxed are skipped)
    2) text after `answer_trigger`
    3) markdown-style #### <answer> ...
    4) markdown-style #### ...
    5) extra_patterns

    Args:
        text: input text
        answer_trigger: e.g. "Therefore, the answer is:"
        extra_patterns: extra regex patterns
        dedup: if True, remove duplicate answers while preserving order

    Returns:
        List of parsed answers in appearance order, or None if nothing matched.
    """
    text = _normalize_special_tokens(text)
    candidates = []

    def add_candidate(source: str, value: str, start: int, end: int):
        value = value.strip()
        if value:
            candidates.append((start, end, source, value))

    # ------------------------------------------------------------
    # 1) boxed candidates
    # ------------------------------------------------------------
    boxed_candidates = _extract_boxed_candidates(text)
    for start, end, value, has_inner_boxed in boxed_candidates:
        # skip suspicious outer boxed like \boxed{ ... \boxed{540} ... }
        if has_inner_boxed:
            continue
        add_candidate("boxed", value, start, end)

    # ------------------------------------------------------------
    # 2) answer_trigger candidates
    # ------------------------------------------------------------
    pattern_trigger = re.compile(
        re.escape(answer_trigger) + r"\s*['\"]?(.+?)['\"]?(?:\n|$)"
    )
    for m in pattern_trigger.finditer(text):
        value = m.group(1).strip()
        # optional heuristic: avoid huge reasoning captures
        if len(value) <= 200:
            add_candidate("trigger", value, m.start(1), m.end(1))

    # ------------------------------------------------------------
    # 3) #### <answer> ...
    # ------------------------------------------------------------
    pattern_markdown_answer = re.compile(
        r"####\s*<answer>\s*(.+?)(?:\n|$)"
    )
    for m in pattern_markdown_answer.finditer(text):
        add_candidate("markdown_answer", m.group(1), m.start(1), m.end(1))

    # ------------------------------------------------------------
    # 4) #### ...
    # ------------------------------------------------------------
    pattern_markdown_header = re.compile(
        r'(?m)^\s*####\s*["\']?(.+?)["\']?\s*(?:\n|$)'
    )
    for m in pattern_markdown_header.finditer(text):
        add_candidate("markdown_header", m.group(1), m.start(1), m.end(1))

    # ------------------------------------------------------------
    # 5) extra patterns
    # ------------------------------------------------------------
    if extra_patterns:
        for pattern in extra_patterns:
            compiled = re.compile(pattern, re.MULTILINE) if isinstance(pattern, str) else pattern

            for m in compiled.finditer(text):
                if m.groups():
                    val = next((g for g in m.groups() if g), "")
                    if val:
                        for gi, g in enumerate(m.groups(), start=1):
                            if g == val:
                                add_candidate("extra", val, m.start(gi), m.end(gi))
                                break
                else:
                    add_candidate("extra", m.group(0), m.start(0), m.end(0))

    if not candidates:
        return None

    # appearance order
    candidates.sort(key=lambda x: (x[1], x[0]))

    answers = [c[3] for c in candidates]

    if dedup:
        seen = set()
        deduped = []
        for ans in answers:
            if ans not in seen:
                seen.add(ans)
                deduped.append(ans)
        answers = deduped

    return answers


# def extract_all_answers(
#     text: str,
#     answer_trigger: str,
#     extra_patterns: Optional[List[Union[str, Pattern[str]]]] = None,
# ) -> Optional[str]:
#     """
#     Extract the final answer candidate appearing latest in the text.

#     Supported sources:
#     1) text after `answer_trigger`
#     2) \\boxed{...}
#     3) markdown-style #### <answer> ...
#     4) markdown-style #### ...
#     5) any user-provided extra_patterns

#     Returns:
#         The last answer candidate found in the text, or None if nothing matched.
#     """
#     candidates = []

#     def add_candidate(value: str, start: int, end: int):
#         value = value.strip()
#         if value:
#             candidates.append((start, end, value))

#     # ------------------------------------------------------------
#     # 1) Matches after answer_trigger
#     # ------------------------------------------------------------
#     pattern_trigger = re.compile(
#         re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
#     )
#     for m in pattern_trigger.finditer(text):
#         add_candidate(m.group(1), m.start(1), m.end(1))

#     # ------------------------------------------------------------
#     # 2) Matches for \boxed{...} with nested brace support
#     #    (manual parser instead of regex)
#     # ------------------------------------------------------------
#     boxed_token = r"\boxed{"
#     idx = 0
#     while True:
#         start = text.find(boxed_token, idx)
#         if start == -1:
#             break

#         content_start = start + len(boxed_token)
#         depth = 1
#         i = content_start

#         while i < len(text) and depth > 0:
#             if text[i] == "{":
#                 depth += 1
#             elif text[i] == "}":
#                 depth -= 1
#             i += 1

#         if depth == 0:
#             boxed_content = text[content_start:i - 1].strip()

#             # Unwrap one extra outer brace if present: {{...}} -> {...}
#             if boxed_content.startswith("{") and boxed_content.endswith("}"):
#                 boxed_content = boxed_content[1:-1].strip()

#             add_candidate(boxed_content, content_start, i - 1)
#             idx = i
#         else:
#             # unmatched brace -> stop parsing boxed
#             break

#     # ------------------------------------------------------------
#     # 3) markdown-style: #### <answer> ...
#     # ------------------------------------------------------------
#     pattern_markdown_answer = re.compile(
#         r"####\s*<answer>\s*([^\n\.]+?)(?:[\n\.]|$)"
#     )
#     for m in pattern_markdown_answer.finditer(text):
#         add_candidate(m.group(1), m.start(1), m.end(1))

#     # ------------------------------------------------------------
#     # 4) markdown-style: #### ...
#     # ------------------------------------------------------------
#     pattern_markdown_header = re.compile(
#         r'(?m)^\s*####\s*["\']?(.+?)["\']?\s*(?:\n|$)'
#     )
#     for m in pattern_markdown_header.finditer(text):
#         add_candidate(m.group(1), m.start(1), m.end(1))

#     # ------------------------------------------------------------
#     # 5) extra patterns
#     #    - if pattern has capture groups, use the first non-empty group
#     #    - otherwise use the whole match
#     # ------------------------------------------------------------
#     if extra_patterns:
#         for pattern in extra_patterns:
#             compiled = re.compile(pattern, re.MULTILINE) if isinstance(pattern, str) else pattern

#             for m in compiled.finditer(text):
#                 if m.groups():
#                     val = next((g for g in m.groups() if g), "")
#                     if val:
#                         # locate the chosen group's span
#                         for gi, g in enumerate(m.groups(), start=1):
#                             if g == val:
#                                 add_candidate(val, m.start(gi), m.end(gi))
#                                 break
#                 else:
#                     add_candidate(m.group(0), m.start(0), m.end(0))

#     if not candidates:
#         return None

#     # Return the candidate that appears latest in the text
#     candidates.sort(key=lambda x: (x[1], x[0]))
#     return [candidates[-1][2]]

def extract_all_answers_not_boxed(text: str, answer_trigger: str):
    """
    Extracts all expressions following the answer_trigger and all \boxed{...} contents.
    Returns a list of all matches.
    """
    results = []

    # 1. Find all matches after answer_trigger
    pattern_trigger = re.escape(answer_trigger) + r"\s*['\"]?([^\n\.]+?)['\"]?(?:[\n\.]|$)"
    matches_trigger = re.findall(pattern_trigger, text)
    for match in matches_trigger:
        val = match.strip()
        if val:
            results.append(val)

    # # 2. Find all \boxed{...} matches with nested {} support
    # pattern_boxed = r'\\boxed\s*{((?:[^{}]|{[^{}]*})+)}'
    # matches_boxed = re.findall(pattern_boxed, text)
    # for boxed_content in matches_boxed:
    #     boxed_content = boxed_content.strip()
    #     # Unwrap double braces if present (e.g., {{...}})
    #     if boxed_content.startswith('{') and boxed_content.endswith('}'):
    #         boxed_content = boxed_content[1:-1].strip()
    #     if boxed_content:
    #         results.append(boxed_content)

    # 3. Return unique results while preserving order
    seen = set()
    unique_results = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique_results.append(r)

    return unique_results if unique_results else None




# def normalize_expression(expr: str) -> str:
#     """
#     Normalize expressions by removing whitespace and wrapping quotes.
#     """
#     if expr is None:
#         return None
#     return re.sub(r'\s+', '', expr).strip("'\"")


def try_parse_float(expr: str):
    """
    Tries to convert an expression to float. Returns None if not possible.
    """
    try:
        return float(expr)
    except ValueError:
        return None


# def is_answer_equal(predicted: str, target: str) -> bool:
#     import math
#     """
#     Compares predicted and target answers:
#     - If both are numeric, compare numerically
#     - Otherwise, compare after normalizing expressions
#     """
#     pred_norm = normalize_expression(predicted)
#     target_norm = normalize_expression(target)

#     pred_num = try_parse_float(pred_norm)
#     target_num = try_parse_float(target_norm)

#     if pred_num is not None and target_num is not None:
#         return math.isclose(pred_num, target_num, rel_tol=1e-6)

#     return pred_norm == target_norm


#####################
## New code for Answer equal
#####################
import re
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional, Set


_QUOTE_STRIP = "'\"`"
_GROUP_SEPARATORS = {" ", "\u00A0", "\u202F", "\u2009", "_", "'", "’"}

# pattern to extract "token" containing numbers
# example: "$7,200", "(1.234,56)", "€77", "- 7,200"

_NUMERIC_TOKEN_RE = re.compile(
    r"""
    (?<!\w)
    [\(\[]?
    \s*
    [+\-−–—]?
    \s*
    (?:[^\w\s]\s*){0,2}
    (?:
        \d{1,3}(?:[ ,._'’\u00A0\u202F\u2009]\d{3})+(?:[.,]\d+)? |
        \d+(?:[.,]\d+)? |
        [.,]\d+
    )
    \s*
    %?
    \s*
    [\)\]]?
    (?!\w)
    """,
    re.VERBOSE,
)


def normalize_expression(expr: Optional[str]) -> Optional[str]:
    """
    basic normalization:
    - Unicode normalization
    - remove whitespace
    - remove outer quotes
    """
    if expr is None:
        return None
    expr = unicodedata.normalize("NFKC", str(expr))
    expr = re.sub(r"\s+", "", expr)
    return expr.strip(_QUOTE_STRIP)


def try_parse_float(expr: Optional[str]):
    """
    for strict numeric comparison.
    """
    try:
        return float(expr)
    except (TypeError, ValueError):
        return None


def _pre_normalize(expr: Optional[str]) -> Optional[str]:
    """
    before relaxed parsing:
    - Unicode normalization
    - unescape LaTeX/escape special characters (e.g. \\$ -> $, \\% -> %)
    - Unicode minus normalization
    """
    if expr is None:
        return None

    s = unicodedata.normalize("NFKC", str(expr))
    s = re.sub(r"\\([^\w\s])", r"\1", s)  # \$ -> $, \% -> %, \( -> (
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s.strip()


def _remove_currency_symbols(s: str) -> str:
    """
    remove Unicode category 'Sc' (currency symbol).
    $ , €, £, ¥, ₩ etc. are mostly covered automatically.
    """
    return "".join(ch for ch in s if unicodedata.category(ch) != "Sc")


def _decimal_to_canonical_str(d: Decimal) -> str:
    """
    convert Decimal to canonical string for comparison.
    7200.0 -> "7200"
    1.2300 -> "1.23"
    """
    s = format(d.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _add_numeric_candidate(core: str, sign: int, out: Set[str]) -> None:
    """
    add to candidates only if core is actually a number.
    """
    if not core:
        return

    if not re.fullmatch(r"(?:\d+(?:\.\d+)?|\.\d+)", core):
        return

    if sign < 0:
        core = "-" + core

    try:
        out.add(_decimal_to_canonical_str(Decimal(core)))
    except InvalidOperation:
        return


def _canonicalize_numeric_candidates(
    expr: Optional[str],
    *,
    allow_percent: bool = False,
    allow_ambiguous_grouping: bool = True,
) -> Set[str]:
    """
    convert a string to 'numeric candidates'.

    examples:
    - "\\$7,200" -> {"7200"}
    - "€77" -> {"77"}
    - "1,234.56" -> {"1234.56"}
    - "1.234,56" -> {"1234.56"}
    - "(1,234.50)" -> {"-1234.5"}

    if allow_ambiguous_grouping=True,
    - "1,234" -> {"1.234", "1234"}
    - "1.234" -> {"1.234", "1234"}
    e.g. ambiguous cases are both added as candidates.
    """
    s = _pre_normalize(expr)
    if not s:
        return set()

    s = s.strip().strip(_QUOTE_STRIP)
    s = _remove_currency_symbols(s).strip()

    # handle cases like outer formula delimiter: $7200$, `7200`
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"$", "`"} and any(ch.isdigit() for ch in s[1:-1]):
        s = _remove_currency_symbols(s[1:-1]).strip()

    sign = 1

    # handle cases like accounting negative: (1234) -> -1234
    if s.startswith("(") and s.endswith(")") and any(ch.isdigit() for ch in s[1:-1]):
        sign *= -1
        s = s[1:-1].strip()

    if allow_percent and s.endswith("%"):
        s = s[:-1].strip()

    # remove whitespace/underscore/apostrophe grouping separators
    for sep in _GROUP_SEPARATORS:
        s = s.replace(sep, "")

    if not s:
        return set()

    if s[0] in "+-":
        if s[0] == "-":
            sign *= -1
        s = s[1:]

    if not s:
        return set()

    # do not remove character units carelessly: avoid false positives like 77kg vs 77
    if re.search(r"[A-Za-z]", s):
        return set()

    out: Set[str] = set()

    # if both comma and dot are present, consider the last separator as decimal mark
    # both 1,234.56 and 1.234,56 are handled
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            core = s.replace(".", "").replace(",", ".")
        else:
            core = s.replace(",", "")
        _add_numeric_candidate(core, sign, out)
        return out

    # only comma is present
    if "," in s:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
            # convert 1,234,567 -> 1234567
            _add_numeric_candidate(s.replace(",", ""), sign, out)
        elif re.fullmatch(r"\d+,\d+", s):
            # convert decimal comma
            _add_numeric_candidate(s.replace(",", "."), sign, out)

            # relaxed fallback: 1,234 ambiguous cases are also added as candidates
            if allow_ambiguous_grouping and re.fullmatch(r"\d+,\d{3}", s):
                _add_numeric_candidate(s.replace(",", ""), sign, out)
        else:
            return set()
        return out

    # only dot is present
    if "." in s:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            # 1.234.567 -> 1234567
            _add_numeric_candidate(s.replace(".", ""), sign, out)
        elif re.fullmatch(r"\d+\.\d+", s):
            # decimal dot
            _add_numeric_candidate(s, sign, out)

            # relaxed fallback: 1.234 ambiguous cases are also added as candidates
            if allow_ambiguous_grouping and re.fullmatch(r"\d+\.\d{3}", s):
                _add_numeric_candidate(s.replace(".", ""), sign, out)
        else:
            return set()
        return out

    # pure integer
    _add_numeric_candidate(s, sign, out)
    return out


def _extract_numeric_candidates(
    expr: Optional[str],
    *,
    allow_percent: bool = False,
    allow_ambiguous_grouping: bool = True,
) -> Set[str]:
    """
    even if the entire sentence is not a number, extract numeric tokens inside and create candidates.
    example: "The answer is $7,200." -> {"7200"}
    """
    s = _pre_normalize(expr)
    if not s:
        return set()

    out: Set[str] = set()

    for m in _NUMERIC_TOKEN_RE.finditer(s):
        token = m.group(0).strip()
        out |= _canonicalize_numeric_candidates(
            token,
            allow_percent=allow_percent,
            allow_ambiguous_grouping=allow_ambiguous_grouping,
        )

    return out


def _select_numeric_candidates(
    expr: Optional[str],
    *,
    allow_percent: bool = False,
    allow_ambiguous_grouping: bool = True,
    aggressive_numeric_match: bool = False,
) -> Set[Decimal]:
    """
    select numeric candidates for comparison.

    default is conservative:
    1) if the entire string is interpreted as a number, use that candidate
    2) otherwise, extract numeric tokens inside
    3) only select if the internal token is 'unique'

    if aggressive_numeric_match=True,
    any of the multiple numeric candidates in the sentence can match.
    """
    full = _canonicalize_numeric_candidates(
        expr,
        allow_percent=allow_percent,
        allow_ambiguous_grouping=allow_ambiguous_grouping,
    )
    if full:
        return {Decimal(x) for x in full}

    extracted = _extract_numeric_candidates(
        expr,
        allow_percent=allow_percent,
        allow_ambiguous_grouping=allow_ambiguous_grouping,
    )

    if aggressive_numeric_match or len(extracted) == 1:
        return {Decimal(x) for x in extracted}

    return set()


def _decimals_close(
    a: Decimal,
    b: Decimal,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 0.0,
) -> bool:
    rel_tol_d = Decimal(str(rel_tol))
    abs_tol_d = Decimal(str(abs_tol))
    diff = abs(a - b)
    bound = max(abs_tol_d, rel_tol_d * max(abs(a), abs(b)))
    return diff <= bound


def is_answer_equal(
    predicted: Optional[str],
    target: Optional[str],
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 0.0,
    allow_percent: bool = False,
    allow_ambiguous_grouping: bool = True,
    aggressive_numeric_match: bool = False,
) -> bool:
    """
    comparison order:
    1) strict string comparison
    2) strict numeric comparison
    3) relaxed numeric candidates comparison
    """
    pred_norm = normalize_expression(predicted)
    target_norm = normalize_expression(target)

    # 1) strict string equality
    if pred_norm == target_norm:
        return True

    # 2) strict numeric equality
    pred_num = try_parse_float(pred_norm)
    target_num = try_parse_float(target_norm)
    if pred_num is not None and target_num is not None:
        if math.isclose(pred_num, target_num, rel_tol=rel_tol, abs_tol=abs_tol):
            return True

    # 3) relaxed numeric candidates
    pred_nums = _select_numeric_candidates(
        predicted,
        allow_percent=allow_percent,
        allow_ambiguous_grouping=allow_ambiguous_grouping,
        aggressive_numeric_match=aggressive_numeric_match,
    )
    target_nums = _select_numeric_candidates(
        target,
        allow_percent=allow_percent,
        allow_ambiguous_grouping=allow_ambiguous_grouping,
        aggressive_numeric_match=aggressive_numeric_match,
    )

    for p in pred_nums:
        for t in target_nums:
            if _decimals_close(p, t, rel_tol=rel_tol, abs_tol=abs_tol):
                return True

    return False

def match_with_answer_labels_v2(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                predicted_answer = extract_last_answer(output, answer_trigger)
                if predicted_answer is None:
                    continue
                is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1

        return correct_count/valid_count if valid_count > 0 else 0

def match_with_answer_labels_v3(tokenized_output, answers, answer_trigger):
        correct_count = 0
        valid_count = 0
        extracted_answer = None
        for output, answer in zip(tokenized_output, answers):
            if answer is not None:
                # predicted_answer = self.extract_answer(output, answer_trigger)
                # predicted_answer = extract_last_answer(output, answer_trigger)
                predicted_answers = extract_all_answers(output, answer_trigger)
                if predicted_answers is None:
                    continue
                is_correct = False
                # Check if any of the predicted answers match the target answer
                for predicted_answer in predicted_answers:
                    if predicted_answer is None:
                        continue
                    if is_answer_equal(predicted_answer, answer):
                        is_correct = True
                        extracted_answer = predicted_answer
                        break
                # is_correct = is_answer_equal(predicted_answer, answer)
                correct_count += is_correct
                valid_count += 1

        accuracy = correct_count/valid_count if valid_count > 0 else 0
        return accuracy


def match_with_answer_labels_v4(tokenized_output, answers, answer_trigger):
    from evaluation.scripts.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            output = strip_latent_markers(output)
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            if type(predicted_answers) != list:
                predicted_answers = [predicted_answers]
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct


    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy


def match_with_answer_labels_not_boxed(tokenized_output, answers, answer_trigger):
    from evaluation.scripts.eval_utils import math_equal

    correct_count = 0
    valid_count = 0
    extracted_answer = None
    for output, answer in zip(tokenized_output, answers):
        if answer is not None:
            valid_count += 1
            output = strip_latent_markers(output)
            # predicted_answer = self.extract_answer(output, answer_trigger)
            # predicted_answer = extract_last_answer(output, answer_trigger)
            predicted_answers = extract_all_answers_not_boxed(output, answer_trigger)
            if predicted_answers is None:
                continue
            is_correct = False
            # Check if any of the predicted answers match the target answer
            for predicted_answer in predicted_answers:
                if predicted_answer is None:
                    continue
                if is_answer_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
                elif math_equal(predicted_answer, answer):
                    is_correct = True
                    extracted_answer = predicted_answer
                    break
            # is_correct = is_answer_equal(predicted_answer, answer)
            correct_count += is_correct


    accuracy = correct_count/valid_count if valid_count > 0 else 0
    return accuracy
