import re
from typing import List, Tuple


MIN_CHARS = 24
MAX_CHARS = 260
MAX_SENTENCES_PER_STEP = 2

LATENT_TAG_RE = re.compile(r"<\|[^|]+?\|>")

BEGIN_ENV_RE = re.compile(
    r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
)
END_ENV_RE = re.compile(
    r"\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
)

BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\.]\s+|\d+\)\s+)")
FINAL_ANSWER_RE = re.compile(r"^\s*#{3,}\s*\S.*$")

SENTENCE_BOUNDARY_RE = re.compile(
    r'(?<=[.!?])\s+(?=(?:["\']?[A-Z]|Let\b|Suppose\b|Assume\b|If\b|Then\b|Thus\b|Therefore\b|Hence\b|So\b|Now\b|Next\b|Finally\b|The\b|We\b|Since\b|Because\b|Subtracting\b|Adding\b|Dividing\b|Multiplying\b|Simplifying\b|Rearranging\b|Substituting\b|Applying\b|Using\b|Calculating\b|Computing\b|Testing\b|Checking\b|Conclusion\b))'
)

DISCOURSE_CUE_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    (
        let(?:'s)?\b
        |suppose\b
        |assume\b
        |if\b
        |then\b
        |thus\b
        |therefore\b
        |hence\b
        |so\b
        |now\b
        |next\b
        |finally\b
        |first\b
        |second\b
        |third\b
        |conclusion\b
        |the\ expression\b
        |the\ difference\b
        |the\ equation\b
        |the\ inequality\b
        |the\ problem\ requires\b
        |we\ have\b
        |we\ get\b
        |this\ means\b
        |this\ gives\b
        |substituting\b
        |subtracting\b
        |adding\b
        |dividing\b
        |multiplying\b
        |simplifying\b
        |rearranging\b
        |solving\b
        |applying\b
        |using\b
        |calculate\b
        |calculating\b
        |compute\b
        |computing\b
        |considering\b
        |testing\b
        |checking\b
        |observe\b
        |note\ that\b
        |apply\b
        |substitute\b
        |rewrite\b
        |rewriting\b
        |factor\b
        |factoring\b
        |expand\b
        |expanding\b
        |cancel\b
        |canceling\b
        |equate\b
        |comparing\b
        |from\ this\b
        |it\ follows\ that\b
        |observe\ that\b
    )
    \b
    """
)

WEAK_FRAGMENT_RE = re.compile(
    r"""(?ix)
    ^
    \s*
    (
        thus[,:\s]*|
        therefore[,:\s]*|
        hence[,:\s]*|
        so[,:\s]*|
        now[,:\s]*|
        next[,:\s]*|
        finally[,:\s]*|
        then[,:\s]*|
        and[,:\s]*|
        but[,:\s]*|
        or[,:\s]*|
        because[,:\s]*
    )
    $
    """
)

# GSM8K류 <<...>> 계산식, 일반 등호/부등호, 사칙연산이 있는 줄을 step 후보로 본다.
CALCULATION_RE = re.compile(
    r"""(?x)
    (
        <<[^>\n]+>>
        |
        (?:\d|[A-Za-z\)\]\}])\s*(?:=|≈|≃|≤|≥|<|>|\\leq|\\geq)\s*(?:[-+]?\d|[A-Za-z\\(])
        |
        [-+]?\d+(?:\.\d+)?\s*(?:\+|-|\*|/|×|÷|\\times|\\cdot|\\div)\s*[-+]?\d
    )
    """
)

LOWERCASE_CONTINUATION_RE = re.compile(
    r"^\s*(?:and|or|but|because|since|that|which|where|when|while|until|as|to|with|without|of|for|from|in|on|by|then)\b"
)
SYMBOL_CONTINUATION_RE = re.compile(r"^\s*(?:[,;:)\]\}]|[+*/=<>≤≥≈])")
SINGLE_DOLLAR_RE = re.compile(r"(?<!\\)\$(?!\$)")


def visible_text(s: str) -> str:
    return LATENT_TAG_RE.sub("", s).strip()


def is_blank_line(line: str) -> bool:
    return visible_text(line) == ""


def is_bullet_line(line: str) -> bool:
    return BULLET_RE.match(visible_text(line)) is not None


def is_final_answer_line(line: str) -> bool:
    return FINAL_ANSWER_RE.match(visible_text(line)) is not None


def has_odd_display_dollar(line: str) -> bool:
    # $$ ... $$ 가 같은 줄에 닫히면 짝수, $$ 로 열리거나 닫히는 줄이면 홀수다.
    return visible_text(line).count("$$") % 2 == 1


def is_single_line_display_math(line: str) -> bool:
    v = visible_text(line)
    if r"\[" in v and r"\]" in v:
        return True
    if v.count("$$") >= 2 and v.count("$$") % 2 == 0:
        return True
    return False


def is_open_display_math(line: str) -> bool:
    v = visible_text(line)
    if v == r"\[":
        return True
    if has_odd_display_dollar(v):
        return True
    if BEGIN_ENV_RE.search(v):
        return True
    if r"\[" in v and r"\]" not in v:
        return True
    return False


def is_close_display_math(line: str) -> bool:
    v = visible_text(line)
    if v == r"\]":
        return True
    if has_odd_display_dollar(v):
        return True
    if END_ENV_RE.search(v):
        return True
    if r"\]" in v and r"\[" not in v:
        return True
    return False


def contains_display_math(text: str) -> bool:
    v = visible_text(text)
    return (
        r"\[" in v
        or r"\]" in v
        or "$$" in v
        or BEGIN_ENV_RE.search(v) is not None
        or END_ENV_RE.search(v) is not None
    )


def contains_calculation(text: str) -> bool:
    return CALCULATION_RE.search(visible_text(text)) is not None


def starts_with_discourse_cue(text: str) -> bool:
    return DISCOURSE_CUE_RE.match(visible_text(text)) is not None


def is_weak_fragment(text: str) -> bool:
    s = visible_text(text)
    return len(s) < 12 or WEAK_FRAGMENT_RE.match(s) is not None


def is_meaningful_short_step(text: str) -> bool:
    s = visible_text(text)
    return is_final_answer_line(s) or contains_display_math(s) or contains_calculation(s)


def normalize_block(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        out.append(line.rstrip())
    return "\n".join(out).strip()


def split_sentences_rough(text: str) -> List[str]:
    parts = SENTENCE_BOUNDARY_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def has_unclosed_inline_math(text: str) -> bool:
    r"""
    display math는 tokenize 단계에서 보호한다. 여기서는 \(...\) 또는 $...$ 가 줄을
    넘어가는 경우 newline split을 막기 위한 보수적 체크만 한다.
    """
    v = visible_text(text)
    if v.count(r"\(") > v.count(r"\)"):
        return True
    # $$ 는 display delimiter로 따로 처리하므로 single $만 본다.
    return len(SINGLE_DOLLAR_RE.findall(v)) % 2 == 1


def looks_like_complete_line(line: str) -> bool:
    s = visible_text(line)
    if not s:
        return False
    if is_final_answer_line(s):
        return True
    if contains_calculation(s):
        return True
    if re.search(r"[.!?][\"')\]]*$", s):
        return True
    return False


def is_equation_like_line(line: str) -> bool:
    s = visible_text(line)
    if not contains_calculation(s):
        return False
    # 설명문이 섞인 줄은 equation-only로 보지 않는다.
    words = re.findall(r"[A-Za-z]+", s)
    prose_words = [w for w in words if len(w) > 1 and not w.startswith("frac")]
    return len(prose_words) <= 2


def starts_independent_step(line: str) -> bool:
    s = visible_text(line)
    if not s:
        return False
    if is_final_answer_line(s) or is_bullet_line(s):
        return True
    if starts_with_discourse_cue(s):
        return True
    if is_equation_like_line(s):
        return True
    # 대문자/숫자로 시작하는 새 줄은 수학 풀이 데이터에서 독립 step일 가능성이 높다.
    if re.match(r"^\s*(?:[A-Z]|\d|\$|\\\[|\\\(|[-+]?\d)", s):
        return True
    return False


def is_continuation_start(line: str) -> bool:
    s = visible_text(line)
    if not s:
        return False
    # lower-case 접속/관계사로 시작하면 줄바꿈 래핑일 가능성이 높다.
    if LOWERCASE_CONTINUATION_RE.match(s):
        return True
    # 기호로 시작하는 줄은 직전 수식/문장의 continuation일 가능성이 높다.
    if SYMBOL_CONTINUATION_RE.match(s):
        return True
    return False


def should_split_prose_newline(current_lines: List[str], next_line: str) -> bool:
    """
    display math 밖의 newline 중에서 '새 reasoning step'으로 보이는 newline만 자른다.
    """
    if not current_lines:
        return False

    prev_line = current_lines[-1]
    prev = visible_text(prev_line)
    nxt = visible_text(next_line)
    cur_text = normalize_block("\n".join(current_lines))

    if not prev or not nxt:
        return True

    if has_unclosed_inline_math(cur_text):
        return False

    if is_final_answer_line(prev) or is_final_answer_line(nxt):
        return True

    if is_bullet_line(prev) or is_bullet_line(nxt):
        return True

    if is_continuation_start(nxt):
        return False

    # ':' 또는 ',' 뒤의 수식/설명은 보통 같은 step에 붙인다.
    if prev.endswith((":", ",")) and not contains_calculation(prev):
        return False

    # 설명문 다음에 bare equation만 오는 경우는 설명+수식을 한 step으로 유지한다.
    if is_equation_like_line(nxt) and not contains_calculation(prev):
        return False

    if starts_with_discourse_cue(nxt) and len(visible_text(cur_text)) >= MIN_CHARS:
        return True

    if looks_like_complete_line(prev) and starts_independent_step(nxt):
        return True

    if len(visible_text(cur_text + "\n" + nxt)) > MAX_CHARS and starts_independent_step(nxt):
        return True

    return False


def split_prose_on_step_newlines(text: str) -> List[str]:
    """
    prose token 내부의 newline을 추가로 분리한다.
    단, display math token 내부 newline은 tokenize_block_lines에서 이미 보호된다.
    """
    lines = [line.rstrip() for line in text.splitlines() if visible_text(line)]
    if len(lines) <= 1:
        return [normalize_block(text)] if visible_text(text) else []

    out: List[str] = []
    cur: List[str] = []

    for line in lines:
        if not cur:
            cur = [line]
            continue

        if should_split_prose_newline(cur, line):
            block = normalize_block("\n".join(cur))
            if visible_text(block):
                out.append(block)
            cur = [line]
        else:
            cur.append(line)

    if cur:
        block = normalize_block("\n".join(cur))
        if visible_text(block):
            out.append(block)

    return out


def split_prose_by_sentence_cues(text: str) -> List[str]:
    sents = split_sentences_rough(text)
    if len(sents) <= 1:
        return [text]

    out: List[str] = []
    cur: List[str] = []

    for sent in sents:
        if not cur:
            cur = [sent]
            continue

        candidate = " ".join(cur + [sent]).strip()
        should_split = False

        if starts_with_discourse_cue(sent):
            should_split = True
        elif len(candidate) > MAX_CHARS:
            should_split = True
        elif len(cur) >= MAX_SENTENCES_PER_STEP:
            should_split = True

        if should_split:
            out.append(" ".join(cur).strip())
            cur = [sent]
        else:
            cur.append(sent)

    if cur:
        out.append(" ".join(cur).strip())

    return out


def split_prose_semantically(text: str) -> List[str]:
    """
    prose-only block을 semantic cue + step-like newline 기준으로 분리한다.
    """
    text = normalize_block(text)
    if not text:
        return []

    if is_bullet_line(text) or is_final_answer_line(text):
        return [text]

    out: List[str] = []
    for line_block in split_prose_on_step_newlines(text):
        out.extend(split_prose_by_sentence_cues(line_block))

    return [x for x in out if visible_text(x)]


def tokenize_block_lines(block: str) -> List[Tuple[str, str]]:
    """
    block을 ('prose', text) / ('math', text) / ('bullet', text) / ('answer', text) 단위로 토큰화한다.
    display math 내부 newline은 절대 prose newline split 대상이 되지 않게 보호한다.
    """
    lines = block.splitlines(keepends=True)
    tokens: List[Tuple[str, str]] = []

    prose_buf: List[str] = []
    math_buf: List[str] = []
    in_math = False

    def flush_prose() -> None:
        nonlocal prose_buf
        if prose_buf and "".join(prose_buf).strip():
            tokens.append(("prose", normalize_block("".join(prose_buf))))
            prose_buf = []

    def flush_math() -> None:
        nonlocal math_buf
        if math_buf and "".join(math_buf).strip():
            tokens.append(("math", normalize_block("".join(math_buf))))
            math_buf = []

    for line in lines:
        if is_blank_line(line) and not in_math:
            flush_prose()
            continue

        if not in_math and is_final_answer_line(line):
            flush_prose()
            tokens.append(("answer", normalize_block(line)))
            continue

        if not in_math and is_bullet_line(line):
            flush_prose()
            tokens.append(("bullet", normalize_block(line)))
            continue

        if in_math:
            math_buf.append(line)
            if is_close_display_math(line):
                flush_math()
                in_math = False
            continue

        # not in math
        if is_single_line_display_math(line):
            flush_prose()
            tokens.append(("math", normalize_block(line)))
            continue

        if is_open_display_math(line):
            flush_prose()
            math_buf.append(line)
            in_math = True
            continue

        prose_buf.append(line)

    flush_prose()
    flush_math()
    return tokens


def segment_block_into_atomic_units(block: str) -> List[str]:
    """
    mixed prose+math block을 atomic step으로 분해한다.
    핵심 규칙:
    - prose 다음 display math가 오면 prose+math를 한 step으로 flush
    - display math 다음 새 prose는 새 step 시작
    - prose 내부는 discourse cue와 step-like newline 기준으로 추가 분해
    """
    tokens = tokenize_block_lines(block)
    if not tokens:
        return []

    out: List[str] = []
    prose_parts_waiting: List[str] = []

    def flush_waiting_prose() -> None:
        nonlocal prose_parts_waiting
        if prose_parts_waiting:
            out.extend([p.strip() for p in prose_parts_waiting if p.strip()])
            prose_parts_waiting = []

    for typ, content in tokens:
        if typ in {"bullet", "answer"}:
            flush_waiting_prose()
            out.append(content.strip())
            continue

        if typ == "prose":
            parts = split_prose_semantically(content)
            prose_parts_waiting.extend(parts)
            continue

        if typ == "math":
            # display math는 직전 prose와 붙여서 즉시 step 하나로 확정한다.
            if prose_parts_waiting:
                last = prose_parts_waiting.pop()
                flush_waiting_prose()
                out.append((last.rstrip() + "\n" + content.strip()).strip())
            else:
                out.append(content.strip())
            continue

    flush_waiting_prose()
    return [x for x in out if visible_text(x)]


def merge_small_fragments(blocks: List[str]) -> List[str]:
    out: List[str] = []
    for b in blocks:
        b = normalize_block(b)
        if not b:
            continue

        if not out:
            out.append(b)
            continue

        # 짧은 final answer는 의미 있는 step이다. 다만 직전 chunk가 "The answer is:" 같은
        # fragment라면 둘을 합친다.
        if is_final_answer_line(b):
            prev = visible_text(out[-1])
            if prev.endswith(":") or (len(prev) < MIN_CHARS and not is_meaningful_short_step(prev)):
                out[-1] = out[-1].rstrip() + "\n" + b
            else:
                out.append(b)
            continue

        if is_bullet_line(b):
            out.append(b)
            continue

        if len(visible_text(b)) < MIN_CHARS and is_weak_fragment(b) and not is_meaningful_short_step(b):
            out[-1] = out[-1].rstrip() + " " + b
        else:
            out.append(b)

    return out


def rescue_split_overlong(blocks: List[str]) -> List[str]:
    out: List[str] = []
    for b in blocks:
        if len(visible_text(b)) <= MAX_CHARS:
            out.append(b)
            continue

        finer = segment_block_into_atomic_units(b)
        if len(finer) > 1:
            out.extend(finer)
        else:
            out.append(b)
    return out


def split_top_level_blocks(text: str) -> List[str]:
    r"""
    빈 줄 기준 coarse split도 display math 내부에서는 하지 않는다.
    원래 re.split(r"\n\s*\n", ...)가 math block 내부 blank line까지 자를 수 있는 점을 보완한다.
    """
    lines = text.splitlines(keepends=True)
    blocks: List[str] = []
    buf: List[str] = []
    in_math = False

    def flush_buf() -> None:
        nonlocal buf
        block = normalize_block("".join(buf))
        if visible_text(block):
            blocks.append(block)
        buf = []

    for line in lines:
        if in_math:
            buf.append(line)
            if is_close_display_math(line):
                in_math = False
            continue

        if is_blank_line(line):
            if buf:
                flush_buf()
            continue

        buf.append(line)
        if is_open_display_math(line) and not is_single_line_display_math(line):
            in_math = True

    if buf:
        flush_buf()

    return blocks


def split_atomic_reasoning_steps(text: str) -> List[str]:
    """
    최종 API
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    coarse = split_top_level_blocks(text)

    refined: List[str] = []
    for block in coarse:
        refined.extend(segment_block_into_atomic_units(block))

    refined = merge_small_fragments(refined)
    refined = rescue_split_overlong(refined)

    return [normalize_block(x) for x in refined if visible_text(x)]


def _normalize_newlines_with_offsets(text: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    offsets: List[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\r":
            normalized_chars.append("\n")
            offsets.append(i)
            if i + 1 < len(text) and text[i + 1] == "\n":
                i += 2
            else:
                i += 1
            continue
        normalized_chars.append(ch)
        offsets.append(i)
        i += 1
    return "".join(normalized_chars), offsets


def _step_search_pattern(step: str):
    chunks = re.findall(r"\S+", step)
    if not chunks:
        return None
    return r"\s+".join(re.escape(chunk) for chunk in chunks)


def split_atomic_reasoning_step_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return original-text spans for the same step boundaries used by
    split_atomic_reasoning_steps. The returned substrings preserve the original
    whitespace inside each matched step. If any step cannot be mapped back to
    the original text, return [] so callers can use a conservative fallback.
    """
    normalized_text, offsets = _normalize_newlines_with_offsets(text)
    if not normalized_text:
        return []

    steps = split_atomic_reasoning_steps(normalized_text)
    normalized_spans: List[Tuple[int, int]] = []
    cursor = 0
    for step in steps:
        pattern = _step_search_pattern(step)
        if pattern is None:
            return []

        match = re.search(pattern, normalized_text[cursor:], flags=re.DOTALL)
        if match is None:
            return []

        norm_start = cursor + match.start()
        norm_end = cursor + match.end()
        if norm_start >= norm_end:
            return []

        normalized_spans.append((norm_start, norm_end))
        cursor = norm_end

    spans: List[Tuple[int, int]] = []
    for idx, (norm_start, norm_end) in enumerate(normalized_spans):
        span_norm_start = 0 if idx == 0 else norm_start
        if idx + 1 < len(normalized_spans):
            span_norm_end = normalized_spans[idx + 1][0]
        else:
            span_norm_end = len(normalized_text)

        if span_norm_start >= span_norm_end:
            return []

        original_start = offsets[span_norm_start]
        if span_norm_end >= len(normalized_text):
            original_end = len(text)
        else:
            original_end = offsets[span_norm_end]
        spans.append((original_start, original_end))

    return spans


# import re
# from typing import List


# MIN_CHARS = 24
# MAX_CHARS = 260
# MAX_SENTENCES_PER_STEP = 2

# LATENT_TAG_RE = re.compile(r"<\|[^|]+?\|>")

# BEGIN_ENV_RE = re.compile(
#     r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
# )
# END_ENV_RE = re.compile(
#     r"\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
# )

# BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\.\)]\s+)")

# SENTENCE_BOUNDARY_RE = re.compile(
#     r'(?<=[.!?])\s+(?=(?:["\']?[A-Z]|Let\b|Suppose\b|Assume\b|If\b|Then\b|Thus\b|Therefore\b|Hence\b|So\b|Now\b|Next\b|Finally\b|The\b|We\b|Since\b|Because\b|Subtracting\b|Adding\b|Dividing\b|Multiplying\b|Simplifying\b|Rearranging\b|Substituting\b|Applying\b|Using\b|Calculating\b|Computing\b|Testing\b|Checking\b|Conclusion\b))'
# )

# DISCOURSE_CUE_RE = re.compile(
#     r"""(?ix)
#     ^
#     \s*
#     (
#         let(?:'s)?\b
#         |suppose\b
#         |assume\b
#         |if\b
#         |then\b
#         |thus\b
#         |therefore\b
#         |hence\b
#         |so\b
#         |now\b
#         |next\b
#         |finally\b
#         |first\b
#         |second\b
#         |third\b
#         |conclusion\b
#         |the\ expression\b
#         |the\ difference\b
#         |the\ equation\b
#         |the\ inequality\b
#         |the\ problem\ requires\b
#         |we\ have\b
#         |we\ get\b
#         |this\ means\b
#         |this\ gives\b
#         |substituting\b
#         |subtracting\b
#         |adding\b
#         |dividing\b
#         |multiplying\b
#         |simplifying\b
#         |rearranging\b
#         |solving\b
#         |applying\b
#         |using\b
#         |calculate\b
#         |calculating\b
#         |compute\b
#         |computing\b
#         |considering\b
#         |testing\b
#         |checking\b
#         |observe\b
#         |note\ that\b
#         |apply\b
#         |applying\b
#         |substitute\b
#         |substituting\b
#         |using\b
#         |calculate\b
#         |calculating\b
#         |compute\b
#         |computing\b
#         |rewrite\b
#         |rewriting\b
#         |factor\b
#         |factoring\b
#         |expand\b
#         |expanding\b
#         |cancel\b
#         |canceling\b
#         |equate\b
#         |comparing\b
#         |from\ this\b
#         |it\ follows\ that\b
#         |note\ that\b
#         |observe\ that\b
#     )
#     \b
#     """
# )

# WEAK_FRAGMENT_RE = re.compile(
#     r"""(?ix)
#     ^
#     \s*
#     (
#         thus[,:\s]*|
#         therefore[,:\s]*|
#         hence[,:\s]*|
#         so[,:\s]*|
#         now[,:\s]*|
#         next[,:\s]*|
#         finally[,:\s]*|
#         then[,:\s]*|
#         and[,:\s]*|
#         but[,:\s]*|
#         or[,:\s]*|
#         because[,:\s]*
#     )
#     $
#     """
# )


# def visible_text(s: str) -> str:
#     return LATENT_TAG_RE.sub("", s).strip()


# def is_blank_line(line: str) -> bool:
#     return visible_text(line) == ""


# def is_bullet_line(line: str) -> bool:
#     return BULLET_RE.match(visible_text(line)) is not None


# def is_single_line_display_math(line: str) -> bool:
#     v = visible_text(line)
#     if r"\[" in v and r"\]" in v:
#         return True
#     if v.count("$$") >= 2:
#         return True
#     return False


# def is_open_display_math(line: str) -> bool:
#     v = visible_text(line)
#     if v == r"\[":
#         return True
#     if v == "$$":
#         return True
#     if BEGIN_ENV_RE.search(v):
#         return True
#     if r"\[" in v and r"\]" not in v:
#         return True
#     return False


# def is_close_display_math(line: str) -> bool:
#     v = visible_text(line)
#     if v == r"\]":
#         return True
#     if v == "$$":
#         return True
#     if END_ENV_RE.search(v):
#         return True
#     if r"\]" in v and r"\[" not in v:
#         return True
#     return False


# def contains_display_math(text: str) -> bool:
#     v = visible_text(text)
#     return (
#         r"\[" in v
#         or r"\]" in v
#         or "$$" in v
#         or BEGIN_ENV_RE.search(v) is not None
#     )


# def starts_with_discourse_cue(text: str) -> bool:
#     return DISCOURSE_CUE_RE.match(visible_text(text)) is not None


# def is_weak_fragment(text: str) -> bool:
#     s = visible_text(text)
#     return len(s) < 12 or WEAK_FRAGMENT_RE.match(s) is not None


# def normalize_block(text: str) -> str:
#     lines = text.splitlines()
#     out = []
#     for line in lines:
#         out.append(line.rstrip())
#     return "\n".join(out).strip()


# def split_sentences_rough(text: str) -> List[str]:
#     parts = SENTENCE_BOUNDARY_RE.split(text.strip())
#     return [p.strip() for p in parts if p.strip()]


# def split_prose_semantically(text: str) -> List[str]:
#     """
#     prose-only block을 semantic cue 기준으로 분리
#     """
#     text = normalize_block(text)
#     if not text:
#         return []

#     if is_bullet_line(text):
#         return [text]

#     sents = split_sentences_rough(text)
#     if len(sents) <= 1:
#         return [text]

#     out = []
#     cur = []

#     for sent in sents:
#         if not cur:
#             cur = [sent]
#             continue

#         candidate = " ".join(cur + [sent]).strip()
#         should_split = False

#         if starts_with_discourse_cue(sent):
#             should_split = True
#         elif len(candidate) > MAX_CHARS:
#             should_split = True
#         elif len(cur) >= MAX_SENTENCES_PER_STEP:
#             should_split = True

#         if should_split:
#             out.append(" ".join(cur).strip())
#             cur = [sent]
#         else:
#             cur.append(sent)

#     if cur:
#         out.append(" ".join(cur).strip())

#     return out


# def tokenize_block_lines(block: str) -> List[tuple]:
#     """
#     block을 ('prose', text) / ('math', text) / ('bullet', text) 단위로 토큰화
#     """
#     lines = block.splitlines(keepends=True)
#     tokens = []

#     prose_buf = []
#     math_buf = []
#     in_math = False

#     def flush_prose():
#         nonlocal prose_buf
#         if prose_buf and "".join(prose_buf).strip():
#             tokens.append(("prose", normalize_block("".join(prose_buf))))
#             prose_buf = []

#     def flush_math():
#         nonlocal math_buf
#         if math_buf and "".join(math_buf).strip():
#             tokens.append(("math", normalize_block("".join(math_buf))))
#             math_buf = []

#     for line in lines:
#         if is_blank_line(line) and not in_math:
#             flush_prose()
#             continue

#         if not in_math and is_bullet_line(line):
#             flush_prose()
#             tokens.append(("bullet", normalize_block(line)))
#             continue

#         if in_math:
#             math_buf.append(line)
#             if is_close_display_math(line):
#                 flush_math()
#                 in_math = False
#             continue

#         # not in math
#         if is_single_line_display_math(line):
#             flush_prose()
#             tokens.append(("math", normalize_block(line)))
#             continue

#         if is_open_display_math(line):
#             flush_prose()
#             math_buf.append(line)
#             in_math = True
#             continue

#         prose_buf.append(line)

#     flush_prose()
#     flush_math()
#     return tokens


# def segment_block_into_atomic_units(block: str) -> List[str]:
#     """
#     mixed prose+math block을 더 잘게 atomic step으로 분해.
#     핵심 규칙:
#     - prose 다음에 math가 오면 prose+math를 한 step으로 flush
#     - math 다음 새 prose는 새 step 시작
#     - prose 내부는 discourse cue 기준으로 추가 분해
#     """
#     tokens = tokenize_block_lines(block)
#     if not tokens:
#         return []

#     out = []
#     prose_parts_waiting = []

#     def flush_waiting_prose():
#         nonlocal prose_parts_waiting
#         if prose_parts_waiting:
#             out.extend([p.strip() for p in prose_parts_waiting if p.strip()])
#             prose_parts_waiting = []

#     for typ, content in tokens:
#         if typ == "bullet":
#             flush_waiting_prose()
#             out.append(content.strip())
#             continue

#         if typ == "prose":
#             parts = split_prose_semantically(content)
#             prose_parts_waiting.extend(parts)
#             continue

#         if typ == "math":
#             # math는 직전 prose와 붙여서 즉시 step 하나로 확정
#             if prose_parts_waiting:
#                 last = prose_parts_waiting.pop()
#                 flush_waiting_prose()
#                 out.append((last.rstrip() + "\n" + content.strip()).strip())
#             else:
#                 out.append(content.strip())
#             continue

#     flush_waiting_prose()
#     return [x for x in out if visible_text(x)]


# def merge_small_fragments(blocks: List[str]) -> List[str]:
#     out = []
#     for b in blocks:
#         b = normalize_block(b)
#         if not b:
#             continue

#         if not out:
#             out.append(b)
#             continue

#         if is_bullet_line(b):
#             out.append(b)
#             continue

#         if len(visible_text(b)) < MIN_CHARS and is_weak_fragment(b):
#             out[-1] = out[-1].rstrip() + " " + b
#         else:
#             out.append(b)

#     return out


# def rescue_split_overlong(blocks: List[str]) -> List[str]:
#     out = []
#     for b in blocks:
#         if len(visible_text(b)) <= MAX_CHARS:
#             out.append(b)
#             continue

#         finer = segment_block_into_atomic_units(b)
#         if len(finer) > 1:
#             out.extend(finer)
#         else:
#             out.append(b)
#     return out


# def split_atomic_reasoning_steps(text: str) -> List[str]:
#     """
#     최종 API
#     """
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     coarse = [normalize_block(x) for x in re.split(r"\n\s*\n", text) if visible_text(x)]

#     refined = []
#     for block in coarse:
#         refined.extend(segment_block_into_atomic_units(block))

#     refined = merge_small_fragments(refined)
#     refined = rescue_split_overlong(refined)

#     return [normalize_block(x) for x in refined if visible_text(x)]
