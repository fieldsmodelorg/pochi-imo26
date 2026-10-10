# Copyright 2026 proof-pilot. Apache-2.0.
"""L3: render one L2 example into (input_ids, labels) for Olmo3Sink SFT.

L2 is tokenizer-independent OpenAI-style messages; L3 renders them with DeepSeek-V4's
official `encode_messages` (vendored sibling `train_core/encoding_dsv4.py`) and tokenizes
with the *transplanted* DeepSeek->Olmo tokenizer, producing a per-token loss mask.

Masking contract (offset-based, exact):
  - We render the conversation message-by-message exactly as `encode_messages` does
    (same merge_tool_messages + sort + per-message render + BOS), tracking each message's
    char span in the single concatenated string.
  - Loss is on ASSISTANT turns only -- their render is `{reasoning}</think>{content}
    {tool_calls}<EOS>`. The `<｜Assistant｜><think>` generation framing is emitted as a
    *transition token* on the PRECEDING message, so it is masked automatically.
  - Assistant turns flagged `{"loss": false}` (malformed terminal-agent actions kept for
    error-recovery context) are masked.
  - We then tokenize the full string once with `return_offsets_mapping=True` and label a
    token iff its char span falls inside a target (loss-bearing assistant) span. Because
    every segment boundary sits on a single-id control token (verified in Phase 0), the
    offsets line up and there is no boundary ambiguity.

Fidelity: the reconstructed string is asserted byte-equal to `encode_messages(...)`; any
divergence is a hard error (fail loud), not a silent mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import encoding_dsv4 as E  # vendored DeepSeek-V4 encoder, same package

IGNORE = -100

# Literal control-token strings. If one appears inside a LOSS-BEARING assistant span it
# tokenizes into a structural control token and corrupts the training target (e.g. a
# literal </think> would prematurely close reasoning, counted as loss) -> such rows are
# dropped. Literals in MASKED context (user/system/tool) are tolerated: they still become
# control tokens but carry no loss, and some prompts legitimately quote <think>/</think>
# in their format instructions (e.g. swe_agentless). Scope = target only.
_CONTROL_LITERALS: tuple[str, ...] = (
    E.bos_token, E.eos_token, "<｜▁pad▁｜>",
    E.USER_SP_TOKEN, E.ASSISTANT_SP_TOKEN, E.LATEST_REMINDER_SP_TOKEN,
    E.thinking_start_token, E.thinking_end_token, E.dsml_token,
    *E.DS_TASK_SP_TOKENS.values(),
)


@dataclass
class Rendered:
    """One tokenized SFT example with an explicit per-token loss mask."""
    input_ids: list[int]
    labels: list[int]  # IGNORE everywhere except assistant target tokens
    n_target: int      # #tokens that carry loss

    def __len__(self) -> int:
        return len(self.input_ids)


def attach_tools(messages: list[dict], tools: Optional[list]) -> list[dict]:
    """Place the tool schema on the system message, where `render_message` reads it."""
    if not tools:
        return messages
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        head["tools"] = tools
        return [head] + messages[1:]
    return [{"role": "system", "content": "", "tools": tools}] + messages


def find_target_control_literals(messages: list[dict]) -> set[str]:
    """Return control-token literals found in LOSS-BEARING assistant turns only.

    These are the rows whose loss target would be corrupted by stray control tokens;
    literals in masked context (user/system/tool) are not flagged. See _CONTROL_LITERALS.
    """
    bad: set[str] = set()

    def scan(s: Optional[str]) -> None:
        if not s:
            return
        for lit in _CONTROL_LITERALS:
            if lit in s:
                bad.add(lit)

    for m in messages:
        if m.get("role") != "assistant" or m.get("loss") is False:
            continue
        scan(m.get("content"))
        scan(m.get("reasoning_content"))
        for tc in (m.get("tool_calls") or []):
            scan((tc.get("function") or {}).get("arguments"))
    return bad


def render_and_mask(
    messages: list[dict],
    tools: Optional[list],
    tokenizer,
    *,
    thinking_mode: str = "thinking",
    drop_thinking: bool = False,
    check_roundtrip: bool = True,
) -> tuple[Optional[Rendered], Optional[str]]:
    """Render + tokenize + mask one example.

    Returns (Rendered, None) on success, or (None, reason) if the row is filtered.
    Raises RuntimeError if our span reconstruction diverges from `encode_messages`.
    """
    msgs = attach_tools(messages, tools)

    bad = find_target_control_literals(msgs)
    if bad:
        return None, f"control-literal:{sorted(bad)}"

    # Replicate encode_messages preprocessing so our per-message spans line up exactly
    # with the single concatenated string the model is trained on.
    merged = E.merge_tool_messages(msgs)
    merged = E.sort_tool_results_by_call_order(merged)

    # tools anywhere -> encode_messages forces drop_thinking False; mirror it.
    eff_drop = drop_thinking and not any(m.get("tools") for m in merged)

    text = E.bos_token  # add_default_bos_token, no context
    target_spans: list[tuple[int, int]] = []
    for idx in range(len(merged)):
        seg = E.render_message(idx, merged, thinking_mode=thinking_mode, drop_thinking=eff_drop)
        start = len(text)
        text += seg
        end = len(text)
        m = merged[idx]
        if m.get("role") == "assistant" and m.get("loss") is not False:
            target_spans.append((start, end))

    if check_roundtrip:
        ref = E.encode_messages(msgs, thinking_mode=thinking_mode, drop_thinking=drop_thinking)
        if text != ref:
            raise RuntimeError("L3 render != encode_messages: offset tracking diverged")

    if not target_spans:
        return None, "no-target"

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    offs = enc["offset_mapping"]
    labels = [IGNORE] * len(ids)

    spans = sorted(target_spans)
    si = 0
    for i, (s, e) in enumerate(offs):
        if s == e:  # zero-width token (not expected w/ add_special_tokens=False)
            continue
        while si < len(spans) and spans[si][1] <= s:
            si += 1
        if si < len(spans) and spans[si][0] <= s < spans[si][1]:
            labels[i] = ids[i]

    n_target = sum(1 for x in labels if x != IGNORE)
    if n_target == 0:
        return None, "no-target-tokens"
    return Rendered(ids, labels, n_target), None
