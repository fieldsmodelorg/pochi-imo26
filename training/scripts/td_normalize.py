#!/usr/bin/env python3
"""Normalize raw L1 datasets -> unified L2 structured messages (tokenizer-independent).

L2 target = OpenAI-style messages that DeepSeek-V4's encode_messages() consumes:
  messages: [{role, content, reasoning_content?, tool_calls?, loss?}], tools: [...]|None

Sources:
- nemotron-math-v3: already OpenAI-ish (reasoning_content separate, OpenAI tool_calls,
  `tool` role, `tools` column). We only assemble a system message carrying tools (TIR).
- nemotron-cascade-2: content is pre-rendered in a Nemotron inline format. We reverse-parse:
    * assistant: `<think>reasoning</think>rest`  -> reasoning_content + (content / tool_calls)
    * math_tool: system `<tools><function>..` -> OpenAI tool schema;
                 assistant `<tool_call><function=NAME><parameter=P>..</parameter>..` -> tool_calls;
                 user `<tool_response>..</tool_response>` -> role:"tool"
    * terminal_agent: assistant emits JSON actions as content; malformed turns (action left
      inside <think>, i.e. empty after </think>) are followed by an agent "parsing error";
      such turns are NOT executed -> marked loss=false (kept for context, excluded from target).

This module is pure transformation; correctness is validated by round-tripping through the
official encoding_dsv4.encode_messages (see td_validate()).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ----- think split -----------------------------------------------------------

def split_think(content: str) -> Tuple[str, str]:
    """Return (reasoning_content, rest). reasoning is text inside the first <think>..</think>."""
    if content.startswith("<think>"):
        end = content.find("</think>")
        if end != -1:
            return content[len("<think>"):end], content[end + len("</think>"):]
        # unterminated <think>: whole thing is reasoning, no post-think content
        return content[len("<think>"):], ""
    return "", content


# ----- cascade math_tool parsing --------------------------------------------

def parse_cascade_tools(system_content: str) -> List[Dict[str, Any]]:
    """Parse Nemotron `<tools><function>..` system block into OpenAI tool schema list."""
    tools = []
    for fb in re.findall(r"<function>(.*?)</function>", system_content, re.DOTALL):
        nm = re.search(r"<name>(.*?)</name>", fb, re.DOTALL)
        if not nm:
            continue
        name = nm.group(1).strip()
        dm = re.search(r"<description>(.*?)</description>", fb, re.DOTALL)
        desc = dm.group(1).strip() if dm else ""
        props: Dict[str, Any] = {}
        params_block = re.search(r"<parameters>(.*?)</parameters>", fb, re.DOTALL)
        req: List[str] = []
        if params_block:
            pblk = params_block.group(1)
            for pb in re.findall(r"<parameter>(.*?)</parameter>", pblk, re.DOTALL):
                pn = re.search(r"<name>(.*?)</name>", pb, re.DOTALL)
                if not pn:
                    continue
                pt = re.search(r"<type>(.*?)</type>", pb, re.DOTALL)
                pd = re.search(r"<description>(.*?)</description>", pb, re.DOTALL)
                props[pn.group(1).strip()] = {
                    "type": (pt.group(1).strip() if pt else "string"),
                    "description": (pd.group(1).strip() if pd else ""),
                }
            rq = re.search(r"<required>(.*?)</required>", pblk, re.DOTALL)
            if rq:
                try:
                    req = json.loads(rq.group(1).strip())
                except Exception:
                    req = []
        tools.append({
            "type": "function",
            "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req},
            },
        })
    return tools


def parse_cascade_tool_calls(rest: str) -> Tuple[str, List[Dict[str, Any]]]:
    """From post-think text, split natural-language content and parse <tool_call> blocks."""
    idx = rest.find("<tool_call>")
    pre = (rest if idx == -1 else rest[:idx]).strip()
    tcs = []
    for tcb in re.findall(r"<tool_call>(.*?)</tool_call>", rest, re.DOTALL):
        nm = re.search(r"<function=([^>\n]+)>", tcb)
        if not nm:
            continue
        name = nm.group(1).strip()
        args: Dict[str, Any] = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", tcb, re.DOTALL):
            args[pm.group(1).strip()] = pm.group(2)
        tcs.append({"type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(args, ensure_ascii=False)}})
    return pre, tcs


# ----- per-source normalizers ------------------------------------------------

def normalize_cascade(domain: str, raw_msgs: List[Dict[str, Any]]
                      ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Return (messages, tools)."""
    tools = None
    out: List[Dict[str, Any]] = []

    if domain == "math_tool":
        tools = parse_cascade_tools(raw_msgs[0]["content"])
        out.append({"role": "system", "content": ""})  # tools attached separately
        body = raw_msgs[1:]
    else:
        # system + user + assistant(+...) ; keep system content verbatim
        out.append({"role": "system", "content": raw_msgs[0]["content"]})
        body = raw_msgs[1:]

    for i, m in enumerate(body):
        role = m["role"]
        c = m["content"]
        if role == "user":
            tr = re.search(r"<tool_response>\n?(.*?)\n?</tool_response>", c, re.DOTALL)
            if domain == "math_tool" and tr:
                out.append({"role": "tool", "content": tr.group(1)})
            else:
                out.append({"role": "user", "content": c})
        elif role == "assistant":
            rc, rest = split_think(c)
            closed = "</think>" in c
            if domain == "math_tool":
                pre, tcs = parse_cascade_tool_calls(rest)
                am: Dict[str, Any] = {"role": "assistant", "reasoning_content": rc, "content": pre}
                if tcs:
                    am["tool_calls"] = tcs
                out.append(am)
            else:
                am = {"role": "assistant", "reasoning_content": rc, "content": rest}
                # terminal malformed-turn detection: action left inside <think>
                if domain == "terminal_agent" and closed and rest.strip() == "":
                    nxt = body[i + 1]["content"] if i + 1 < len(body) else ""
                    if ("No valid JSON" in nxt) or ("parsing error" in nxt.lower()):
                        am["loss"] = False  # not executed; exclude from training target
                out.append(am)
        else:
            out.append({"role": role, "content": c})
    return out, (tools or None)


def normalize_v3(raw_msgs: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]]
                 ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """v3 messages are already OpenAI-style; just attach a system msg carrying tools (TIR)."""
    msgs = [dict(m) for m in raw_msgs]
    if tools:
        if msgs and msgs[0].get("role") == "system":
            return msgs, tools
        return [{"role": "system", "content": ""}] + msgs, tools
    return msgs, None


# ----- record builder --------------------------------------------------------

def n_assistant_turns(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")
