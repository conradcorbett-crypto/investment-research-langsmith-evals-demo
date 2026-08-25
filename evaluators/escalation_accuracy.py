"""LLM-as-judge for escalation-accuracy using the starter rubric."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

_openai_client: OpenAI | None = None
_DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "judge_prompts" / "escalation_accuracy.md"
)


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _judge_model() -> str:
    return os.getenv("RESEARCH_JUDGE_MODEL") or _DEFAULT_JUDGE_MODEL


def _load_starter_prompt() -> str:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # Drop HTML facilitator comments so the model only sees the rubric.
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def _outputs(run: Any) -> dict[str, Any]:
    if hasattr(run, "outputs"):
        return run.outputs or {}
    if isinstance(run, dict):
        return run.get("outputs") or run.get("actual_outputs") or {}
    return {}


def _example_io(example: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(example, "inputs"):
        return example.inputs or {}, example.outputs or {}
    if isinstance(example, dict):
        return (
            example.get("inputs") or {},
            example.get("outputs")
            or example.get("expected_outputs")
            or example.get("reference_outputs")
            or {},
        )
    return {}, {}


def _dump_for_judge(payload: dict[str, Any]) -> str:
    """Prefer full messages trajectory so the judge can see tool calls."""
    if "messages" in payload:
        return json.dumps(payload.get("messages"), ensure_ascii=False, indent=2)[:8000]
    return json.dumps(payload, ensure_ascii=False, indent=2)[:8000]


def escalation_accuracy_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
    """Score 1 if escalation behavior matches the starter policy rubric."""
    outputs = _outputs(run)
    inputs, _reference = _example_io(example)
    if not outputs:
        return {
            "key": "escalation_accuracy",
            "score": 0,
            "comment": "missing run outputs",
        }

    prompt = _load_starter_prompt()
    user_block = (
        f"{prompt}\n\n"
        f"<input>\n{_dump_for_judge(inputs)}\n</input>\n\n"
        f"<output>\n{_dump_for_judge(outputs)}\n</output>\n\n"
        'Respond with JSON only: {"score": 0 or 1, "reasoning": "brief"}'
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=_judge_model(),
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict escalation-policy judge. "
                    "Return JSON with score (0 or 1) and reasoning."
                ),
            },
            {"role": "user", "content": user_block},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        score = 1 if int(parsed.get("score", 0)) == 1 else 0
        comment = str(parsed.get("reasoning") or "").strip() or raw
    except (json.JSONDecodeError, TypeError, ValueError):
        score = 1 if "score\": 1" in raw or '"score":1' in raw else 0
        comment = raw[:500] or "unparseable judge response"

    return {
        "key": "escalation_accuracy",
        "score": score,
        "comment": comment,
    }
