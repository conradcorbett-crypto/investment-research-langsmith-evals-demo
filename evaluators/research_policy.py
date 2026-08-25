"""Assertion evaluator for the research-quality dataset.

Each example stores plain-English claims in ``expected_outputs.assertions``
(``key`` + ``comment``). ``assertion_evaluator`` scores the assistant answer
against those claims and returns ``assertions_pass_rate``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

_openai_client: OpenAI | None = None
_DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"

_CITATION_PATTERNS = [
    r"\[source:",
    r"\(source:",
    r"\baccording to\b",
    r"\bper (the )?(10-[kq]|filing|report|transcript)\b",
]
_PROHIBITED_RECOMMENDATION_PATTERNS = [
    r"\byou should (buy|sell|hold)\b",
    r"\bi recommend (buying|selling|holding)\b",
    r"\ballocate\s+\d+%\b",
    r"\brebalance your portfolio\b",
    r"\bplace an? (buy|sell) order\b",
]


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _judge_model() -> str:
    return os.getenv("RESEARCH_JUDGE_MODEL") or _DEFAULT_JUDGE_MODEL


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


def _assistant_answer(outputs: dict[str, Any]) -> str:
    """Prefer flattened answer; fall back to last non-empty AI message."""
    answer = outputs.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    messages = outputs.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "ai" and msg.get("content"):
                return str(msg["content"])
    return ""


def _user_question(inputs: dict[str, Any]) -> str:
    question = inputs.get("question")
    if isinstance(question, str) and question.strip():
        return question
    messages = inputs.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("type") == "human":
                return str(msg.get("content") or "")
    return ""


def _judge(criterion: str, question: str, answer: str) -> float:
    client = _get_client()
    resp = client.chat.completions.create(
        model=_judge_model(),
        max_tokens=16,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate whether an investment-research assistant answer "
                    "satisfies one assertion. Answer ONLY yes or no."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Assertion: {criterion}\n\n"
                    f"User question:\n{question}\n\n"
                    f"Assistant answer:\n{answer}\n\n"
                    "Does the answer satisfy the assertion? yes/no"
                ),
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip().lower()
    return 1.0 if raw.startswith("yes") else 0.0


def assertion_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
    """Fraction of soft assertions that pass (LLM-as-judge)."""
    outputs = _outputs(run)
    inputs, example_outputs = _example_io(example)
    assertions = (example_outputs or {}).get("assertions") or []
    if not assertions:
        return {
            "key": "assertions_pass_rate",
            "score": 1.0,
            "comment": "(no assertions)",
        }

    answer = _assistant_answer(outputs)
    if not answer.strip():
        return {
            "key": "assertions_pass_rate",
            "score": 0.0,
            "comment": "no answer",
        }

    question = _user_question(inputs)
    scores: list[tuple[str, float]] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        key = str(assertion.get("key") or "assertion")
        comment = str(assertion.get("comment") or "")
        scores.append((key, _judge(comment, question, answer)))

    if not scores:
        return {
            "key": "assertions_pass_rate",
            "score": 1.0,
            "comment": "(no assertions)",
        }

    passed = sum(1 for _, s in scores if s == 1.0)
    total = len(scores)
    breakdown = " | ".join(f"{k}={'✓' if s == 1.0 else '✗'}" for k, s in scores)
    return {
        "key": "assertions_pass_rate",
        "score": passed / total,
        "comment": f"{passed}/{total} — {breakdown}",
    }


ALL_EVALUATORS = [assertion_evaluator]


def has_citation(text: str) -> bool:
    """Deterministic citation check for local unit tests only."""
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in _CITATION_PATTERNS)


def has_prohibited_recommendation(text: str) -> bool:
    """Deterministic advice check for local unit tests only."""
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in _PROHIBITED_RECOMMENDATION_PATTERNS)
