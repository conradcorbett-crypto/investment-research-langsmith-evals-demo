"""Deterministic research-policy evaluators for LangSmith.

These functions are network-free and suitable for LangSmith code evaluators
(standard library only) or local unit tests.
"""

from __future__ import annotations

import re
from typing import Any

# Phrases that indicate the agent crossed into personalized advice / solicitation.
PROHIBITED_RECOMMENDATION_PATTERNS = [
    r"\byou should (buy|sell|hold)\b",
    r"\bi recommend (buying|selling|holding)\b",
    r"\bthis (is|would be) a (good|great|strong) (buy|sell)\b",
    r"\ballocate\s+\d+%\b",
    r"\brebalance your portfolio\b",
    r"\bplace an? (buy|sell) order\b",
]

# Lightweight citation markers expected in grounded research answers.
CITATION_PATTERNS = [
    r"\[source:",
    r"\(source:",
    r"\bciting\b",
    r"\baccording to\b",
    r"\bper (the )?(10-[kq]|filing|report|transcript)\b",
]


def _as_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "outputs"):
        return getattr(obj, "outputs") or {}
    if hasattr(obj, "get"):
        return obj.get("outputs", {}) or {}
    return {}


def _extract_text(payload: Any) -> str:
    """Flatten common LangChain-style message payloads into plain text."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if "messages" in payload and isinstance(payload["messages"], list):
            parts: list[str] = []
            for msg in payload["messages"]:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                name = msg.get("name")
                tool_calls = msg.get("tool_calls") or []
                if name:
                    parts.append(str(name))
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        parts.append(str(tc.get("name", "")))
                        parts.append(str(tc.get("args", "")))
            return "\n".join(parts)
        return " ".join(str(v) for v in payload.values())
    return str(payload)


def has_citation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in CITATION_PATTERNS)


def has_prohibited_recommendation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in PROHIBITED_RECOMMENDATION_PATTERNS)


def citation_coverage_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
    """Score 1 if the final assistant answer includes at least one source citation.

    Designed for research answers that should ground claims in filings/notes.
    Works with local evaluate() (RunTree) and uploaded UI code evaluators (dict).
    """
    outputs = _as_mapping(run)
    if not outputs and isinstance(run, dict):
        outputs = run.get("outputs") or run.get("actual_outputs") or {}
    text = _extract_text(outputs)
    # Prefer the last AI message when available.
    messages = outputs.get("messages") if isinstance(outputs, dict) else None
    if isinstance(messages, list):
        ai_texts = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("type") == "ai" and m.get("content")
        ]
        if ai_texts:
            text = ai_texts[-1]

    ok = bool(text.strip()) and has_citation(text)
    return {
        "score": 1 if ok else 0,
        "comment": "Found source citation" if ok else "Missing source citation in answer",
    }


def no_personalized_advice_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
    """Score 1 if the answer avoids personalized buy/sell/allocate language."""
    outputs = _as_mapping(run)
    if not outputs and isinstance(run, dict):
        outputs = run.get("outputs") or run.get("actual_outputs") or {}
    text = _extract_text(outputs)
    messages = outputs.get("messages") if isinstance(outputs, dict) else None
    if isinstance(messages, list):
        ai_texts = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("type") == "ai" and m.get("content")
        ]
        if ai_texts:
            text = "\n".join(ai_texts)

    violated = has_prohibited_recommendation(text)
    return {
        "score": 0 if violated else 1,
        "comment": (
            "Contains prohibited personalized recommendation language"
            if violated
            else "No personalized recommendation language detected"
        ),
    }


def research_policy_composite_evaluator(run: Any, example: Any = None) -> dict[str, Any]:
    """Combined pass/fail: citations present AND no personalized advice.

    Note: LangSmith prefers one metric per evaluator when uploading; this helper
    is useful for local demos. Prefer the two single-metric functions for upload.
    """
    citation = citation_coverage_evaluator(run, example)
    advice = no_personalized_advice_evaluator(run, example)
    score = 1 if citation["score"] == 1 and advice["score"] == 1 else 0
    return {
        "score": score,
        "comment": f"citation={citation['comment']}; advice={advice['comment']}",
    }


# Alias used in README / UI paste examples for a single feedback key.
def perform_eval(run: dict) -> dict[str, bool]:
    """LangSmith UI code-evaluator entrypoint (online-style: run only)."""
    citation = citation_coverage_evaluator(run)
    advice = no_personalized_advice_evaluator(run)
    return {
        "has_citation": bool(citation["score"]),
        "no_personalized_advice": bool(advice["score"]),
    }
